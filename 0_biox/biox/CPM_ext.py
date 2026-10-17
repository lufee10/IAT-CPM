"""
CPM 建模后的扩展工具。

本模块作为 ``biox.CPM`` 的延续，放置 CPM 模型完成之后的分析函数，例如
Haufe predictive-feature maps、结果矩阵整理和可视化前的数据汇总。

``biox.CPM`` 负责建模、交叉验证和置换检验；``biox.CPM_ext`` 负责解释和
整理这些模型输出。
"""

import numpy as np

from .CPM import fast_partCorr_train


def residualize_features(X, Covariates):
    """
    对特征矩阵 X 的每一列回归协变量，返回残差化后的特征矩阵。

    该函数适合在 CPM 建模后或可视化前使用，例如先从 FC profile 中剔除年龄、
    性别、头动、站点等协变量的线性影响，再做 PCA、GMM、相似性分析或其他
    下游统计。

    计算方式与 ``biox.CPM.fast_partCorr_train()`` 中处理 X 的步骤一致：

    1. 构造包含截距的设计矩阵 ``C = [1, Covariates]``；
    2. 用 Moore-Penrose 伪逆一次性估计所有特征列的回归系数；
    3. 计算 ``X_resid = X - C @ beta_X``。

    注意：
    - 函数内部固定加入截距项；
    - 函数一次性处理全部特征；
    - ``Covariates=None`` 表示只回归截距，此时输出等价于对 X 每列中心化。

    Parameters
    ----------
    X : array-like, shape (n_subjects, n_features)
        待残差化的特征矩阵。每一行是一名被试，每一列是一条 FC edge 或其他特征。
    Covariates : array-like or None, shape (n_subjects, n_Covariates)
        需要控制的协变量矩阵，不包含截距。若传入一维数组，会被视为单个协变量；
        若传入 ``None``，则只控制截距。

    Returns
    -------
    X_resid : np.ndarray, shape (n_subjects, n_features)
        残差化后的特征矩阵，表示 X 中不能被协变量线性解释的部分。
    beta_X : np.ndarray, shape (n_Covariates + 1, n_features)
        每一列特征对应的回归系数。第一行是截距，其余行依次对应
        ``Covariates`` 的各列。

    Raises
    ------
    ValueError
        当 X 不是二维数组，Covariates 维度不合法，样本数不匹配，或输入包含
        NaN/Inf 时抛出。
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape (n_subjects, n_features).")
    if X.shape[1] < 1:
        raise ValueError("X must contain at least one column.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")

    if Covariates is None:
        Covariates = np.empty((X.shape[0], 0), dtype=float)
    else:
        Covariates = np.asarray(Covariates, dtype=float)
        if Covariates.ndim == 1:
            Covariates = Covariates[:, None]
        if Covariates.ndim != 2:
            raise ValueError(
                "Covariates must be None, a 1D array, or a 2D array "
                "with shape (n_subjects, n_Covariates)."
            )
        if Covariates.shape[0] != X.shape[0]:
            raise ValueError("Covariates.shape[0] must match X.shape[0].")
        if not np.all(np.isfinite(Covariates)):
            raise ValueError("Covariates contains NaN or infinite values.")

    # 与 fast_partCorr_train() 保持一致：协变量矩阵前插入一列 1 作为截距。
    C = np.column_stack([np.ones(X.shape[0]), Covariates])

    # 一次性拟合所有特征列的线性模型：X = C @ beta_X。
    C_pinv = np.linalg.pinv(C)
    beta_X = C_pinv @ X

    # 残差即原始特征减去协变量可解释的线性部分。
    X_resid = X - C @ beta_X

    return X_resid, beta_X


def compute_loocv_selected_haufe_chunk(
    X,
    y,
    cov,
    edge_records,
    fold_positions,
    standardize_selected_features=True,
    eps=1e-12,
):
    """
    对一批 LOOCV folds 计算 selected-space Haufe feature。

    这里的 chunk 指一组 fold 编号。主程序可以先把所有 LOOCV folds 分成若干
    小组，再用 joblib 并行调用本函数；每次调用只负责其中一个 chunk。

    重要限制：本函数仅用于外层 ``cv_type='loocv'`` 且 ``n_repeats=1`` 的 CPM
    输出。这里的限制来自 LOOCV 的数据结构，而不是来自 fixed/nested 的区别。
    因此，只要满足上述 LOOCV 条件，``edge_records`` 可以来自：

    - ``biox.CPM.cpm_fixed_params_cv()``；
    - ``biox.CPM.cpm_fixed_params_cv_parallel()``；
    - ``biox.CPM.cpm_nested_cv()``；
    - ``biox.CPM.cpm_nested_cv_parallel()``。

    本函数假设第 k 条 ``edge_record`` 对应第 k 个被试作为 outer test subject。
    若 ``edge_records`` 来自 KFold 或 repeated CV，则一条记录通常对应多个测试
    被试，且记录顺序不再等价于被试编号，本函数不能直接使用。

    本函数不重新训练 CPM，也不重新选边；它只使用主模型已经保存的
    ``edge_idx`` 和 ``coef``，重建该 fold 的训练集预测值，再在当前 fold
    已入模的 selected edges 上计算这些 FC 与预测值的 Haufe 关系。

    注意这里不对全脑所有 FC 计算 Haufe。原因是本函数的目标是
    selected-space Haufe：每个 fold 的解释空间应与该 fold 实际进入预测模型的
    edge 空间一致。若先对全脑 ``Xr_train`` 计算，再把未入模 edge 置零，结果
    在数值上可以得到相同的 selected edge 值，但会多做大量无用矩阵运算。

    Parameters
    ----------
    X : array-like, shape (n_subjects, n_edges)
        当前 subgroup 的 FC edge matrix；行是被试，列是 FC。
    y : array-like, shape (n_subjects,)
        当前 subgroup 的目标变量，例如 IAT 分数。
    cov : array-like, shape (n_subjects, n_covariates)
        当前 subgroup 的协变量矩阵。
    edge_records : list of dict
        ``biox.CPM`` 中 fixed 或 nested CV 函数保存的 fold-level 结果。
        本函数要求它来自外层 LOOCV 且只有 1 次 repeat，因此
        ``len(edge_records) == n_subjects``。每个元素至少包含：

        - ``edge_idx`` : np.ndarray, shape (n_selected_edges,)
          当前 LOOCV fold 中被模型实际使用的 edge 编号。
        - ``coef`` : np.ndarray, shape (n_selected_edges,)
          与 ``edge_idx`` 一一对应的模型系数。
    fold_positions : array-like of int
        当前 worker 需要处理的 LOOCV fold 编号。
    standardize_selected_features : bool, default=True
        是否按主 CPM-Ridge 模型的规则，对 selected FC 使用训练集均值和标准差
        做 z-score。该参数必须与主模型的 ``standardize_features`` 一致。
    eps : float, default=1e-12
        数值稳定性阈值，用于避免除以 0。

    Returns
    -------
    out : dict
        当前 chunk 的累计结果。不同 chunk 的同名数组可以直接相加，最后再除以
        对应计数，得到每条 FC 的平均 Haufe 值。包含：

        - ``n_valid_folds`` : int
          成功计算 Haufe 的 LOOCV fold 数。若某个 fold 没有入模 edge，或重建
          的训练集预测值方差过小，该 fold 会被跳过。
        - ``selected_slope_sum`` : np.ndarray, shape (n_edges,), dtype float64
          每条 FC 在所有有效 folds 中的 selected-space Haufe slope 累加值。
          某条 FC 未被当前 fold 选中时，该 fold 对它贡献 0。
        - ``selected_corr_sum`` : np.ndarray, shape (n_edges,), dtype float64
          每条 FC 在所有有效 folds 中的 selected-space Haufe correlation 累加值。
          correlation 被限制在 [-1, 1]，未被当前 fold 选中的 FC 贡献 0。
        - ``selected_count`` : np.ndarray, shape (n_edges,), dtype int64
          每条 FC 在所有有效 folds 中被模型选中的次数。通常用
          ``selected_slope_sum / selected_count`` 或
          ``selected_corr_sum / selected_count`` 计算“被选中时”的平均 Haufe；
          对 ``selected_count == 0`` 的 FC 应保留为缺失或 0，取决于后续分析目的。
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.float64).ravel()
    cov = np.asarray(cov, dtype=np.float64)
    fold_positions = np.asarray(fold_positions, dtype=int).ravel()

    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape (n_subjects, n_edges).")
    if y.shape[0] != X.shape[0]:
        raise ValueError("y must have the same number of rows as X.")
    if cov.shape[0] != X.shape[0]:
        raise ValueError("cov must have the same number of rows as X.")
    if len(edge_records) != X.shape[0]:
        raise ValueError(
            "compute_loocv_selected_haufe_chunk() is LOOCV-only and expects one edge_record "
            "per subject. edge_records may come from fixed or nested CPM, but the outer CV "
            "must be LOOCV with n_repeats=1."
        )

    n_subjects, n_edges = X.shape
    out = {
        "n_valid_folds": 0,
        "selected_slope_sum": np.zeros(n_edges, dtype=np.float64),
        "selected_corr_sum": np.zeros(n_edges, dtype=np.float64),
        "selected_count": np.zeros(n_edges, dtype=np.int64),
    }

    for fold_pos in fold_positions:
        if fold_pos < 0 or fold_pos >= n_subjects:
            raise IndexError(f"fold_pos out of range: {fold_pos}")

        # LOOCV 专用：第 fold_pos 个被试是当前 fold 的 outer test subject。
        record = edge_records[fold_pos]
        train_mask = np.ones(n_subjects, dtype=bool)
        train_mask[fold_pos] = False
        train_idx = np.flatnonzero(train_mask)

        # biox.CPM 标准 edge_record：edge_idx 与 coef 一一对应。
        sel_idx = np.asarray(record["edge_idx"], dtype=np.int64).ravel()
        coef = np.asarray(record["coef"], dtype=np.float64).ravel()
        if sel_idx.size == 0:
            continue
        if sel_idx.size != coef.size:
            raise ValueError(f"Fold {fold_pos}: edge_idx length != coef length.")

        X_train = np.asarray(X[train_idx, :], dtype=np.float64)
        y_train = y[train_idx]
        cov_train = cov[train_idx, :]

        # 复用 CPM 主模型相同的协变量残差化逻辑。
        _, _, Xr_train, _, _, _ = fast_partCorr_train(
            X_train,
            y_train,
            cov_train,
        )

        # 只取当前 fold 被 CPM 选中、且真正进入预测模型的 FC。
        #
        # 这里不用全脑 Xr_train 直接计算 Haufe，是因为本函数定义的是
        # selected-space Haufe：当前 fold 没有入模的 FC 不参与模型预测，也不应
        # 在该 fold 中贡献 Haufe 值。只在 F_selected 上计算，可以避免每个 fold
        # 都对所有 n_edges 做一次大矩阵乘法，速度和内存都会更友好。
        F_selected = np.asarray(Xr_train[:, sel_idx], dtype=np.float64)
        sd = F_selected.std(axis=0, ddof=0)
        keep = sd > eps
        if not np.any(keep):
            continue

        F_selected = F_selected[:, keep]
        coef = coef[keep]
        sel_idx = sel_idx[keep]
        sd = sd[keep]

        # F_model 是“模型实际看到的 selected feature matrix”。
        # 若主 CPM 建模时 standardize_features=True，Ridge/Lasso/ElasticNet 的 coef
        # 对应的是 z-score 后的特征，因此这里必须用同样的标准化方式重建 yhat。
        # 若主模型没有标准化，则 F_model 与 F_selected 相同。
        if standardize_selected_features:
            F_model = (
                F_selected - F_selected.mean(axis=0, keepdims=True)
            ) / sd.reshape(1, -1)
            # 标准化后的特征均值已经是 0，因此 Haufe 计算中的 demean 结果就是
            # F_model 本身。这里直接复用，避免后面重复减均值。
            F_dm = F_model
        else:
            F_model = F_selected
            F_dm = F_model - F_model.mean(axis=0, keepdims=True)

        # 用保存的 coef 重建当前 LOOCV fold 的训练集预测值。
        # 这里的 yhat_train 不是重新拟合得到的，而是“原模型在训练集上的线性输出”。
        yhat_train = F_model @ coef
        yhat_dm = yhat_train - yhat_train.mean()
        yhat_ss = float(yhat_dm @ yhat_dm)
        if (not np.isfinite(yhat_ss)) or yhat_ss <= eps:
            continue

        # Haufe slope 与 Haufe correlation 只在 selected features 上计算。
        #
        # Haufe slope:
        #   slope_e = Cov(F_e, yhat) / Var(yhat)
        # 它等价于把第 e 条 selected FC 反向回归到模型预测值 yhat 上时的斜率：
        #   F_e ~= slope_e * yhat + intercept
        #
        # Haufe correlation:
        #   corr_e = Corr(F_e, yhat)
        # 它去除了 FC 和 yhat 的量纲影响，范围固定在 [-1, 1]，更适合跨 edge
        # 可视化和比较。
        #
        # 注意：这里使用 F_model，而不是全脑 Xr_train。这样 Haufe 的解释空间与
        # 主模型的 selected feature space 完全一致；未入模 edge 在当前 fold 中
        # 没有预测系数，因此不参与本 fold 的 Haufe 计算。
        numerator = F_dm.T @ yhat_dm
        selected_slope = numerator / yhat_ss
        f_ss = np.sum(F_dm**2, axis=0)
        selected_corr = numerator / np.sqrt(np.maximum(f_ss * yhat_ss, eps))
        selected_corr = np.clip(selected_corr, -1.0, 1.0)

        # 把 selected edge 的结果写回全脑 edge vector 对应位置。
        # 输出数组仍保持 shape=(n_edges,)，便于后续直接转成 214x214 FC 矩阵；
        # 当前 fold 未入模的 edge 不写入，相当于该 fold 对它贡献 0。
        out["selected_slope_sum"][sel_idx] += selected_slope
        out["selected_corr_sum"][sel_idx] += selected_corr
        out["selected_count"][sel_idx] += 1
        out["n_valid_folds"] += 1

    return out

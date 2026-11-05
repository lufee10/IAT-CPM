"""
Connectome-based predictive modeling（CPM）分析工具。

本模块围绕 CPM workflow 提供四类功能：

1. 在训练折内控制协变量，并计算连接边与目标变量的偏相关；
2. 根据 p 值阈值或 Top-K 规则选择 CPM edges；
3. 使用固定超参数 CV 或 nested CV 生成折外预测；
4. 计算预测性能，并通过置换检验评估统计显著性。

所有数据驱动参数均只在训练折估计，再应用于对应测试折，以避免数据泄漏。

``model_type='linear'`` 使用标准 CPM 的正/负 network-strength 汇总；Ridge、
Lasso、ElasticNet、PLS 和 PCR 则保留每条入选边，属于高维扩展 CPM。

使用方法
--------
输入数据应提前整理为数值矩阵：

- ``X``：形状为 ``(n_subjects, n_edges)`` 的连接边矩阵；
- ``y``：形状为 ``(n_subjects,)`` 的连续预测目标；
- ``Covariates``：形状为 ``(n_subjects, n_Covariates)`` 的协变量矩阵；
  若不需要控制协变量，可传入 ``None``。

1. 标准 CPM：正/负 network strength + 普通线性回归

标准 CPM 对训练折内选出的正边和负边分别求和，每个被试最多得到两个特征：
``pos_strength`` 和 ``neg_strength``。此时应使用 ``model_type='linear'``。

Examples
--------
>>> cv_results = cpm_fixed_params_cv(
...     X,
...     y,
...     Covariates,
...     model_type="linear",
...     p_thresh=0.01,
...     n_splits=10,
...     n_repeats=50,
...     random_state=0,
... )
>>> performance = calculate_prediction_performance(
...     y_resid_true_mat=cv_results[3],
...     y_pred_mat=cv_results[2],
... )
>>> edge_records = cv_results[4]

``cv_results[2]`` 是每次 repeat 的折外预测矩阵 ``y_pred_mat``；
``cv_results[3]`` 是与之逐行配对的折特异真实残差 ``y_resid_true_mat``；
``cv_results[4]`` 是统一的 fold-level ``edge_records``。其中 ``pos_idx`` 和
``neg_idx`` 可用于分析 stable edges；对 Ridge、Lasso 和 ElasticNet，
``edge_idx`` 与 ``coef`` 一一对应，此时 ``coef`` 可用于分析单条 edge 的模型系数。

2. 高维扩展 CPM：保留入选边作为模型特征

若 ``model_type`` 为 ``'ridge'``、``'lasso'``、``'elasticnet'``、``'pls'`` 或
``'pcr'``，每条入选边会作为独立特征进入对应预测模型。

>>> cv_results = cpm_fixed_params_cv(
...     X,
...     y,
...     Covariates,
...     model_type="ridge",
...     ridge_alpha=1.0,
...     p_thresh=0.01,
...     standardize_features=True,
... )

3. Nested CV：在外层测试折之外选择模型超参数

当需要从一组候选超参数中选择最优值时，使用 ``cpm_nested_cv()``。内层 CV 只在
当前 outer training set 内运行，outer test fold 不参与调参。

>>> nested_results = cpm_nested_cv(
...     X,
...     y,
...     Covariates,
...     model_type="ridge",
...     param_grid={"ridge_alpha": [0.1, 1.0, 10.0]},
...     n_splits=10,
...     n_repeats=50,
...     inner_n_splits=5,
...     scoring="pearson_r",
...     random_state=0,
... )
>>> nested_performance = calculate_prediction_performance(
...     y_resid_true_mat=nested_results[3],
...     y_pred_mat=nested_results[2],
... )
>>> nested_edge_records = nested_results[4]
>>> selected_params = nested_results[5]

4. 置换检验：先计算真实模型，再生成零分布和 p 值

``cpm_permutation_test()`` 只负责生成置换零分布；真实模型性能需要先由
``cpm_fixed_params_cv()`` 或 ``cpm_nested_cv()`` 得到，再用
``calculate_prediction_performance()`` 计算。

>>> observed_cv = cpm_fixed_params_cv(
...     X,
...     y,
...     Covariates,
...     model_type="linear",
...     p_thresh=0.01,
... )
>>> observed_perf = calculate_prediction_performance(
...     y_resid_true_mat=observed_cv[3],
...     y_pred_mat=observed_cv[2],
... )
>>> observed_score = observed_perf["mean_prediction"]["pearson_r"]
>>> perm = cpm_permutation_test(
...     X,
...     y,
...     Covariates,
...     n_permutations=1000,
...     model_type="linear",
...     p_thresh=0.01,
...     random_state=0,
... )
>>> p_result = calculate_permutation_p_value(
...     observed_score,
...     perm["null_scores"],
...     performance_metric="pearson_r",
... )

建议报告 ``observed_score``、``p_result['p_value']``，并同时查看
``calculate_prediction_performance()`` 返回的 ``r2``、``mae`` 和 ``rmse``。
"""

from itertools import product

import numpy as np
import scipy.stats as stats

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.pipeline import Pipeline


_EPS = 1e-12
_SUPPORTED_MODELS = ("linear", "ridge", "lasso", "elasticnet", "pls", "pcr")
_TUNABLE_PARAMS = {
    "linear": set(),
    "ridge": {"ridge_alpha"},
    "lasso": {"lasso_alpha"},
    "elasticnet": {"elasticnet_alpha", "l1_ratio"},
    "pls": {"pls_components"},
    "pcr": {"pca_components"},
}


def _validate_integer(value, name, minimum=0, allow_none=False):
    """
    验证并规范化模块内部使用的整数参数。

    本函数用于检查具有“整数语义”的配置项，例如交叉验证折数、重复次数、
    选边数量或 PLS/PCA 成分数。它不会直接调用 ``int(value)``，因为 Python
    会静默截断小数，例如 ``int(2.9) == 2``，这会掩盖参数设置错误。这里会先
    判断输入是否为有限数值，并且数值上确实等价于整数；只有通过检查后，才会
    转换为 Python 原生 ``int``。因此，``2``、``2.0`` 和 ``"2"`` 会被接受，
    而 ``2.5``、``NaN``、``Inf`` 会报错。布尔值虽然是 Python 中整数的子类，
    但在这些参数语境中通常表示调用错误，因此也会被拒绝。

    ``allow_none`` 用于保留可选整数参数中的特殊取值 ``None``。对于必须明确
    给出的参数，例如 ``n_splits`` 或 ``n_repeats``，``None`` 是无效输入；
    对于可选上限或可自动推断的参数，例如 ``max_edges_total=None`` 表示
    不限制总边数，``pls_components=None`` 表示稍后自动确定成分数。此时
    ``None`` 有明确含义，应原样返回给后续逻辑解释，而不能转换为 ``0``。

    Parameters
    ----------
    value : object
        待验证的参数值。
    name : str
        参数名称，仅用于生成清晰的错误信息。
    minimum : int, default=0
        允许的最小整数值，包含该边界。
    allow_none : bool, default=False
        是否允许 ``None``。允许时直接返回 ``None``，常用于可选成分数或边数上限。

    Returns
    -------
    int or None
        规范化后的 Python ``int``；若允许且输入为 ``None``，则返回 ``None``。

    Raises
    ------
    ValueError
        输入为不允许的 ``None``、布尔值、非数值、非有限值、非整数数值，
        或小于 ``minimum`` 时抛出。
    """
    if value is None:
        # 某些可选整数参数需要保留 None 语义。例如：
        #   max_edges_total=None 表示不限制总边数；
        #   pls_components=None 表示稍后根据训练折自动确定成分数。
        # 把 None 转成 0 会改变这些参数的含义，因此允许 None 时直接返回。
        if allow_none:
            return None
        raise ValueError(f"{name} must not be None.")

    # bool/np.bool_ 可被 int() 转成 0 或 1，但这通常会掩盖参数传递错误。
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean.")

    try:
        # 先转成 float，便于同时接受 Python/NumPy 数值以及 "2" 这类数字字符串。
        value_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc

    # is_integer() 区分 2.0 与 2.5；isfinite() 排除 NaN 和正负无穷。
    if not np.isfinite(value_float) or not value_float.is_integer():
        raise ValueError(f"{name} must be an integer.")

    value_int = int(value_float)
    # minimum 由调用位置决定，例如 CV 折数要求至少为 2，成分数至少为 1。
    if value_int < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")

    return value_int


def _validate_boolean(value, name):
    """
    验证并规范化模块内部使用的布尔开关。

    本函数用于检查具有“开/关”语义的参数，例如是否标准化特征、是否启用
    fallback。这里不使用 Python 的隐式真值判断，因为 ``bool()`` 不会按字符串
    的字面含义解析真假。例如，``bool("False")`` 的结果是 ``True``，原因是
    任何非空字符串都会被视为真值。如果用户误把 ``standardize_features`` 写成
    ``"False"``，程序反而会开启标准化。为了避免这类隐蔽错误，本函数只接受
    真正的布尔值 ``True``/``False`` 或 NumPy 的 ``np.bool_``。

    Parameters
    ----------
    value : object
        待验证的开关值。
    name : str
        参数名称，仅用于生成错误信息。

    Returns
    -------
    bool
        规范化后的 Python 原生布尔值。

    Raises
    ------
    ValueError
        输入不是 ``bool`` 或 ``np.bool_`` 时抛出。
    """
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean.")
    # 将 np.bool_ 统一转换为 Python bool，保证后续条件判断与序列化行为一致。
    return bool(value)


def _validate_finite_float(value, name):
    """
    验证并规范化模块内部使用的有限浮点参数。

    本函数只负责两件事：第一，确认输入可以转换为数值；第二，确认转换后的数值
    不是 ``NaN`` 或正负无穷。参数的取值范围（例如是否必须非负、是否必须位于
    0 到 1 之间）由调用位置根据具体语义继续检查。

    Parameters
    ----------
    value : object
        待验证的参数值。
    name : str
        参数名称，仅用于生成清晰的错误信息。

    Returns
    -------
    float
        规范化后的 Python ``float``。

    Raises
    ------
    ValueError
        输入不能转换为数值，或转换后不是有限数值时抛出。
    """
    try:
        value_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc

    if not np.isfinite(value_float):
        raise ValueError(f"{name} must be finite.")

    return value_float


def _validate_X_y_Covariates(
    X,
    y,
    Covariates,
    minimum_n_sub=None,
):
    """
    统一检查 X、y 和 Covariates 的基础格式。

    本函数只处理多个 CPM/偏相关函数都会用到的通用输入规则：X 必须是二维数组，
    y 必须是一维目标向量，Covariates 可以是 None、一维或二维数组，并且三者的
    样本数必须一致。这里也统一拒绝 NaN/Inf；本模块不在建模函数内部做缺失值
    插补。

    Parameters
    ----------
    X : array-like, shape (n_sub, n_features)
        自变量矩阵。
    y : array-like, shape (n_sub,)
        连续目标变量。
    Covariates : array-like or None, shape (n_sub, n_Covariates)
        协变量矩阵，不包含截距；传入 None 表示无协变量。
    minimum_n_sub : int or None, default=None
        若给定，则要求样本数至少达到该值。

    Returns
    -------
    X : np.ndarray, shape (n_sub, n_features)
        规范化后的二维浮点数组。
    y : np.ndarray, shape (n_sub,)
        规范化后的一维浮点数组。
    Covariates : np.ndarray, shape (n_sub, n_Covariates)
        规范化后的二维浮点数组；无协变量时为零列数组。
    n_sub : int
        样本数。
    n_columns : int
        X 的列数。

    Raises
    ------
    ValueError
        输入维度、样本数、最小样本数或有限性检查不通过时抛出。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape (n_sub, n_features).")

    if Covariates is None:
        Covariates = np.empty((X.shape[0], 0), dtype=float)
    else:
        Covariates = np.asarray(Covariates, dtype=float)
        if Covariates.ndim == 1:
            Covariates = Covariates[:, None]
        if Covariates.ndim != 2:
            raise ValueError(
                "Covariates must be None, a 1D array, or a 2D array "
                "with shape (n_sub, n_Covariates)."
            )

    if y.shape[0] != X.shape[0]:
        raise ValueError("y length must match X.shape[0].")
    if Covariates.shape[0] != X.shape[0]:
        raise ValueError("Covariates.shape[0] must match X.shape[0].")

    n_sub, n_columns = X.shape
    if minimum_n_sub is not None and n_sub < minimum_n_sub:
        raise ValueError(f"At least {minimum_n_sub} subjects are required.")
    if n_columns < 1:
        raise ValueError("X must contain at least one column.")

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values.")
    if not np.all(np.isfinite(Covariates)):
        raise ValueError("Covariates contains NaN or infinite values.")

    return X, y, Covariates, n_sub, n_columns


def fast_partCorr_train(X, y, Covariates):
    """
    计算训练集中 X 的每一列与 y 之间的偏相关系数，并控制协变量 Covariates。

    Parameters
    ----------
    X : array-like, shape (n_sub, p)
        自变量矩阵，每一列为一个待分析的连接边或其他预测特征。
    y : array-like, shape (n_sub,)
        因变量（一维），例如行为得分。
    Covariates : array-like or None, shape (n_sub, k)
        需要控制的协变量（不包括截距），如年龄、性别、头动参数等。
        传入 None 表示只控制截距。

    Returns
    -------
    r_vals : np.ndarray, shape (p,)
        偏相关系数，r_vals[j] 是 X 的第 j 列与 y 在控制 Covariates 后的相关系数。
    p_vals : np.ndarray, shape (p,)
        对应的双尾 p 值（t 检验）。
    X_resid : np.ndarray, shape (n_sub, p)
        X 的残差矩阵（已剔除 Covariates 的线性影响）。
    y_resid : np.ndarray, shape (n_sub,)
        y 的残差向量（已剔除 Covariates 的线性影响）。
    beta_X : np.ndarray, shape (k+1, p)
        对 X 的每一列拟合的回归系数（第一行对应截距）。
    beta_y : np.ndarray, shape (k+1,)
        对 y 拟合的回归系数。

    Raises
    ------
    ValueError
        输入形状不一致、包含非有限值，或训练样本不足以提供有效残差自由度时抛出。
    """

    # ----- 0. 输入格式统一与完整性检查 -----
    # 通用的 X/y/Covariates 形状、样本轴和有限性检查由辅助函数统一完成。
    X, y, Covariates, n_sub, n_features = _validate_X_y_Covariates(
        X,
        y,
        Covariates,
    )

    # ----- 1. 构建设计矩阵（包含截距） -----
    # 在协变量前面插入一列全 1，对应线性模型中的截距项。
    C = np.column_stack([np.ones(n_sub), Covariates])  # shape: (n_sub, k+1)

    # ----- 2. 利用伪逆高效拟合回归系数 -----
    # 预计算设计矩阵 C 的 Moore-Penrose 伪逆，避免在循环中重复计算
    C_pinv = np.linalg.pinv(C)  # shape: (k+1, n_sub)

    # 对 X 的每一列（特征）拟合线性模型：X = C @ beta_X
    # beta_X 的每一列对应 X 中一列的回归系数（包含截距）
    beta_X = C_pinv @ X  # shape: (k+1, p)

    # 对 y 拟合线性模型：y = C @ beta_y
    # beta_y 是长度为 k+1 的向量（截距 + 各协变量的系数）
    beta_y = C_pinv @ y  # shape: (k+1,)

    # ----- 3. 计算残差（剔除 C 的线性贡献）-----
    # X 的残差：原始 X 减去用 C 预测的部分
    X_resid = X - C @ beta_X  # shape: (n_sub, p)

    # y 的残差：原始 y 减去用 C 预测的部分
    y_resid = y - C @ beta_y  # shape: (n_sub,)

    # ----- 4. 计算残差之间的相关系数（即偏相关系数）-----
    # 分子：X_resid 每一列与 y_resid 的点积（残差已由截距项中心化）。
    num = X_resid.T @ y_resid  # shape: (n_features,)

    # 分母第一项：X_resid 每一列的平方和（L2 范数的平方）
    # 注意：(X_resid ** 2).sum(axis=0) 会生成临时数组；数据很大时可改用 einsum。
    # 但对于大多数应用速度足够；若内存紧张可用 einsum 直接计算平方和。
    ss_X = (X_resid**2).sum(axis=0)  # shape: (n_features,)

    # 分母第二项：y_resid 的平方和（标量）
    ss_y = y_resid @ y_resid  # 标量

    # 分母：sqrt(ss_X * ss_y)  （逐元素乘法，结果 shape (p,)）
    den = np.sqrt(ss_X * ss_y)

    # 仅在分母有效时计算相关。常量特征或常量 y 没有定义相关系数，应返回 NaN，
    # 由 select_edges() 排除，而不是把它们伪装成 r=0 的有效边。
    r_vals = np.divide(
        num,
        den,
        out=np.full(n_features, np.nan, dtype=float),
        where=den > _EPS,
    )
    r_vals = np.clip(r_vals, -1.0, 1.0)

    # ----- 5. 计算 t 统计量及对应的 p 值（双尾检验）-----
    # 设计矩阵 C 的秩（通常为 k+1，除非协变量存在完全共线性）
    rank_C = np.linalg.matrix_rank(C)

    # 偏相关的自由度：样本数 - 设计矩阵的秩 - 1
    df = n_sub - rank_C - 1

    if df <= 0:
        # 若继续返回全 NaN p 值，CPM 会把这一折当作“没有显著边”并输出全零预测，
        # 容易掩盖样本量相对协变量过少这一真正问题，因此直接给出明确错误。
        raise ValueError(
            "Insufficient residual degrees of freedom: "
            f"n_sub={n_sub}, rank(design)={rank_C}, df={df}."
        )

    # 仅为计算 t 值把 |r|=1 裁到浮点安全范围；返回的 r 本身仍保留 [-1, 1]。
    r_for_t = np.clip(r_vals, -1.0 + _EPS, 1.0 - _EPS)
    t = r_for_t * np.sqrt(df / (1 - r_for_t**2))
    # 双尾检验的 p 值：2 * 生存函数（1 - CDF）
    p_vals = 2 * stats.t.sf(np.abs(t), df)

    # 返回结果：偏相关系数、p 值、残差矩阵、残差向量、回归系数。
    return r_vals, p_vals, X_resid, y_resid, beta_X, beta_y


def select_edges(
    r_vals,
    p_vals,
    selection_method="pvalue",
    p_thresh=0.01,
    top_k=None,
    top_k_mode="total",
    max_edges_total=None,
    min_edges_total=0,
    fallback_if_too_few=False,
    max_edges_per_sign=None,
):
    """
    根据 p 值阈值或 Top-K 规则选择连接边，并按 p 值从小到大排序。

    Parameters
    ----------
    r_vals : array-like
        边的相关系数（或其他效应量），用于区分正负方向。
    p_vals : array-like
        每条边对应的 p 值。
    selection_method : {'pvalue', 'topk'}, default='pvalue'
        选边方法。
    p_thresh : float, default=0.01
        ``selection_method='pvalue'`` 时使用的严格阈值，即选择 ``p < p_thresh``。
    top_k : int or None, default=None
        当 selection_method='topk' 时选择的边的数量。
    top_k_mode : {'total', 'per_sign'}, default='total'
        ``total`` 在全部有效边中选择；``per_sign`` 分别从正、负边中各选择。
    max_edges_total : int or None, default=None
        最终返回的最大边数，若超过则截断。
    min_edges_total : int, default=0
        当 fallback_if_too_few=True 时，最少需要返回的边数。
    fallback_if_too_few : bool, default=False
        初始结果不足时，是否回退到全局 p 值最小的 ``min_edges_total`` 条有效边。
    max_edges_per_sign : int or None, default=None
        正边和负边各自最多保留的数量。该限制在 fallback 之后执行。

    Returns
    -------
    idx : np.ndarray
        被选中边的索引数组（已按 p 值升序排列）。
    used_fallback : bool
        是否触发了回退机制。只有 ``fallback_if_too_few=True`` 且初始选边数少于
        ``min_edges_total`` 时才可能为 True；它提示当前 fold 的边选择不是完全
        由原始阈值或 Top-K 规则得到，而是回退到按 p 值补足候选边。

    Raises
    ------
    ValueError
        输入数组形状不一致、r/p 值超出合法范围，或选边参数无效时抛出。

    Notes
    -----
    所有返回索引按 p 值升序排列。NaN/Inf 以及 ``r=0`` 的边不会进入候选集；
    分方向上限和总上限在 fallback 后执行，因此最终边数可能低于
    ``min_edges_total``。
    """
    # ---------- 0. 输入格式统一与参数检查 ----------
    r_vals = np.asarray(r_vals, dtype=float)
    p_vals = np.asarray(p_vals, dtype=float)

    if r_vals.ndim != 1 or p_vals.ndim != 1:
        raise ValueError("r_vals and p_vals must both be 1D arrays.")
    if r_vals.shape != p_vals.shape:
        raise ValueError("r_vals and p_vals must have the same shape.")

    finite_r = r_vals[np.isfinite(r_vals)]
    finite_p = p_vals[np.isfinite(p_vals)]
    if np.any((finite_r < -1) | (finite_r > 1)):
        raise ValueError("Finite r_vals must be between -1 and 1.")
    if np.any((finite_p < 0) | (finite_p > 1)):
        raise ValueError("Finite p_vals must be between 0 and 1.")

    selection_method = str(selection_method).lower()

    if selection_method not in {"pvalue", "topk"}:
        raise ValueError("selection_method must be 'pvalue' or 'topk'.")

    max_edges_total = _validate_integer(
        max_edges_total,
        "max_edges_total",
        allow_none=True,
    )
    min_edges_total = _validate_integer(min_edges_total, "min_edges_total")
    max_edges_per_sign = _validate_integer(
        max_edges_per_sign,
        "max_edges_per_sign",
        allow_none=True,
    )

    fallback_if_too_few = _validate_boolean(
        fallback_if_too_few,
        "fallback_if_too_few",
    )

    if selection_method == "pvalue":
        p_thresh = _validate_finite_float(p_thresh, "p_thresh")
        if not 0 <= p_thresh <= 1:
            raise ValueError("p_thresh must be between 0 and 1.")
    else:
        top_k = _validate_integer(top_k, "top_k", minimum=1)
        top_k_mode = str(top_k_mode).lower()
        if top_k_mode not in {"total", "per_sign"}:
            raise ValueError("top_k_mode must be 'total' or 'per_sign'.")

    # NaN/Inf 的效应量或 p 值不能参与选择；r=0 的边没有正负方向，也无法进入
    # 后续正/负 CPM strength，因此在这里一并排除。
    valid_mask = np.isfinite(r_vals) & np.isfinite(p_vals) & (r_vals != 0)
    valid_idx = np.flatnonzero(valid_mask)

    # 记录当前 fold 是否触发 fallback；该标记会随 CV 结果返回，便于检查哪些
    # fold 的选边来自回退规则。
    used_fallback = False

    # ---------- 1. 根据选择方法筛选边的索引 ----------
    if selection_method == "pvalue":
        # 在有效边中找 p 值低于阈值的边，并稳定地按 p 值升序排列。
        idx = valid_idx[p_vals[valid_idx] < p_thresh]
        idx = idx[np.argsort(p_vals[idx], kind="stable")]

    elif selection_method == "topk":
        if top_k_mode == "total":
            # 全局选择有效边中 p 值最小的 top_k 条。
            sorted_valid = valid_idx[np.argsort(p_vals[valid_idx], kind="stable")]
            idx = sorted_valid[:top_k]

        elif top_k_mode == "per_sign":
            # 分别处理正相关（r > 0）和负相关（r < 0）的边。
            pos_all = valid_idx[r_vals[valid_idx] > 0]
            neg_all = valid_idx[r_vals[valid_idx] < 0]

            # 在各自类内按 p 值升序选前 k 条
            pos_idx = pos_all[np.argsort(p_vals[pos_all], kind="stable")[:top_k]]
            neg_idx = neg_all[np.argsort(p_vals[neg_all], kind="stable")[:top_k]]

            # 合并两类索引，再整体按 p 值排序
            idx = np.concatenate([pos_idx, neg_idx])
            idx = idx[np.argsort(p_vals[idx], kind="stable")]

    # ---------- 2. 边数不足时执行 fallback ----------
    # fallback 仍然只从有效边中选择，不会为了凑数引入 NaN/Inf 或 r=0 的边。
    if fallback_if_too_few and idx.size < min_edges_total:
        used_fallback = True
        sorted_valid = valid_idx[np.argsort(p_vals[valid_idx], kind="stable")]
        idx = sorted_valid[:min_edges_total]

    # ---------- 3. 分方向上限 ----------
    # 该限制在 fallback 后执行，因此最终边数可能低于 min_edges_total。
    if max_edges_per_sign is not None:
        pos_idx = idx[r_vals[idx] > 0][:max_edges_per_sign]
        neg_idx = idx[r_vals[idx] < 0][:max_edges_per_sign]
        idx = np.concatenate([pos_idx, neg_idx])
        idx = idx[np.argsort(p_vals[idx], kind="stable")]

    # ---------- 4. 总边数硬上限 ----------
    # 放在最后执行，确保 fallback 不会突破 max_edges_total。
    if max_edges_total is not None:
        idx = idx[:max_edges_total]

    # 返回索引数组（整数类型）以及是否使用了回退的标记。
    return np.asarray(idx, dtype=int), used_fallback


def _validate_model_params(
    model_type,
    ridge_alpha=1.0,
    lasso_alpha=1.0,
    elasticnet_alpha=1.0,
    l1_ratio=0.5,
    pls_components=None,
    pca_components=None,
):
    """
    验证预测模型类型及其相关超参数，并返回规范化后的参数字典。

    只检查当前 model_type 实际会使用的参数。例如 ``model_type='ridge'`` 时只
    检查 ``ridge_alpha``；``model_type='elasticnet'`` 时检查
    ``elasticnet_alpha`` 和 ``l1_ratio``。

    返回的 ``model_params`` 也只包含当前模型真正需要的键。这样可以让后续代码
    更清楚地区分“当前模型的有效参数”和“其他模型的默认参数”，避免在记录
    nested CV 调参结果时混入无关字段。

    Parameters
    ----------
    model_type : str
        预测模型类型，可选 ``'linear'``、``'ridge'``、``'lasso'``、
        ``'elasticnet'``、``'pls'`` 或 ``'pcr'``。
    ridge_alpha, lasso_alpha, elasticnet_alpha : float
        Ridge、Lasso 和 ElasticNet 的正则强度；仅对应模型会使用相应参数。
    l1_ratio : float
        ElasticNet 中 L1 惩罚的比例，仅在 ``model_type='elasticnet'`` 时使用。
    pls_components, pca_components : int or None
        PLS 或 PCA 成分数；``None`` 表示在每个 fold 内根据训练集规模自动确定。

    Returns
    -------
    model_type : str
        规范化后的小写模型名称。
    model_params : dict
        只包含当前模型实际使用的参数。可能的键和值类型为：

        - ``ridge_alpha`` : float，仅 ``model_type='ridge'`` 返回；
        - ``lasso_alpha`` : float，仅 ``model_type='lasso'`` 返回；
        - ``elasticnet_alpha`` : float，仅 ``model_type='elasticnet'`` 返回；
        - ``l1_ratio`` : float，仅 ``model_type='elasticnet'`` 返回；
        - ``pls_components`` : int or None，仅 ``model_type='pls'`` 返回；
        - ``pca_components`` : int or None，仅 ``model_type='pcr'`` 返回。

        ``model_type='linear'`` 返回空字典。

    Raises
    ------
    ValueError
        模型类型不受支持，或当前模型需要的参数不能转换为合法数值时抛出。
    """
    model_type = str(model_type).lower()
    if model_type not in _SUPPORTED_MODELS:
        raise ValueError(
            "Unsupported model_type. Choose from: "
            "'linear', 'ridge', 'lasso', 'elasticnet', 'pls', 'pcr'."
        )

    model_params = {}

    if model_type == "ridge":
        ridge_alpha = _validate_finite_float(
            ridge_alpha,
            "ridge_alpha",
        )
        if ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative.")
        model_params["ridge_alpha"] = ridge_alpha

    elif model_type == "lasso":
        lasso_alpha = _validate_finite_float(
            lasso_alpha,
            "lasso_alpha",
        )
        if lasso_alpha <= 0:
            raise ValueError("lasso_alpha must be positive.")
        model_params["lasso_alpha"] = lasso_alpha

    elif model_type == "elasticnet":
        elasticnet_alpha = _validate_finite_float(
            elasticnet_alpha,
            "elasticnet_alpha",
        )
        if elasticnet_alpha <= 0:
            raise ValueError("elasticnet_alpha must be positive.")

        l1_ratio = _validate_finite_float(l1_ratio, "l1_ratio")
        if not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio must be between 0 and 1.")
        model_params["elasticnet_alpha"] = elasticnet_alpha
        model_params["l1_ratio"] = l1_ratio

    elif model_type == "pls":
        model_params["pls_components"] = _validate_integer(
            pls_components,
            "pls_components",
            minimum=1,
            allow_none=True,
        )

    elif model_type == "pcr":
        model_params["pca_components"] = _validate_integer(
            pca_components,
            "pca_components",
            minimum=1,
            allow_none=True,
        )

    return model_type, model_params


def _validate_cv_config(cv_type, n_splits, n_repeats, n_sub):
    """
    验证并规范化外层交叉验证设置。

    本函数只处理“外层 CV”的结构参数，模型参数和特征筛选参数在其他函数中
    单独验证。这样可以保证固定参数 CV 和 nested CV 复用同一套外层切分规则。

    支持两种模式：

    - ``cv_type='loocv'``：每次只留出 1 个被试作为测试集，因此实际 fold 数等于
      ``n_sub``，且重复次数固定为 1。LOOCV 本身已经穷尽了所有单被试测试折，
      重复运行不会产生新的随机切分。
    - ``cv_type='kfold'``：检查 ``n_repeats`` 和 ``n_splits`` 是否为合法整数，
      并要求 ``n_splits <= n_sub``。实际切分由后续 ``_make_cv_iterator()``
      根据 repeat-specific random_state 生成。

    返回值统一为 ``(cv_type, n_repeats_actual, n_splits_actual)``，便于主函数
    后续不再区分用户原始输入和实际采用的 CV 设置。

    Parameters
    ----------
    cv_type : {'kfold', 'loocv'}
        外层交叉验证类型。
    n_splits : int
        K 折交叉验证的折数，仅在 ``cv_type='kfold'`` 时使用。
    n_repeats : int
        K 折随机划分的重复次数，仅在 ``cv_type='kfold'`` 时使用。
    n_sub : int
        样本量，用于检查折数是否超过可用被试数。

    Returns
    -------
    cv_type : str
        规范化后的小写交叉验证类型。
    n_repeats_actual : int
        实际执行的重复次数；LOOCV 固定为 1。
    n_splits_actual : int
        实际执行的折数；LOOCV 等于 ``n_sub``。

    Raises
    ------
    ValueError
        CV 类型不受支持，折数/重复次数不是合法整数，或 ``n_splits > n_sub`` 时
        抛出。
    """
    cv_type = str(cv_type).lower()

    if cv_type == "loocv":
        return cv_type, 1, n_sub

    if cv_type == "kfold":
        n_repeats_actual = _validate_integer(
            n_repeats,
            "n_repeats",
            minimum=1,
        )
        n_splits_actual = _validate_integer(
            n_splits,
            "n_splits",
            minimum=2,
        )
        if n_splits_actual > n_sub:
            raise ValueError("n_splits cannot be larger than n_sub.")
        return cv_type, n_repeats_actual, n_splits_actual

    raise ValueError("cv_type must be 'kfold' or 'loocv'.")


def _make_cv_iterator(cv_type, n_splits_actual, X, random_state=None):
    """
    根据已验证的 CV 设置生成 sklearn split iterator。

    Parameters
    ----------
    cv_type : {'kfold', 'loocv'}
        已通过 ``_validate_cv_config()`` 检查的交叉验证类型。
    n_splits_actual : int
        实际折数。KFold 使用该值；LOOCV 会忽略该值并按样本数逐一留出。
    X : array-like, shape (n_sub, n_features)
        用于提供样本数的特征矩阵。
    random_state : int or None, default=None
        KFold 随机打乱样本时使用的随机种子；LOOCV 不使用随机种子。

    Returns
    -------
    iterator
        sklearn split iterator，每次产生 ``(train_idx, test_idx)``。
    """
    if cv_type == "kfold":
        return KFold(
            n_splits=n_splits_actual,
            shuffle=True,
            random_state=random_state,
        ).split(X)

    return LeaveOneOut().split(X)


def _select_and_prepare_fold_features(
    X_train,
    y_train,
    Covariates_train,
    X_test,
    y_test,
    Covariates_test,
    selection_method,
    p_thresh,
    top_k,
    top_k_mode,
    max_edges_total,
    min_edges_total,
    fallback_if_too_few,
    max_edges_per_sign,
    model_type,
    standardize_features,
):
    """
    在单个 fold 内选择 CPM 边，并准备训练折和测试折特征。

    本函数的输出是 ``prepared_fold`` 字典，随后会交给
    ``_fit_and_predict_prepared_fold()`` 拟合模型并预测测试折。它把一个 fold 内
    所有容易产生信息泄漏的步骤都限制在训练折中完成，包括协变量残差化、
    偏相关选边、测试折残差化、标准 CPM 的 network strength 汇总或扩展 CPM
    的入选边特征构建、常量特征筛除，以及可选的训练集 Z-score 标准化。

    固定参数 CV 和 nested CV 都调用本函数，以保证训练折和测试折处理规则完全
    一致。nested CV 在同一个 inner fold 内比较多个模型参数时，还可以复用这里
    得到的 F_train/F_test，避免重复执行选边和特征构建。

    Parameters
    ----------
    X_train, X_test : np.ndarray
        当前 fold 的训练集和测试集连接边矩阵，形状分别为
        ``(n_train, n_edges)`` 和 ``(n_test, n_edges)``。
    y_train, y_test : np.ndarray
        当前 fold 的训练集和测试集连续目标变量。
    Covariates_train, Covariates_test : np.ndarray
        当前 fold 的训练集和测试集协变量矩阵；无协变量时为零列矩阵。
    selection_method, p_thresh, top_k, top_k_mode, max_edges_total,
    min_edges_total, fallback_if_too_few, max_edges_per_sign
        传递给 ``select_edges()`` 的选边参数。
    model_type : {'linear', 'ridge', 'lasso', 'elasticnet', 'pls', 'pcr'}
        决定特征组织方式。``linear`` 使用标准 CPM 的正/负 network strength；
        其他模型保留每条入选边作为独立特征。
    standardize_features : bool
        是否用训练折均值和标准差对 ``F_train`` 和 ``F_test`` 进行标准化。

    Returns
    -------
    prepared_fold : dict
        当前 fold 的特征和记录信息。主要键包括：

        - ``F_train`` : np.ndarray or None, shape (n_train, n_features_kept)
          当前 fold 的训练特征。若无有效特征，则为 ``None``。
        - ``F_test`` : np.ndarray or None, shape (n_test, n_features_kept)
          当前 fold 的测试特征。若无有效特征，则为 ``None``。
        - ``y_resid_train`` : np.ndarray, shape (n_train,)
          训练折中去除协变量后的目标残差。
        - ``y_resid_true_test`` : np.ndarray, shape (n_test,)
          测试折真实目标残差，由训练折估计的 ``beta_y`` 计算。
        - ``pos_idx`` / ``neg_idx`` : np.ndarray, shape (n_pos,) / (n_neg,)
          ``select_edges()`` 选中的正边和负边索引。
        - ``feature_edge_idx`` : np.ndarray, shape (n_features_kept,)
          实际进入高维模型的 edge 索引。它不是 ``select_edges()`` 最初选出的
          全部边，而是进一步排除了训练折常量特征后仍保留的 edge；
          ``model_type='linear'`` 使用 network strength，不把单条 edge 作为
          模型特征，因此该字段为空数组。
        - ``feature_names`` : np.ndarray, shape (n_features_kept,)
          实际进入模型的特征名称。标准 CPM 通常为 ``pos_strength`` 和/或
          ``neg_strength``；高维模型通常为 ``edge_10``、``edge_25`` 这类
          edge 标签，并与模型系数一一对应。
        - ``used_fallback`` : bool
          本 fold 是否触发 fallback 选边。若为 True，表示初始阈值或 Top-K
          规则得到的边数不足，函数回退到按 p 值补足候选边。
        - ``has_features`` : bool
          是否存在可用于建模的非恒定特征。若没有任何有效边，或入选特征在
          训练折内均为常量，则为 False；后续预测函数会跳过模型拟合并返回
          全 0 预测。

    Raises
    ------
    ValueError
        ``fast_partCorr_train()`` 返回值形状异常，或下游选边参数无效时抛出。
    """
    n_edges = X_train.shape[1]

    (
        r_vals,
        p_vals,
        X_resid_train,
        y_resid_train,
        beta_X,
        beta_y,
    ) = fast_partCorr_train(X_train, y_train, Covariates_train)

    r_vals = np.asarray(r_vals, dtype=float).ravel()
    p_vals = np.asarray(p_vals, dtype=float).ravel()
    X_resid_train = np.asarray(X_resid_train, dtype=float)
    y_resid_train = np.asarray(y_resid_train, dtype=float).ravel()
    beta_X = np.asarray(beta_X, dtype=float)
    beta_y = np.asarray(beta_y, dtype=float).ravel()

    if r_vals.shape != (n_edges,) or p_vals.shape != (n_edges,):
        raise ValueError(
            "fast_partCorr_train() must return r_vals and p_vals "
            "with shape (n_edges,)."
        )
    if X_resid_train.shape != X_train.shape:
        raise ValueError(
            "fast_partCorr_train() must return X_resid_train with the "
            "same shape as X_train."
        )
    if y_resid_train.shape != y_train.shape:
        raise ValueError(
            "fast_partCorr_train() must return y_resid_train with the "
            "same shape as y_train."
        )

    expected_beta_x_shape = (Covariates_train.shape[1] + 1, n_edges)
    if beta_X.shape != expected_beta_x_shape:
        raise ValueError(
            "fast_partCorr_train() returned beta_X with shape "
            f"{beta_X.shape}; expected {expected_beta_x_shape}."
        )

    expected_beta_y_shape = (Covariates_train.shape[1] + 1,)
    if beta_y.shape != expected_beta_y_shape:
        raise ValueError(
            "fast_partCorr_train() returned beta_y with shape "
            f"{beta_y.shape}; expected {expected_beta_y_shape}."
        )

    C_test = np.column_stack([np.ones(X_test.shape[0]), Covariates_test])
    y_resid_true_test = y_test - C_test @ beta_y

    idx, used_fallback = select_edges(
        r_vals=r_vals,
        p_vals=p_vals,
        selection_method=selection_method,
        p_thresh=p_thresh,
        top_k=top_k,
        top_k_mode=top_k_mode,
        max_edges_total=max_edges_total,
        min_edges_total=min_edges_total,
        fallback_if_too_few=fallback_if_too_few,
        max_edges_per_sign=max_edges_per_sign,
    )
    idx = np.asarray(idx, dtype=int)

    if idx.size > 0:
        pos_idx = idx[r_vals[idx] > 0]
        neg_idx = idx[r_vals[idx] < 0]
    else:
        pos_idx = np.array([], dtype=int)
        neg_idx = np.array([], dtype=int)

    if pos_idx.size + neg_idx.size == 0:
        # 没有任何正/负边可进入 CPM 特征构建；后续预测函数会识别
        # has_features=False 并返回全 0 预测，避免空特征矩阵导致模型报错。
        return {
            "F_train": None,
            "F_test": None,
            "y_resid_train": y_resid_train,
            "y_resid_true_test": y_resid_true_test,
            "pos_idx": pos_idx,
            "neg_idx": neg_idx,
            "feature_edge_idx": np.array([], dtype=int),
            "feature_names": np.array([], dtype=object),
            "used_fallback": used_fallback,
            "has_features": False,
        }

    X_resid_test = X_test - C_test @ beta_X

    if model_type == "linear":
        F_train_list = []
        F_test_list = []
        # 标准 CPM 的模型特征是正/负 network strength，而不是单条 edge。
        # 因此 feature_edge_idx 为空；feature_names 用于和 strength 系数对齐。
        feature_edge_idx = np.array([], dtype=int)
        feature_names = []

        if pos_idx.size > 0:
            F_train_list.append(np.sum(X_resid_train[:, pos_idx], axis=1))
            F_test_list.append(np.sum(X_resid_test[:, pos_idx], axis=1))
            feature_names.append("pos_strength")

        if neg_idx.size > 0:
            F_train_list.append(np.sum(X_resid_train[:, neg_idx], axis=1))
            F_test_list.append(np.sum(X_resid_test[:, neg_idx], axis=1))
            feature_names.append("neg_strength")

        F_train = np.column_stack(F_train_list)
        F_test = np.column_stack(F_test_list)
        feature_names = np.asarray(feature_names, dtype=object)

    else:
        # 高维扩展 CPM 将每条入选 edge 作为独立特征。这里先记录 edge 索引和
        # 可读名称，后面会随常量特征筛除同步更新。
        F_train = X_resid_train[:, idx].copy()
        F_test = X_resid_test[:, idx].copy()
        feature_edge_idx = idx.copy()
        feature_names = np.asarray(
            [f"edge_{edge_idx}" for edge_idx in feature_edge_idx],
            dtype=object,
        )

    train_var = np.var(F_train, axis=0)
    keep = np.isfinite(train_var) & (train_var > _EPS)

    if not np.any(keep):
        # 入选特征若在训练折中全是常量，就无法为模型提供可学习信息；此时该
        # fold 视为没有有效特征。
        return {
            "F_train": None,
            "F_test": None,
            "y_resid_train": y_resid_train,
            "y_resid_true_test": y_resid_true_test,
            "pos_idx": pos_idx,
            "neg_idx": neg_idx,
            "feature_edge_idx": np.array([], dtype=int),
            "feature_names": np.array([], dtype=object),
            "used_fallback": used_fallback,
            "has_features": False,
        }

    F_train = F_train[:, keep]
    F_test = F_test[:, keep]
    # feature_names 必须和 F_train/F_test 的列保持一一对应；高维模型的
    # feature_edge_idx 也同步筛除常量边，便于之后将 coef 对齐到原始 edge。
    feature_names = feature_names[keep]
    if model_type != "linear":
        feature_edge_idx = feature_edge_idx[keep]

    if standardize_features:
        mu = np.mean(F_train, axis=0)
        sd = np.std(F_train, axis=0, ddof=0)

        F_train = (F_train - mu) / sd
        F_test = (F_test - mu) / sd

    return {
        "F_train": F_train,
        "F_test": F_test,
        "y_resid_train": y_resid_train,
        "y_resid_true_test": y_resid_true_test,
        "pos_idx": pos_idx,
        "neg_idx": neg_idx,
        "feature_edge_idx": feature_edge_idx,
        "feature_names": feature_names,
        "used_fallback": used_fallback,
        "has_features": True,
    }


def _build_prediction_model(model_type, model_params, F_train, random_state):
    """
    为当前 fold 创建一个全新的预测模型对象。

    模型参数已经在调用前验证；但模型对象本身必须在每个 fold 内重新创建。
    sklearn 模型在 ``fit()`` 后会保存当前训练集学到的状态，例如回归系数、
    PLS 权重或 PCA 主成分。交叉验证要求每个测试折只能使用对应训练折拟合出的
    模型，因此不能在多个 fold 之间复用同一个已拟合模型对象。

    对 PLS/PCR 来说，模型结构还依赖当前 fold 的有效特征数。选边结果和常量
    特征筛除都可能让 ``F_train.shape[1]`` 在不同 fold 中变化，因此合法成分数
    也需要在当前 fold 内确定。

    Parameters
    ----------
    model_type : {'linear', 'ridge', 'lasso', 'elasticnet', 'pls', 'pcr'}
        要创建的预测模型类型。
    model_params : dict
        已通过 ``_validate_model_params()`` 验证的模型参数，只包含当前模型实际
        使用的键。
    F_train : np.ndarray, shape (n_train, n_features)
        当前 fold 的训练特征矩阵，用于确定 PLS/PCR 的最大合法成分数。
    random_state : int
        Lasso 和 ElasticNet 使用的随机种子。

    Returns
    -------
    model : sklearn estimator
        一个尚未拟合的 sklearn 模型或 pipeline。

    Raises
    ------
    ValueError
        ``model_type`` 不受支持时抛出。
    """
    if model_type == "linear":
        return LinearRegression()

    if model_type == "ridge":
        return Ridge(alpha=model_params["ridge_alpha"])

    if model_type == "lasso":
        return Lasso(
            alpha=model_params["lasso_alpha"],
            max_iter=10000,
            random_state=random_state,
        )

    if model_type == "elasticnet":
        return ElasticNet(
            alpha=model_params["elasticnet_alpha"],
            l1_ratio=model_params["l1_ratio"],
            max_iter=10000,
            random_state=random_state,
        )

    if model_type in ["pls", "pcr"]:
        if model_type == "pcr":
            n_comp = model_params["pca_components"]
        else:
            n_comp = model_params["pls_components"]

        n_train, n_features = F_train.shape
        max_components = min(
            n_features,
            max(1, n_train - 1),
        )

        if n_comp is None:
            n_comp = max_components
        else:
            n_comp = min(n_comp, max_components)

        if model_type == "pls":
            return PLSRegression(n_components=n_comp, scale=False)

        return Pipeline(
            [
                ("pca", PCA(n_components=n_comp)),
                ("linear_regression", LinearRegression()),
            ]
        )

    raise ValueError(f"Unsupported model_type: {model_type}.")


def _fit_and_predict_prepared_fold(
    prepared_fold,
    model_type,
    model_params,
    random_state,
    return_model=False,
):
    """
    使用已经准备好的 fold 拟合模型，并预测测试折。

    Parameters
    ----------
    prepared_fold : dict
        ``_select_and_prepare_fold_features()``
        返回的 fold 特征字典。
    model_type : {'linear', 'ridge', 'lasso', 'elasticnet', 'pls', 'pcr'}
        预测模型类型。
    model_params : dict
        当前模型的已验证参数。
    random_state : int
        随机种子，传递给需要随机状态的模型。
    return_model : bool, default=False
        是否同时返回当前 fold 已拟合的模型对象。outer CV 会启用该选项，并由
        ``_extract_fold_edge_record()`` 生成当前 fold 的 edge record；inner CV
        调参只需要预测值，通常关闭。

    Returns
    -------
    y_pred_test : np.ndarray, shape (n_test,)
        测试折的预测值。若当前 fold 没有有效特征，则返回全 0 向量。
    model : sklearn estimator or None, optional
        仅当 ``return_model=True`` 时返回。若当前 fold 没有有效特征，则返回
        ``None``；否则返回已在 ``F_train`` 和 ``y_resid_train`` 上拟合好的模型。
    """
    y_resid_true_test = prepared_fold["y_resid_true_test"]
    if not prepared_fold["has_features"]:
        y_pred_test = np.zeros(y_resid_true_test.shape[0], dtype=float)
        if not return_model:
            return y_pred_test
        return y_pred_test, None

    model = _build_prediction_model(
        model_type=model_type,
        model_params=model_params,
        F_train=prepared_fold["F_train"],
        random_state=random_state,
    )
    model.fit(prepared_fold["F_train"], prepared_fold["y_resid_train"])
    y_pred_test = np.asarray(model.predict(prepared_fold["F_test"])).ravel()

    if not return_model:
        return y_pred_test

    return y_pred_test, model


def _extract_fold_edge_record(repeat_idx, fold_idx, prepared_fold, model, model_type):
    """
    从单个 fold 的拟合模型中整理一条统一的 edge record。

    本函数从已经拟合的模型中取出当前 fold 特征空间里的系数，并把这些系数与
    repeat/fold 编号、特征名称、edge 索引和选边记录合并成一个可追踪的字典。
    PCR 的系数会从 PCA 成分空间回投到原始 fold 特征空间；PLS/PCR 的 ``coef``
    解释仍比 Ridge/Lasso/ElasticNet 更间接。

    ``coef`` 是唯一的系数字段，含义由 ``edge_idx`` 决定：

    - 若 ``edge_idx`` 非空，说明模型特征可以逐一对应到单条 edge，此时
      ``coef`` 与 ``edge_idx`` 一一对应，``coef`` 就是这些 edge 的模型权重。
    - 若 ``edge_idx`` 为空，说明模型特征不是单条 edge。标准 CPM 的
      ``linear`` 模型属于这种情况：``coef`` 对应 ``pos_strength`` 和/或
      ``neg_strength``，不能直接解释为单条 edge 的权重。

    Parameters
    ----------
    repeat_idx : int
        当前 outer repeat 的零基编号。
    fold_idx : int
        当前 outer fold 的零基编号。
    prepared_fold : dict
        ``_select_and_prepare_fold_features()``
        返回的 fold 特征字典。
    model : sklearn estimator or None
        当前 fold 已拟合的预测模型。若 ``prepared_fold['has_features']`` 为
        False，则为 ``None``。
    model_type : {'linear', 'ridge', 'lasso', 'elasticnet', 'pls', 'pcr'}
        当前 fold 使用的预测模型类型。

    Returns
    -------
    edge_record : dict
        单个 fold 的统一 edge 记录。主要键包括：

        - ``repeat`` : int
          当前 outer repeat 的零基编号。
        - ``fold`` : int
          当前 outer fold 的零基编号。
        - ``model_type`` : str
          当前 fold 使用的预测模型类型。
        - ``feature_names`` : np.ndarray, shape (n_features_kept,)
          与 ``coef`` 一一对应的模型特征名称。
        - ``coef`` : np.ndarray, shape (n_features_kept,)
          当前 fold 模型在实际特征空间中的系数。
        - ``edge_idx`` : np.ndarray, shape (n_features_kept,) or (0,)
          高维模型中与 ``coef`` 对应的 edge 索引；标准 CPM 中为空数组，因为其
          系数对应 network strength，而不是单条 edge。
        - ``pos_idx`` / ``neg_idx`` : np.ndarray
          当前 fold 被 CPM 选中的正边和负边索引。
        - ``n_pos`` / ``n_neg`` : int
          当前 fold 被 CPM 选中的正边数和负边数；等价于
          ``pos_idx.size`` 和 ``neg_idx.size``，用于快速查看边数。
        - ``used_fallback`` : bool
          当前 fold 是否触发 fallback 选边。
        - ``has_features`` : bool
          当前 fold 是否存在可用于建模的非恒定特征。

    Raises
    ------
    ValueError
        提取到的系数数量与当前 fold 的实际特征数不一致时抛出。
    """
    # feature_names 是模型真正使用的特征名称；feature_edge_idx 是这些特征在
    # 原始 edge 空间中的索引。标准 CPM 的特征是 network strength，因此
    # feature_edge_idx 为空；高维模型才会保留单条 edge 的索引。
    feature_edge_idx = np.asarray(prepared_fold["feature_edge_idx"], dtype=int)
    feature_names = np.asarray(prepared_fold["feature_names"], dtype=object)

    if not prepared_fold["has_features"]:
        # 当前 fold 没有可用于建模的非恒定特征，因此没有系数可记录。
        coef = np.array([], dtype=float)
    elif model_type == "pcr":
        # PCR 先把 edge 特征投影到 PCA 成分，再在成分空间做线性回归。
        # 为了让权重回到可解释的 fold 特征空间，需要把成分系数回投到
        # PCA 之前的特征空间。
        pca = model.named_steps["pca"]
        reg = model.named_steps["linear_regression"]
        component_coef = np.asarray(reg.coef_, dtype=float).ravel()
        coef = pca.components_.T @ component_coef
    else:
        # 其他模型的 coef_ 已经定义在当前 fold 的模型输入特征空间中。
        coef = np.asarray(model.coef_, dtype=float).ravel()

    n_features = feature_names.size
    # 防御性检查：每个模型特征必须有且只有一个系数，否则后续 coef
    # 与 feature name / edge index 的对应关系不可信。
    if coef.size != n_features:
        raise ValueError(
            "Model coefficients must have the same length as fold features; "
            f"got {coef.size}, expected {n_features}."
        )
    coef = coef.astype(float, copy=False)

    return {
        "repeat": repeat_idx,
        "fold": fold_idx,
        "model_type": model_type,
        "feature_names": feature_names.copy(),
        "coef": coef.copy(),
        "edge_idx": feature_edge_idx.copy(),
        "pos_idx": prepared_fold["pos_idx"].copy(),
        "neg_idx": prepared_fold["neg_idx"].copy(),
        "n_pos": int(prepared_fold["pos_idx"].size),
        "n_neg": int(prepared_fold["neg_idx"].size),
        "used_fallback": bool(prepared_fold["used_fallback"]),
        "has_features": bool(prepared_fold["has_features"]),
    }


# -------------------------------------------------------------------------
# CV-integrated selected-space Haufe helper functions
# -------------------------------------------------------------------------
#
# cpm_fixed_params_cv_with_haufe() / cpm_nested_cv_with_haufe()
#     |
#     |-- regular CPM outer fold preparation and model fitting
#     |-- _compute_fold_selected_haufe_record()
#     |-- _summarize_haufe_records()
#
# 这组 helper 的目标是在 CPM 外层 fold 训练完成后立即计算 selected-space Haufe，
# 避免事后重新残差化、重新标准化 selected FC、再重建训练集预测值。Haufe 的解释
# 空间与当前 fold 实际进入模型的 selected edge feature space 完全一致。


def _validate_haufe_supported_model_type(model_type):
    """
    检查当前模型是否适合计算 edge-level selected-space Haufe。

    第一版只支持高维扩展 CPM，即每个模型特征都对应一条 FC edge 的模型：
    ridge、lasso、elasticnet、pls 和 pcr。标准 CPM 的 ``linear`` 模型使用
    ``pos_strength`` / ``neg_strength`` 两个汇总特征，不能直接把模型输入列一一
    映射回单条 FC，因此本函数暂不支持。
    """
    if model_type == "linear":
        raise ValueError(
            "Haufe output is only supported for high-dimensional CPM models "
            "('ridge', 'lasso', 'elasticnet', 'pls', or 'pcr'). "
            "model_type='linear' uses pos/neg network-strength features and is "
            "not supported in the first implementation."
        )


def _compute_fold_selected_haufe_record(
    repeat_idx,
    fold_idx,
    prepared_fold,
    model,
    model_type,
    eps=1e-12,
):
    """
    在一个已经完成 CPM 训练的 outer fold 内计算 selected-space Haufe。

    本函数不重新选边、不重新训练模型，也不重新残差化 FC/y。它直接复用
    ``_select_and_prepare_fold_features()`` 已经得到的 ``F_train`` 和当前 fold
    已拟合的 ``model``：

    1. ``F_train`` 是当前 fold 真正进入模型的 selected edge feature matrix；
       若 ``standardize_features=True``，它已经是训练折 z-score 后的特征；
    2. ``model.predict(F_train)`` 给出当前训练折上的模型输出 ``yhat_train``；
    3. 对每一列 selected edge feature 计算与 ``yhat_train`` 的 Haufe slope 和
       Haufe correlation；
    4. 通过 ``prepared_fold['feature_edge_idx']`` 将结果映射回原始 FC edge 编号。

    这里计算的是 selected-space Haufe：当前 fold 没有进入模型的 FC 不参与该
    fold 的 Haufe 计算，后续汇总时等价于对这些 FC 贡献 0。

    Returns
    -------
    haufe_record : dict
        单个 outer fold 的 Haufe 记录，主要包含：

        - ``repeat`` / ``fold`` : int
          当前 outer repeat 和 fold 的零基编号。
        - ``edge_idx`` : np.ndarray, shape (n_selected_features,)
          当前 fold 实际进入模型、且可计算 Haufe 的 edge 编号。
        - ``haufe_slope`` : np.ndarray, shape (n_selected_features,)
          ``Cov(F_e, yhat) / Var(yhat)``。
        - ``haufe_corr`` : np.ndarray, shape (n_selected_features,)
          ``Corr(F_e, yhat)``，范围限制在 [-1, 1]。
        - ``n_selected_features`` : int
          当前 fold 可计算 Haufe 的 selected edge 数。
        - ``valid_haufe`` : bool
          当前 fold 是否成功得到 Haufe 值。
    """
    _validate_haufe_supported_model_type(model_type)

    empty = {
        "repeat": int(repeat_idx),
        "fold": int(fold_idx),
        "model_type": str(model_type),
        "edge_idx": np.array([], dtype=int),
        "haufe_slope": np.array([], dtype=float),
        "haufe_corr": np.array([], dtype=float),
        "n_selected_features": 0,
        "valid_haufe": False,
    }

    if (not prepared_fold["has_features"]) or model is None:
        return empty

    edge_idx = np.asarray(prepared_fold["feature_edge_idx"], dtype=int).ravel()
    F_train = np.asarray(prepared_fold["F_train"], dtype=np.float64)
    if edge_idx.size == 0 or F_train.ndim != 2 or F_train.shape[1] != edge_idx.size:
        return empty

    yhat_train = np.asarray(model.predict(F_train), dtype=np.float64).ravel()
    yhat_dm = yhat_train - yhat_train.mean()
    yhat_ss = float(yhat_dm @ yhat_dm)
    if (not np.isfinite(yhat_ss)) or yhat_ss <= eps:
        return empty

    F_dm = F_train - F_train.mean(axis=0, keepdims=True)
    numerator = F_dm.T @ yhat_dm
    f_ss = np.sum(F_dm**2, axis=0)
    valid = np.isfinite(f_ss) & (f_ss > eps)
    if not np.any(valid):
        return empty

    edge_idx = edge_idx[valid]
    numerator = numerator[valid]
    f_ss = f_ss[valid]

    haufe_slope = numerator / yhat_ss
    haufe_corr = numerator / np.sqrt(np.maximum(f_ss * yhat_ss, eps))
    haufe_corr = np.clip(haufe_corr, -1.0, 1.0)

    return {
        "repeat": int(repeat_idx),
        "fold": int(fold_idx),
        "model_type": str(model_type),
        "edge_idx": edge_idx.astype(int, copy=False),
        "haufe_slope": haufe_slope.astype(float, copy=False),
        "haufe_corr": haufe_corr.astype(float, copy=False),
        "n_selected_features": int(edge_idx.size),
        "valid_haufe": True,
    }


def _summarize_haufe_records(haufe_records, n_edges):
    """
    将 fold-level Haufe records 汇总为全脑 edge-level selected-space Haufe map。

    ``haufe_records`` 中每条记录只包含当前 fold 实际进入模型的 selected edges。
    本函数把这些 fold-level 结果累加回长度为 ``n_edges`` 的全脑 edge vector：

    - 被某个 fold 选中并成功计算 Haufe 的 edge，累加该 fold 的 slope/correlation；
    - 未被该 fold 选中的 edge 对该 fold 贡献 0；
    - 最后除以 ``n_valid_folds``，得到 selected-space mean across folds。

    对 repeated KFold，``n_valid_folds`` 是所有 repeats × folds 中成功计算 Haufe 的
    fold 总数；对 LOOCV，它通常等于被试数，但若某些 fold 没有有效特征或
    ``yhat`` 方差为 0，则会少于被试数。
    """
    selected_slope_sum = np.zeros(int(n_edges), dtype=np.float64)
    selected_corr_sum = np.zeros(int(n_edges), dtype=np.float64)
    selected_count = np.zeros(int(n_edges), dtype=np.int64)
    n_valid_folds = 0

    for record in haufe_records:
        if not record.get("valid_haufe", False):
            continue
        edge_idx = np.asarray(record["edge_idx"], dtype=int).ravel()
        slope = np.asarray(record["haufe_slope"], dtype=np.float64).ravel()
        corr = np.asarray(record["haufe_corr"], dtype=np.float64).ravel()
        if edge_idx.size == 0:
            continue
        if edge_idx.size != slope.size or edge_idx.size != corr.size:
            raise ValueError("Each Haufe record must align edge_idx, haufe_slope, and haufe_corr.")

        selected_slope_sum[edge_idx] += slope
        selected_corr_sum[edge_idx] += corr
        selected_count[edge_idx] += 1
        n_valid_folds += 1

    if n_valid_folds > 0:
        haufe_slope_mean_all = selected_slope_sum / float(n_valid_folds)
        haufe_corr_mean_all = selected_corr_sum / float(n_valid_folds)
        haufe_selection_freq = selected_count / float(n_valid_folds)
    else:
        haufe_slope_mean_all = np.zeros(int(n_edges), dtype=np.float64)
        haufe_corr_mean_all = np.zeros(int(n_edges), dtype=np.float64)
        haufe_selection_freq = np.zeros(int(n_edges), dtype=np.float64)

    return {
        "fold_records": haufe_records,
        "n_valid_folds": int(n_valid_folds),
        "selected_slope_sum": selected_slope_sum,
        "selected_corr_sum": selected_corr_sum,
        "selected_count": selected_count,
        "haufe_slope_selected_mean_all_folds": haufe_slope_mean_all,
        "haufe_corr_selected_mean_all_folds": haufe_corr_mean_all,
        "haufe_slope_abs_selected_mean_all_folds": np.abs(haufe_slope_mean_all),
        "haufe_corr_abs_selected_mean_all_folds": np.abs(haufe_corr_mean_all),
        "haufe_selection_count": selected_count.copy(),
        "haufe_selection_freq": haufe_selection_freq,
    }


# -------------------------------------------------------------------------
# Fixed-parameter CV helper functions
# -------------------------------------------------------------------------
#
# cpm_fixed_params_cv()
#     |
#     |-- _validate_X_y_Covariates()
#     |-- _validate_model_params()
#     |-- _validate_cv_config()
#     |
#     |-- repeat loop
#            |
#            |-- _make_cv_iterator()
#            |
#            |-- fold loop
#                   |
#                   |-- _select_and_prepare_fold_features()
#                   |      |-- fast_partCorr_train()
#                   |      |      |-- _validate_X_y_Covariates()
#                   |      |-- select_edges()
#                   |
#                   |-- _fit_and_predict_prepared_fold()
#                          |-- _build_prediction_model()
#                   |-- _extract_fold_edge_record()
#
# 固定参数 CV 不做超参数搜索。每个 fold 只用当前训练折估计协变量效应、选择边、
# 构建特征并拟合一次模型；测试折只接收训练折学到的参数和变换。


def cpm_fixed_params_cv(
    X,
    y,
    Covariates,
    # ---------- 交叉验证参数 ----------
    n_splits=10,
    n_repeats=50,
    random_state=0,
    cv_type="kfold",
    # ---------- 选边参数 ----------
    selection_method="pvalue",
    p_thresh=0.01,
    top_k=None,
    top_k_mode="total",
    max_edges_total=None,
    min_edges_total=0,
    fallback_if_too_few=False,
    max_edges_per_sign=None,
    # ---------- 模型参数 ----------
    ridge_alpha=1.0,
    lasso_alpha=1.0,
    elasticnet_alpha=1.0,
    l1_ratio=0.5,
    model_type="ridge",
    pls_components=None,
    pca_components=None,
    # ---------- 特征处理参数 ----------
    standardize_features=True,
    # ---------- 输出控制 ----------
    verbose=False,
):
    """
    使用固定模型超参数运行 K 折/LOOCV CPM 交叉验证。

    所有会学习数据分布的步骤都在训练折内完成，包括协变量回归、边选择、
    常量特征筛除和标准化。测试折只使用训练折估计的参数，从而避免信息泄漏。
    当前实现要求 X、y 和 Covariates（若提供）均为有限数值，不在函数内插补
    缺失值。

    几个固定参数 CV 辅助函数的分工如下：

    - ``_validate_X_y_Covariates()``：统一检查并格式化 X、y 和 Covariates；
    - ``_validate_model_params()``：只保留当前 ``model_type`` 实际使用的模型
      参数；
    - ``_validate_cv_config()`` 和 ``_make_cv_iterator()``：确定并生成外层 CV
      切分；
    - ``_select_and_prepare_fold_features()``：
      在每个训练折内完成协变量残差化、偏相关选边、测试折残差化和特征构建；
    - ``_fit_and_predict_prepared_fold()``：基于已经准备好的 fold 拟合模型，
      并预测对应测试折。
    - ``_extract_fold_edge_record()``：提取每个 fold 的模型系数，并与选边
      结果合并为统一的 ``edge_records``。

    计算流程为：

    1. 验证输入数据、模型参数和 CV 参数；
    2. 对每次 repeat 生成一组 KFold 或 LOOCV 切分；
    3. 在每个 fold 内只使用训练折估计协变量效应、选择连接边并构建特征；
    4. 用固定模型参数拟合当前训练折，并预测当前测试折；
    5. 汇总所有 fold 的折外预测、折特异真实残差和统一的 edge 记录。

    Parameters
    ----------
    X : array-like, shape (n_sub, n_edges)
        被试乘连接边的特征矩阵。当前实现要求所有值有限，不在函数内插补缺失值。
    y : array-like, shape (n_sub,)
        连续预测目标。模型实际拟合的是训练折内去除协变量后的 y 残差。
    Covariates : array-like or None, shape (n_sub, n_Covariates)
        需要控制的协变量，不包含截距；传入 None 表示只控制截距。
    n_splits : int, default=10
        K 折交叉验证的折数，仅在 ``cv_type='kfold'`` 时使用。
    n_repeats : int, default=50
        K 折随机划分的重复次数。LOOCV 无随机划分，因此该参数会被忽略。
    random_state : int, default=0
        第一次 K 折划分的随机种子；第 r 次重复使用 ``random_state + r``。
    cv_type : {'kfold', 'loocv'}, default='kfold'
        外层交叉验证方案。
    selection_method : {'pvalue', 'topk'}, default='pvalue'
        训练折内的边选择方法，具体规则由 ``select_edges()`` 实现。
    p_thresh : float, default=0.01
        p 值选边阈值。
    top_k : int or None, default=None
        Top-K 模式的选边数；``top_k_mode='per_sign'`` 时表示每个方向的数量。
    top_k_mode : {'total', 'per_sign'}, default='total'
        Top-K 是在全部边中选择，还是分别在正边和负边中选择。
    max_edges_total : int or None, default=None
        ``select_edges()`` 返回的最大总边数。
    min_edges_total : int, default=0
        启用 fallback 时希望保留的最少总边数。
    fallback_if_too_few : bool, default=False
        初始选边不足时，是否回退为按 p 值选择 ``min_edges_total`` 条边。
    max_edges_per_sign : int or None, default=None
        正边和负边各自的上限。该限制在 fallback 之后执行，因此最终总边数
        仍可能低于 ``min_edges_total``。
    ridge_alpha : float, default=1.0
        Ridge 的 L2 正则强度。
    lasso_alpha : float, default=1.0
        Lasso 的 L1 正则强度，仅在 ``model_type='lasso'`` 时使用。
    elasticnet_alpha : float, default=1.0
        ElasticNet 的总体正则强度。
    l1_ratio : float, default=0.5
        ElasticNet 的 L1/L2 混合比例。
    model_type : str, default='ridge'
        预测模型及特征组织方式。可选 ``'linear'``、``'ridge'``、``'lasso'``、
        ``'elasticnet'``、``'pls'`` 或 ``'pcr'``。``linear`` 使用标准 CPM：
        将正、负边分别求和，形成最多两个 network-strength 特征并执行普通
        线性回归；其他模型将每条入选边作为独立高维特征。
    pls_components, pca_components : int or None
        PLS/PCA 成分数；会限制在训练折样本数和有效特征数允许的范围内。
    standardize_features : bool, default=True
        是否使用训练折均值和标准差标准化预测特征。
    verbose : bool, default=False
        是否打印重复交叉验证进度。

    Returns
    -------
    y_pred_mean : np.ndarray, shape (n_sub,)
        所有重复的折外预测平均值，目标尺度为 y 的训练折协变量残差。

    y_resid_true_mean : np.ndarray, shape (n_sub,)
        所有重复中折特异真实 y 残差的平均值，与 ``y_pred_mean`` 一一对应。

    y_pred_mat : np.ndarray, shape (n_repeats_actual, n_sub)
        每次重复的完整折外预测矩阵；LOOCV 时第一维固定为 1。

    y_resid_true_mat : np.ndarray, shape (n_repeats_actual, n_sub)
        每次重复对应的真实 y 残差矩阵。第 r 行必须与 ``y_pred_mat`` 第 r 行
        配对计算预测性能。

    edge_records : list of dict
        每个 fold 的统一 edge 记录。元素按 repeat 优先、fold 次优先排列；
        对 K 折重复 CV，零基记录位置为
        ``record_idx = repeat_idx * n_splits_actual + fold_idx``。每个 dict 主要包含：

        - ``repeat`` / ``fold`` : int
          当前记录对应的 repeat 和 fold 零基编号。
        - ``model_type`` : str
          当前 fold 使用的预测模型类型。
        - ``feature_names`` : np.ndarray, shape (n_features_kept,)
          与 ``coef`` 一一对应的模型特征名称。
        - ``coef`` : np.ndarray, shape (n_features_kept,)
          当前 fold 模型在实际特征空间中的系数。
        - ``edge_idx`` : np.ndarray, shape (n_features_kept,) or (0,)
          高维模型中与 ``coef`` 对应的 edge 索引；标准 CPM 中为空数组。
        - ``pos_idx`` / ``neg_idx`` : np.ndarray
          当前 fold 被 CPM 选中的正边和负边索引。
        - ``n_pos`` / ``n_neg`` : int
          当前 fold 被 CPM 选中的正边数和负边数。
        - ``used_fallback`` / ``has_features`` : bool
          当前 fold 是否触发 fallback，以及是否存在有效建模特征。

        对 Ridge、Lasso 和 ElasticNet，非空 ``edge_idx`` 与 ``coef`` 一一
        对应，可直接用于统计每条 edge 的平均权重、平均绝对权重和符号一致性。
        PLS/PCR 的 ``coef`` 为回投到当前 fold 特征空间后的系数，解释应更谨慎。

    Raises
    ------
    ValueError
        输入形状、模型参数、交叉验证参数或训练折自由度无效时抛出。

    Notes
    -----
    返回的是 y 残差的预测值，不是原始 y 的预测值。每个测试折的真实 y 残差
    由该折训练集拟合的 ``beta_y`` 计算，未使用测试折估计协变量效应。

    重复交叉验证建议优先逐行比较 ``y_pred_mat`` 和 ``y_resid_true_mat``，得到每次
    repeat 的性能分布；也可以比较 ``y_pred_mean`` 和 ``y_resid_true_mean``，
    评估所有重复平均后的预测。

    ``linear`` 模型的预测对特征线性缩放不敏感；高维 Ridge、Lasso、ElasticNet、
    PLS 和 PCR 通常建议保留 ``standardize_features=True``。若希望根据数据选择
    正则强度或成分数，请使用 ``cpm_nested_cv()``。无论使用哪个函数，都不能
    利用外层测试折调参。

    高维模型的解释不应只看 edge selection frequency。建议基于
    ``edge_records`` 同时报告每条 edge 的入选频率、平均权重、平均绝对权重和
    符号一致性。
    """

    # ---------- 1. 输入数据与全局参数验证 ----------
    # 这一层只做和具体 fold 无关的检查：样本数、矩阵形状、模型类型、固定模型
    # 参数和 CV 设置。真正会学习数据分布的操作稍后都放进训练折内部。
    X, y, Covariates, n_sub, _ = _validate_X_y_Covariates(
        X,
        y,
        Covariates,
        minimum_n_sub=3,
    )
    model_type, model_params = _validate_model_params(
        model_type=model_type,
        ridge_alpha=ridge_alpha,
        lasso_alpha=lasso_alpha,
        elasticnet_alpha=elasticnet_alpha,
        l1_ratio=l1_ratio,
        pls_components=pls_components,
        pca_components=pca_components,
    )
    random_state = _validate_integer(random_state, "random_state")
    standardize_features = _validate_boolean(
        standardize_features,
        "standardize_features",
    )
    verbose = _validate_boolean(verbose, "verbose")
    cv_type, n_repeats_actual, n_splits_actual = _validate_cv_config(
        cv_type=cv_type,
        n_splits=n_splits,
        n_repeats=n_repeats,
        n_sub=n_sub,
    )

    if verbose and cv_type == "loocv" and n_repeats != 1:
        print("cv_type='loocv': n_repeats is ignored and set to 1.")

    # ---------- 2. 汇总 fold 内固定使用的选边参数 ----------
    # select_edges() 的参数在所有 fold 中相同；但具体选中的边必须在每个训练折内
    # 重新计算，不能提前在全样本上选择。
    selection_params = {
        "selection_method": selection_method,
        "p_thresh": p_thresh,
        "top_k": top_k,
        "top_k_mode": top_k_mode,
        "max_edges_total": max_edges_total,
        "min_edges_total": min_edges_total,
        "fallback_if_too_few": fallback_if_too_few,
        "max_edges_per_sign": max_edges_per_sign,
    }

    # ---------- 3. 初始化输出容器 ----------
    # y_pred_mat 和 y_resid_true_mat 的每一行对应一次 repeat。每个被试的预测值都来自
    # 其所在 fold 的折外预测，因此后续可直接逐行计算无泄漏预测性能。
    y_pred_mat = np.zeros((n_repeats_actual, n_sub), dtype=float)
    y_resid_true_mat = np.zeros((n_repeats_actual, n_sub), dtype=float)
    edge_records = []

    for rep in range(n_repeats_actual):
        if verbose:
            print(f"Repeat {rep + 1}/{n_repeats_actual}")

        # ---------- 4. 当前 repeat 的外层 CV 切分 ----------
        # KFold 每个 repeat 使用不同随机种子；LOOCV 则由 _make_cv_iterator()
        # 忽略随机种子并逐个被试留出。
        cv_iter = _make_cv_iterator(
            cv_type,
            n_splits_actual,
            X,
            random_state=random_state + rep,
        )
        y_pred = np.zeros(n_sub, dtype=float)
        y_resid_true = np.zeros(n_sub, dtype=float)

        for fold_idx, (train_idx, test_idx) in enumerate(cv_iter):
            # ---------- 5. fold 内 CPM 特征准备 ----------
            # 这里是避免信息泄漏的核心：协变量回归、偏相关、选边、常量特征删除和
            # 标准化全部只用训练折估计，再应用到当前测试折。
            prepared = _select_and_prepare_fold_features(
                X_train=X[train_idx],
                y_train=y[train_idx],
                Covariates_train=Covariates[train_idx],
                X_test=X[test_idx],
                y_test=y[test_idx],
                Covariates_test=Covariates[test_idx],
                model_type=model_type,
                standardize_features=standardize_features,
                **selection_params,
            )

            # y_resid_true_test 是用训练折 beta_y 作用于测试折得到的真实残差，
            # 必须和同一 fold 的预测值配对评估。
            y_resid_true[test_idx] = prepared["y_resid_true_test"]

            # ---------- 6. 用固定模型参数拟合当前训练折并预测测试折 ----------
            # 同时取回 fitted model，用于强制生成统一 edge_records；调参逻辑不在
            # 本函数中发生。
            fold_pred, fitted_model = _fit_and_predict_prepared_fold(
                prepared,
                model_type=model_type,
                model_params=model_params,
                random_state=random_state + rep,
                return_model=True,
            )
            edge_records.append(
                _extract_fold_edge_record(
                    repeat_idx=rep,
                    fold_idx=fold_idx,
                    prepared_fold=prepared,
                    model=fitted_model,
                    model_type=model_type,
                )
            )

            y_pred[test_idx] = fold_pred

        # 当前 repeat 的所有 fold 都完成后，整行预测矩阵已经覆盖全部被试。
        y_pred_mat[rep] = y_pred
        y_resid_true_mat[rep] = y_resid_true

    # ---------- 7. 汇总跨 repeat 的平均预测和 fold 级记录 ----------
    # y_pred_mean/y_resid_true_mean 是平均预测层面的结果；完整 repeat 层面的评估
    # 应优先使用 y_pred_mat 和 y_resid_true_mat。
    y_pred_mean = y_pred_mat.mean(axis=0)
    y_resid_true_mean = y_resid_true_mat.mean(axis=0)

    return (
        y_pred_mean,
        y_resid_true_mean,
        y_pred_mat,
        y_resid_true_mat,
        edge_records,
    )


def cpm_fixed_params_cv_with_haufe(
    X,
    y,
    Covariates,
    # ---------- 交叉验证参数 ----------
    n_splits=10,
    n_repeats=50,
    random_state=0,
    cv_type="kfold",
    # ---------- 选边参数 ----------
    selection_method="pvalue",
    p_thresh=0.01,
    top_k=None,
    top_k_mode="total",
    max_edges_total=None,
    min_edges_total=0,
    fallback_if_too_few=False,
    max_edges_per_sign=None,
    # ---------- 模型参数 ----------
    ridge_alpha=1.0,
    lasso_alpha=1.0,
    elasticnet_alpha=1.0,
    l1_ratio=0.5,
    model_type="ridge",
    pls_components=None,
    pca_components=None,
    # ---------- 特征处理参数 ----------
    standardize_features=True,
    # ---------- 输出控制 ----------
    verbose=False,
):
    """
    使用固定模型超参数运行 CPM CV，并在每个 outer fold 内同步计算 selected-space Haufe。

    本函数与 ``cpm_fixed_params_cv()`` 使用相同的建模流程和相同的前 5 个返回值。
    区别是：每个 outer fold 的模型训练完成后，立即用当前 fold 已经准备好的
    ``F_train`` 和 fitted model 计算 selected-space Haufe feature，避免事后再
    重新残差化、标准化和重建训练集预测值。

    当前第一版只支持高维扩展 CPM 模型：``ridge``、``lasso``、``elasticnet``、
    ``pls`` 和 ``pcr``。这些模型的每个输入特征都对应一条 selected FC edge，
    因此可以得到 edge-level Haufe。``model_type='linear'`` 的标准 CPM 使用
    ``pos_strength`` / ``neg_strength`` 汇总特征，暂不支持。

    Returns
    -------
    与 ``cpm_fixed_params_cv()`` 相同的前 5 个返回值，后面追加：

    haufe_results : dict
        selected-space Haufe 汇总结果。主要字段包括：

        - ``fold_records`` : list of dict
          每个 outer fold 的 Haufe 记录。
        - ``n_valid_folds`` : int
          成功计算 Haufe 的 outer folds 数。
        - ``haufe_slope_selected_mean_all_folds`` : np.ndarray, shape (n_edges,)
          每条 FC 的 selected-space Haufe slope，未入选 fold 贡献 0。
        - ``haufe_corr_selected_mean_all_folds`` : np.ndarray, shape (n_edges,)
          每条 FC 的 selected-space Haufe correlation，未入选 fold 贡献 0。
        - ``haufe_selection_count`` / ``haufe_selection_freq`` : np.ndarray
          每条 FC 在有效 Haufe folds 中被模型实际使用的次数和频率。
    """
    # ---------- 1. 输入数据与全局参数验证 ----------
    # 与 cpm_fixed_params_cv() 完全一致：这一层只做和具体 fold 无关的检查。
    X, y, Covariates, n_sub, n_edges = _validate_X_y_Covariates(
        X,
        y,
        Covariates,
        minimum_n_sub=3,
    )
    model_type, model_params = _validate_model_params(
        model_type=model_type,
        ridge_alpha=ridge_alpha,
        lasso_alpha=lasso_alpha,
        elasticnet_alpha=elasticnet_alpha,
        l1_ratio=l1_ratio,
        pls_components=pls_components,
        pca_components=pca_components,
    )
    _validate_haufe_supported_model_type(model_type)
    random_state = _validate_integer(random_state, "random_state")
    standardize_features = _validate_boolean(
        standardize_features,
        "standardize_features",
    )
    verbose = _validate_boolean(verbose, "verbose")
    cv_type, n_repeats_actual, n_splits_actual = _validate_cv_config(
        cv_type=cv_type,
        n_splits=n_splits,
        n_repeats=n_repeats,
        n_sub=n_sub,
    )

    if verbose and cv_type == "loocv" and n_repeats != 1:
        print("cv_type='loocv': n_repeats is ignored and set to 1.")

    # ---------- 2. 汇总 fold 内固定使用的选边参数 ----------
    selection_params = {
        "selection_method": selection_method,
        "p_thresh": p_thresh,
        "top_k": top_k,
        "top_k_mode": top_k_mode,
        "max_edges_total": max_edges_total,
        "min_edges_total": min_edges_total,
        "fallback_if_too_few": fallback_if_too_few,
        "max_edges_per_sign": max_edges_per_sign,
    }

    # ---------- 3. 初始化输出容器 ----------
    y_pred_mat = np.zeros((n_repeats_actual, n_sub), dtype=float)
    y_resid_true_mat = np.zeros((n_repeats_actual, n_sub), dtype=float)
    edge_records = []
    haufe_records = []

    for rep in range(n_repeats_actual):
        if verbose:
            print(f"Repeat {rep + 1}/{n_repeats_actual}")

        # ---------- 4. 当前 repeat 的外层 CV 切分 ----------
        cv_iter = _make_cv_iterator(
            cv_type,
            n_splits_actual,
            X,
            random_state=random_state + rep,
        )
        y_pred = np.zeros(n_sub, dtype=float)
        y_resid_true = np.zeros(n_sub, dtype=float)

        for fold_idx, (train_idx, test_idx) in enumerate(cv_iter):
            # ---------- 5. fold 内 CPM 特征准备 ----------
            # 协变量回归、偏相关、选边、常量特征删除和标准化全部只用训练折估计。
            prepared = _select_and_prepare_fold_features(
                X_train=X[train_idx],
                y_train=y[train_idx],
                Covariates_train=Covariates[train_idx],
                X_test=X[test_idx],
                y_test=y[test_idx],
                Covariates_test=Covariates[test_idx],
                model_type=model_type,
                standardize_features=standardize_features,
                **selection_params,
            )

            y_resid_true[test_idx] = prepared["y_resid_true_test"]

            # ---------- 6. 用固定模型参数拟合当前训练折并预测测试折 ----------
            fold_pred, fitted_model = _fit_and_predict_prepared_fold(
                prepared,
                model_type=model_type,
                model_params=model_params,
                random_state=random_state + rep,
                return_model=True,
            )
            edge_records.append(
                _extract_fold_edge_record(
                    repeat_idx=rep,
                    fold_idx=fold_idx,
                    prepared_fold=prepared,
                    model=fitted_model,
                    model_type=model_type,
                )
            )

            # ---------- 6.5. 在同一个 outer fold 内即时计算 selected-space Haufe ----------
            # 这里直接复用 prepared["F_train"] 和 fitted_model，不再重复残差化或重建 yhat。
            haufe_records.append(
                _compute_fold_selected_haufe_record(
                    repeat_idx=rep,
                    fold_idx=fold_idx,
                    prepared_fold=prepared,
                    model=fitted_model,
                    model_type=model_type,
                )
            )

            y_pred[test_idx] = fold_pred

        y_pred_mat[rep] = y_pred
        y_resid_true_mat[rep] = y_resid_true

    # ---------- 7. 汇总跨 repeat 的平均预测、fold 级记录和 Haufe map ----------
    y_pred_mean = y_pred_mat.mean(axis=0)
    y_resid_true_mean = y_resid_true_mat.mean(axis=0)
    haufe_results = _summarize_haufe_records(haufe_records, n_edges=n_edges)

    return (
        y_pred_mean,
        y_resid_true_mean,
        y_pred_mat,
        y_resid_true_mat,
        edge_records,
        haufe_results,
    )


# -------------------------------------------------------------------------
# Nested CV helper functions
# -------------------------------------------------------------------------
#
# cpm_nested_cv()
#     |
#     |-- _expand_param_grid()
#     |      |-- _grid_value_list()
#     |
#     |-- outer fold loop
#            |
#            |-- _select_best_hyperparams_inner_cv()
#            |      |-- _select_and_prepare_fold_features()
#            |      |-- _fit_and_predict_prepared_fold()
#            |      |-- _score_predictions()
#            |
#            |-- _select_and_prepare_fold_features()
#            |-- _fit_and_predict_prepared_fold()
#            |-- _extract_fold_edge_record()
#
# 这组 helper 的核心目标是复用 fold 内 CPM 特征准备逻辑：inner CV 用它来选择
# 模型超参数，outer CV 再用选出的参数重新训练并预测外层测试折。


def _grid_value_list(value, name):
    """
    将 param_grid 中某个参数的候选值统一为列表。

    这是参数网格展开前的格式整理层。用户可能传入单个值、list、tuple 或
    ``np.ndarray``；后续 ``_expand_param_grid()`` 只希望面对统一的 list。
    例如 ``1.0`` 会变成 ``[1.0]``，``np.array([0.1, 1.0])`` 会变成
    ``[0.1, 1.0]``。

    Parameters
    ----------
    value : object
        单个候选值，或 list/tuple/np.ndarray 形式的候选值集合。
    name : str
        参数名称，仅用于生成错误信息。

    Returns
    -------
    values : list
        至少包含一个元素的候选值列表。

    Raises
    ------
    ValueError
        候选值集合为空时抛出。
    """
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            values = [value.item()]
        else:
            values = value.tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]

    if len(values) == 0:
        raise ValueError(f"param_grid['{name}'] must contain at least one value.")

    return values


def _expand_param_grid(param_grid, model_type, base_model_params):
    """
    展开参数网格，并生成每个候选方案对应的完整模型参数。

    本函数生成 nested CV 需要逐一评估的候选参数方案。若 ``param_grid`` 是
    dict，则对各参数候选值做笛卡尔积；若已经是 list-of-dict，则认为用户已经
    手动给出了候选组合。随后，本函数会把每个候选方案与
    ``base_model_params`` 合并，并重新执行模型参数验证。

    例如 ``{"elasticnet_alpha": [0.1, 1.0], "l1_ratio": [0.2, 0.8]}``
    会展开为四个候选组合。展开前先检查参数名是否属于当前模型，避免无意义的
    超参数进入 inner CV。

    Parameters
    ----------
    param_grid : dict or list of dict
        nested CV 的候选超参数。dict 会按键对应的候选值做笛卡尔积；list of dict
        表示用户已经手动指定候选组合。
    model_type : {'linear', 'ridge', 'lasso', 'elasticnet', 'pls', 'pcr'}
        当前预测模型类型，用于限制 ``param_grid`` 中允许出现的参数名。
    base_model_params : dict
        由用户显式参数或默认参数得到的基础模型参数。若某个候选方案没有覆盖
        某个基础参数，该基础参数会继续保留。

    Returns
    -------
    param_candidates : list of dict
        展开后的候选参数列表。每个元素只包含本次候选需要覆盖的参数；键为
        当前模型允许调节的参数名，值为用户提供的单个候选值。允许的键包括
        ``ridge_alpha``、``lasso_alpha``、``elasticnet_alpha``、``l1_ratio``、
        ``pls_components`` 和 ``pca_components`` 中与当前 ``model_type`` 匹配
        的部分。
    candidate_model_params : list of dict
        每个候选方案与 ``base_model_params`` 合并并重新验证后的完整模型参数。
        该列表与 ``param_candidates`` 一一对应，可直接传给模型构建函数。

    Raises
    ------
    ValueError
        ``param_grid`` 类型错误、候选列表为空、包含当前模型不支持的参数名，
        或合并后的模型参数不合法时抛出。
    """
    # 当前模型允许出现在 param_grid 中的超参数名称。例如 Ridge 只能调
    # ridge_alpha，ElasticNet 可以调 elasticnet_alpha 和 l1_ratio。
    allowed = set(_TUNABLE_PARAMS.get(model_type, set()))

    if isinstance(param_grid, dict):
        # set(param_grid) 取出用户提供的参数名；减去 allowed 后，剩下的就是
        # 当前模型不支持的参数，例如 ridge 模型下的 lasso_alpha。
        unknown = set(param_grid) - allowed
        if unknown:
            raise ValueError(
                "param_grid contains unsupported parameter(s) for "
                f"model_type='{model_type}': {sorted(unknown)}."
            )

        if len(param_grid) == 0:
            # 空 dict 表示不覆盖任何基础参数，因此只有一个“空候选方案”。
            candidates = [{}]
        else:
            # keys 保存参数名顺序；value_lists 保存每个参数对应的候选值列表。
            # 例如 ElasticNet 中：
            # keys = ["elasticnet_alpha", "l1_ratio"]
            # value_lists = [[0.1, 1.0], [0.2, 0.8]]
            keys = list(param_grid)
            value_lists = [_grid_value_list(param_grid[key], key) for key in keys]

            # product(*value_lists) 生成所有候选值组合；zip(keys, values) 再把
            # 每组候选值配回参数名，形成一个完整的候选参数字典。
            # 若 values = (0.1, 0.2)，则 zip(keys, values) 相当于：
            # [("elasticnet_alpha", 0.1), ("l1_ratio", 0.2)]，
            # dict(...) 得到 {"elasticnet_alpha": 0.1, "l1_ratio": 0.2}。
            candidates = [dict(zip(keys, values)) for values in product(*value_lists)]

    elif isinstance(param_grid, (list, tuple)):
        # list-of-dict 表示用户已经手动指定候选组合，不再做笛卡尔积展开。
        candidates = []
        for candidate in param_grid:
            if not isinstance(candidate, dict):
                raise ValueError("Each item in param_grid must be a dict.")
            # 每个手动候选组合也要检查参数名是否适用于当前模型。
            unknown = set(candidate) - allowed
            if unknown:
                raise ValueError(
                    "param_grid contains unsupported parameter(s) for "
                    f"model_type='{model_type}': {sorted(unknown)}."
                )
            candidates.append(candidate.copy())

    else:
        raise ValueError("param_grid must be a dict or a list of dicts.")

    if len(candidates) == 0:
        raise ValueError("param_grid must contain at least one candidate.")

    if model_type == "linear" and any(candidates):
        raise ValueError("model_type='linear' has no tunable model hyperparameters.")

    candidate_model_params = []
    for candidate in candidates:
        # base_model_params 是当前模型的基础配置。copy() 确保候选参数只覆盖当前
        # 这一份 merged，不会改动循环中反复复用的 base_model_params。
        merged = base_model_params.copy()
        if candidate:
            # candidate 是一个候选 dict，例如 {"l1_ratio": 0.8}；update() 只
            # 覆盖同名键，未出现在 candidate 中的基础参数会保留下来。
            merged.update(candidate)

        # 合并后重新验证，确保每个实际进入模型的候选参数都满足类型和取值范围
        # 要求。例如 ElasticNet 的 l1_ratio 必须落在 [0, 1]。
        _, merged = _validate_model_params(model_type=model_type, **merged)
        candidate_model_params.append(merged)

    return candidates, candidate_model_params


def _score_predictions(y_true, y_pred, scoring):
    """
    计算 nested CV 内层验证集的调参分数；分数越大越好。

    该辅助函数用于比较候选参数的验证集表现。为了统一选择逻辑，所有指标都被
    转换成“越大越好”。因此 RMSE 和 MAE 这类本来应越小越好的误差指标，在这里
    返回负值，即 ``neg_rmse`` 和 ``neg_mae``。

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        内层验证样本的真实目标残差。
    y_pred : array-like, shape (n_samples,)
        与 ``y_true`` 对应的预测值。
    scoring : {'pearson_r', 'r2', 'neg_rmse', 'neg_mae'}
        内层 CV 用于选择超参数的指标。误差类指标返回负值，使所有指标都遵循
        “越大越好”的选择规则。

    Returns
    -------
    score : float
        当前候选参数的验证分数。若相关或 R² 因零方差无法定义，则返回 ``np.nan``。

    Raises
    ------
    ValueError
        ``y_true`` 与 ``y_pred`` 形状不一致，或 ``scoring`` 不受支持时抛出。
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    scoring = str(scoring).lower()

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if scoring == "pearson_r":
        if y_true.size < 2 or np.var(y_true) <= _EPS or np.var(y_pred) <= _EPS:
            return np.nan
        return float(stats.pearsonr(y_true, y_pred)[0])

    errors = y_pred - y_true

    if scoring == "neg_rmse":
        return float(-np.sqrt(np.mean(errors**2)))

    if scoring == "neg_mae":
        return float(-np.mean(np.abs(errors)))

    if scoring == "r2":
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_total <= _EPS:
            return np.nan
        ss_res = np.sum(errors**2)
        return float(1.0 - ss_res / ss_total)

    raise ValueError("scoring must be 'pearson_r', 'r2', 'neg_rmse', or 'neg_mae'.")


def _select_best_hyperparams_inner_cv(
    X,
    y,
    Covariates,
    param_candidates,
    candidate_model_params,
    model_type,
    inner_cv_type,
    inner_n_splits,
    random_state,
    selection_params,
    standardize_features,
    scoring,
):
    """
    在一个 outer training set 内运行 inner CV，并返回最佳模型参数。

    同一个 inner fold 的 CPM 选边和特征构建只执行一次；随后对所有候选参数重复
    模型拟合和预测。这是 nested CV 比反复调用固定参数 CV 更高效的关键。

    直观流程如下：

    1. 把当前 outer training set 再切分为 inner train 和 inner validation；
    2. 在每个 inner train 中完成 CPM 的训练折步骤，包括协变量残差化、偏相关、
       选边和特征构建；
    3. 对同一份 inner fold 特征，分别用所有候选模型参数拟合并预测
       inner validation；
    4. 将各 inner validation 的真实残差和预测值拼接成一组完整的 inner
       out-of-fold 预测；
    5. 用 ``_score_predictions()`` 为每组候选参数的 out-of-fold 预测打分；
    6. 返回 inner CV 分数最高的参数，供外层 fold 重新训练和预测测试折。

    关键原则是：本函数接收的数据只来自某个 outer training set，因此 inner CV
    的调参过程不会接触 outer test set。候选参数的分数基于拼接后的 inner
    out-of-fold 预测，而不是先计算每个 inner fold 的分数再取平均；这样可使
    Pearson r 和 R² 等依赖整体方差结构的指标在同一批 out-of-fold 预测上计算。

    Parameters
    ----------
    X : np.ndarray, shape (n_outer_train, n_edges)
        当前 outer training set 的连接边矩阵。
    y : np.ndarray, shape (n_outer_train,)
        当前 outer training set 的连续目标变量。
    Covariates : np.ndarray, shape (n_outer_train, n_Covariates)
        当前 outer training set 的协变量矩阵；无协变量时为零列矩阵。
    param_candidates : list of dict
        从 ``param_grid`` 展开的原始候选参数，用于记录调参结果。
    candidate_model_params : list of dict
        每个候选参数与基础参数合并并验证后的模型参数。
    model_type : {'linear', 'ridge', 'lasso', 'elasticnet', 'pls', 'pcr'}
        预测模型类型。
    inner_cv_type : {'kfold', 'loocv'}
        内层交叉验证类型。
    inner_n_splits : int
        内层 KFold 折数，仅在 ``inner_cv_type='kfold'`` 时使用。
    random_state : int
        内层 KFold 切分和随机模型使用的随机种子。
    selection_params : dict
        固定的 CPM 选边参数，会传递给
        ``_select_and_prepare_fold_features()``。
    standardize_features : bool
        是否在每个 inner fold 内对训练特征和验证特征进行训练集标准化。
    scoring : {'pearson_r', 'r2', 'neg_rmse', 'neg_mae'}
        内层 CV 的调参指标。

    Returns
    -------
    best_model_params : dict
        inner out-of-fold 分数最高的模型参数。若所有候选分数均为 ``np.nan``，则返回
        第一个候选参数。
    best_score : float
        最佳候选参数对应的内层验证分数；若所有候选均无法评分，则为 ``np.nan``。
    candidate_scores : list of dict
        每个候选参数的原始候选值、完整模型参数和内层验证分数。每个 dict 包含：

        - ``candidate_params`` : dict
          从 ``param_grid`` 展开的原始候选参数，只包含本候选显式覆盖的键。
        - ``model_params`` : dict
          与基础参数合并并重新验证后的完整模型参数。
        - ``score`` : float
          当前候选参数的 inner out-of-fold 分数；无法评分时为 ``np.nan``。
    n_splits_actual : int
        内层 CV 实际使用的折数；LOOCV 时等于当前 outer training set 的样本量。

    Raises
    ------
    ValueError
        候选参数列表为空、两个候选列表长度不一致、内层 CV 类型/折数无效，
        或 fold 内 CPM 特征准备失败时抛出。
    """
    n_sub = X.shape[0]
    inner_cv_type = str(inner_cv_type).lower()

    if len(param_candidates) == 0:
        raise ValueError("param_candidates must contain at least one candidate.")
    if len(param_candidates) != len(candidate_model_params):
        raise ValueError(
            "param_candidates and candidate_model_params must have the same length."
        )

    if inner_cv_type == "loocv":
        n_splits_actual = n_sub
        cv_iter = _make_cv_iterator("loocv", n_splits_actual, X)
    elif inner_cv_type == "kfold":
        n_splits_actual = _validate_integer(
            inner_n_splits,
            "inner_n_splits",
            minimum=2,
        )
        if n_splits_actual > n_sub:
            raise ValueError("inner_n_splits cannot be larger than outer train size.")
        cv_iter = _make_cv_iterator(
            "kfold",
            n_splits_actual,
            X,
            random_state=random_state,
        )
    else:
        raise ValueError("inner_cv_type must be 'kfold' or 'loocv'.")

    true_parts = []
    pred_parts = [[] for _ in candidate_model_params]

    for inner_fold_idx, (train_idx, valid_idx) in enumerate(cv_iter):
        prepared = _select_and_prepare_fold_features(
            X_train=X[train_idx],
            y_train=y[train_idx],
            Covariates_train=Covariates[train_idx],
            X_test=X[valid_idx],
            y_test=y[valid_idx],
            Covariates_test=Covariates[valid_idx],
            model_type=model_type,
            standardize_features=standardize_features,
            **selection_params,
        )
        true_parts.append(prepared["y_resid_true_test"])

        for candidate_idx, model_params in enumerate(candidate_model_params):
            pred = _fit_and_predict_prepared_fold(
                prepared,
                model_type=model_type,
                model_params=model_params,
                random_state=random_state + inner_fold_idx,
            )
            pred_parts[candidate_idx].append(pred)

    y_true_inner = np.concatenate(true_parts)
    scores = []
    for candidate_idx in range(len(candidate_model_params)):
        y_pred_inner = np.concatenate(pred_parts[candidate_idx])
        scores.append(_score_predictions(y_true_inner, y_pred_inner, scoring))

    finite_scores = np.asarray(scores, dtype=float)
    finite_mask = np.isfinite(finite_scores)
    if np.any(finite_mask):
        best_idx = int(
            np.flatnonzero(finite_mask)[np.argmax(finite_scores[finite_mask])]
        )
        best_score = float(finite_scores[best_idx])
    else:
        best_idx = 0
        best_score = np.nan

    candidate_scores = []
    for candidate, model_params, score in zip(
        param_candidates,
        candidate_model_params,
        scores,
    ):
        candidate_scores.append(
            {
                "candidate_params": candidate.copy(),
                "model_params": model_params.copy(),
                "score": float(score) if np.isfinite(score) else np.nan,
            }
        )

    return (
        candidate_model_params[best_idx],
        best_score,
        candidate_scores,
        n_splits_actual,
    )


def cpm_nested_cv(
    X,
    y,
    Covariates,
    param_grid,
    # ---------- 外层交叉验证参数 ----------
    n_splits=10,
    n_repeats=50,
    random_state=0,
    cv_type="kfold",
    # ---------- 内层交叉验证参数 ----------
    inner_n_splits=5,
    inner_cv_type="kfold",
    scoring="pearson_r",
    # ---------- 固定选边参数 ----------
    selection_method="pvalue",
    p_thresh=0.01,
    top_k=None,
    top_k_mode="total",
    max_edges_total=None,
    min_edges_total=0,
    fallback_if_too_few=False,
    max_edges_per_sign=None,
    # ---------- 基础模型参数 ----------
    ridge_alpha=1.0,
    lasso_alpha=1.0,
    elasticnet_alpha=1.0,
    l1_ratio=0.5,
    model_type="ridge",
    pls_components=None,
    pca_components=None,
    # ---------- 特征处理参数 ----------
    standardize_features=True,
    # ---------- 输出控制 ----------
    verbose=False,
):
    """
    使用 nested CV 选择模型超参数，并返回外层交叉验证预测结果。

    本函数与 ``cpm_fixed_params_cv()`` 使用相同的 CPM 数据处理流程，但会在每个
    outer training fold 内再运行一层 inner CV，从 ``param_grid`` 中选择当前外层
    训练集上的最佳模型超参数。选边参数（例如 ``p_thresh``、``top_k``）在本函数
    中仍然是固定值；当前 nested CV 只优选模型层面的超参数，例如
    ``ridge_alpha``、``lasso_alpha``、``elasticnet_alpha``、``l1_ratio``、
    ``pls_components`` 或 ``pca_components``。

    几个 nested CV 辅助函数的分工如下：

    - ``_expand_param_grid()``：生成“要尝试哪些候选参数方案”，并把候选参数
      与基础参数合并成可建模的完整配置；内部会检查合法参数名，并用
      ``_grid_value_list()`` 统一候选值格式；
    - ``_select_best_hyperparams_inner_cv()``：只在 outer training set 内比较候选
      参数，并返回最佳参数；它会复用
      ``_select_and_prepare_fold_features()`` 和
      ``_fit_and_predict_prepared_fold()``；
    - ``_score_predictions()``：用统一的“越大越好”规则给 inner CV 预测打分。
    - ``_extract_fold_edge_record()``：提取每个 outer fold 用最佳参数拟合后的
      模型系数，并与选边结果合并为统一的 ``edge_records``。

    计算流程为：

    1. 在每个 outer training fold 内运行 inner CV；
    2. inner CV 的每个 fold 只准备一次 CPM 特征，然后对所有候选超参数重复拟合；
    3. 选择 inner out-of-fold 预测表现最好的候选参数；
    4. 用该参数在完整 outer training fold 上重新训练，并预测 outer test fold；
    5. 保存 outer fold 的统一 edge 记录和最佳参数记录。

    Parameters
    ----------
    X : array-like, shape (n_sub, n_edges)
        被试乘连接边的特征矩阵。当前实现要求所有值有限，不在函数内插补缺失值。
    y : array-like, shape (n_sub,)
        连续预测目标。模型实际拟合的是训练折内去除协变量后的 y 残差。
    Covariates : array-like or None, shape (n_sub, n_Covariates)
        需要控制的协变量，不包含截距；传入 None 表示只控制截距。
    param_grid : dict or list of dict
        待优选的模型超参数范围。dict 会展开为笛卡尔积；list of dict 则按给定
        组合逐一评估。允许的键取决于 ``model_type``：

        - ``'ridge'``：``ridge_alpha``；
        - ``'lasso'``：``lasso_alpha``；
        - ``'elasticnet'``：``elasticnet_alpha``、``l1_ratio``；
        - ``'pls'``：``pls_components``；
        - ``'pcr'``：``pca_components``。

    inner_n_splits : int, default=5
        内层 K 折 CV 的折数，仅在 ``inner_cv_type='kfold'`` 时使用。
    inner_cv_type : {'kfold', 'loocv'}, default='kfold'
        内层交叉验证方案。
    scoring : {'pearson_r', 'r2', 'neg_rmse', 'neg_mae'}, default='pearson_r'
        内层 CV 的调参指标。所有指标均按“越大越好”选择最佳参数。
    n_splits : int, default=10
        外层 K 折交叉验证的折数，仅在 ``cv_type='kfold'`` 时使用。
    n_repeats : int, default=50
        外层 K 折随机划分的重复次数。LOOCV 无随机划分，因此该参数会被忽略。
    random_state : int, default=0
        外层 CV、内层 CV 以及随机模型使用的基础随机种子。
    cv_type : {'kfold', 'loocv'}, default='kfold'
        外层交叉验证方案。
    selection_method : {'pvalue', 'topk'}, default='pvalue'
        训练折内的边选择方法，具体规则由 ``select_edges()`` 实现。
    p_thresh : float, default=0.01
        p 值选边阈值，仅在 ``selection_method='pvalue'`` 时使用。
    top_k : int or None, default=None
        Top-K 模式的选边数；``top_k_mode='per_sign'`` 时表示每个方向的数量。
    top_k_mode : {'total', 'per_sign'}, default='total'
        Top-K 是在全部边中选择，还是分别在正边和负边中选择。
    max_edges_total : int or None, default=None
        ``select_edges()`` 返回的最大总边数。
    min_edges_total : int, default=0
        启用 fallback 时希望保留的最少总边数。
    fallback_if_too_few : bool, default=False
        初始选边不足时，是否回退为按 p 值选择 ``min_edges_total`` 条边。
    max_edges_per_sign : int or None, default=None
        正边和负边各自的上限。该限制在 fallback 之后执行。
    ridge_alpha, lasso_alpha, elasticnet_alpha : float
        固定基础参数；若对应参数不在 ``param_grid`` 中，会作为该模型的默认值。
    l1_ratio : float, default=0.5
        ElasticNet 的 L1/L2 混合比例，可在 ``param_grid`` 中覆盖。
    model_type : str, default='ridge'
        预测模型及特征组织方式。可选 ``'ridge'``、``'lasso'``、
        ``'elasticnet'``、``'pls'`` 或 ``'pcr'``。``linear`` 无模型超参数，
        因而通常应使用 ``cpm_fixed_params_cv()``。
    pls_components, pca_components : int or None
        PLS/PCA 的基础成分数；若对应参数不在 ``param_grid`` 中，会作为默认值。
    standardize_features : bool, default=True
        是否在每个 fold 内使用训练折均值和标准差标准化预测特征。
    verbose : bool, default=False
        是否打印外层重复交叉验证进度。

    Returns
    -------
    y_pred_mean : np.ndarray, shape (n_sub,)
        所有外层重复的折外预测平均值，目标尺度为 y 的训练折协变量残差。
    y_resid_true_mean : np.ndarray, shape (n_sub,)
        所有外层重复中折特异真实 y 残差的平均值。
    y_pred_mat : np.ndarray, shape (n_repeats_actual, n_sub)
        每次外层重复的完整折外预测矩阵；LOOCV 时第一维固定为 1。
    y_resid_true_mat : np.ndarray, shape (n_repeats_actual, n_sub)
        每次外层重复对应的真实 y 残差矩阵，应与 ``y_pred_mat`` 逐行配对评估。
    edge_records : list of dict
        每个 outer fold 的统一 edge 记录。每个 dict 主要包含：

        - ``repeat`` / ``fold`` : int
          当前记录对应的 outer repeat 和 outer fold 零基编号。
        - ``model_type`` : str
          当前 fold 使用的预测模型类型。
        - ``feature_names`` : np.ndarray, shape (n_features_kept,)
          与 ``coef`` 一一对应的模型特征名称。
        - ``coef`` : np.ndarray, shape (n_features_kept,)
          当前 fold 模型在实际特征空间中的系数。
        - ``edge_idx`` : np.ndarray, shape (n_features_kept,) or (0,)
          高维模型中与 ``coef`` 对应的 edge 索引；标准 CPM 中为空数组。
        - ``pos_idx`` / ``neg_idx`` : np.ndarray
          当前 fold 被 CPM 选中的正边和负边索引。
        - ``n_pos`` / ``n_neg`` : int
          当前 fold 被 CPM 选中的正边数和负边数。
        - ``used_fallback`` / ``has_features`` : bool
          当前 fold 是否触发 fallback，以及是否存在有效建模特征。
    selected_param_records : list of dict
        每个 outer fold 的调参记录，包括 repeat、fold、最佳参数、最佳 inner
        score，以及所有候选参数的 inner score。每个 dict 主要包含：

        - ``repeat`` / ``fold`` : int
          当前 outer repeat 和 outer fold 的零基编号。
        - ``best_params`` : dict
          inner CV 选出的最佳模型参数，只包含当前模型实际使用的键。
        - ``inner_score`` : float
          最佳参数对应的 inner out-of-fold 调参分数；无法评分时为 ``np.nan``。
        - ``scoring`` : str
          inner CV 使用的评分指标。
        - ``inner_cv_type`` : str
          inner CV 类型，取值为 ``'kfold'`` 或 ``'loocv'``。
        - ``inner_n_splits_actual`` : int
          inner CV 实际使用的折数。
        - ``candidate_scores`` : list of dict
          所有候选参数的评分记录。每个元素包含 ``candidate_params`` : dict、
          ``model_params`` : dict 和 ``score`` : float。

    Raises
    ------
    ValueError
        输入形状、外层/内层 CV 参数、模型参数、``param_grid`` 或 fold 内残差自由度
        无效时抛出。

    Notes
    -----
    nested CV 的作用是无泄漏地选择模型超参数。外层测试折不参与 inner CV 调参，
    只用于最终评估。由于每个 outer fold 会额外运行 inner CV，本函数计算量明显
    大于固定参数版本。

    对 Ridge、Lasso 和 ElasticNet 等高维模型，``edge_records`` 同时描述某条
    edge 是否被 CPM 选中，以及该 edge 在最终预测模型中的系数。若 ``edge_idx``
    非空，``coef`` 即为对应 edge 的权重。建议同时查看入选频率、平均权重、
    平均绝对权重和符号一致性。PLS/PCR 的 ``coef`` 为成分模型回投到当前 fold
    特征空间后的系数，解释应更谨慎。
    """
    # ---------- 1. 输入数据和基础模型参数验证 ----------
    # base_model_params 是当前模型的默认/基础配置；param_grid 中的候选值会在
    # outer fold 内覆盖这些基础值，再由 inner CV 选择最佳组合。
    X, y, Covariates, n_sub, _ = _validate_X_y_Covariates(
        X,
        y,
        Covariates,
        minimum_n_sub=3,
    )
    model_type, base_model_params = _validate_model_params(
        model_type=model_type,
        ridge_alpha=ridge_alpha,
        lasso_alpha=lasso_alpha,
        elasticnet_alpha=elasticnet_alpha,
        l1_ratio=l1_ratio,
        pls_components=pls_components,
        pca_components=pca_components,
    )
    random_state = _validate_integer(random_state, "random_state")
    standardize_features = _validate_boolean(
        standardize_features,
        "standardize_features",
    )
    verbose = _validate_boolean(verbose, "verbose")

    # ---------- 2. 验证 inner CV 的评分指标 ----------
    # _score_predictions() 统一采用“越大越好”的规则，因此误差指标使用负值形式。
    scoring = str(scoring).lower()
    if scoring not in {"pearson_r", "r2", "neg_rmse", "neg_mae"}:
        raise ValueError("scoring must be 'pearson_r', 'r2', 'neg_rmse', or 'neg_mae'.")

    # ---------- 3. 验证 outer CV 设置 ----------
    # outer test fold 是最终评估层，不能参与 inner CV 的超参数选择。
    cv_type, n_repeats_actual, n_splits_actual = _validate_cv_config(
        cv_type=cv_type,
        n_splits=n_splits,
        n_repeats=n_repeats,
        n_sub=n_sub,
    )
    if verbose and cv_type == "loocv" and n_repeats != 1:
        print("cv_type='loocv': n_repeats is ignored and set to 1.")

    # ---------- 4. 展开并验证待搜索的模型超参数候选 ----------
    # param_candidates 保留用户提供的“覆盖项”；candidate_model_params 是与基础参数
    # 合并并重新验证后的完整模型参数，供 inner CV 直接建模使用。
    param_candidates, candidate_model_params = _expand_param_grid(
        param_grid,
        model_type,
        base_model_params,
    )

    # ---------- 5. 汇总 fold 内固定选边参数 ----------
    # 当前 nested CV 只调模型层超参数；CPM 选边规则在 outer/inner fold 中保持固定，
    # 但实际选边仍然必须在每个训练折内重新执行。
    selection_params = {
        "selection_method": selection_method,
        "p_thresh": p_thresh,
        "top_k": top_k,
        "top_k_mode": top_k_mode,
        "max_edges_total": max_edges_total,
        "min_edges_total": min_edges_total,
        "fallback_if_too_few": fallback_if_too_few,
        "max_edges_per_sign": max_edges_per_sign,
    }

    # ---------- 6. 初始化 outer CV 输出容器 ----------
    # selected_param_records 记录每个 outer fold 的 inner CV 选择结果；
    # edge_records 记录使用最佳参数重训 outer training fold 后的选边和模型系数。
    y_pred_mat = np.zeros((n_repeats_actual, n_sub), dtype=float)
    y_resid_true_mat = np.zeros((n_repeats_actual, n_sub), dtype=float)
    edge_records = []
    selected_param_records = []

    for rep in range(n_repeats_actual):
        if verbose:
            print(f"Outer repeat {rep + 1}/{n_repeats_actual}")

        # ---------- 7. 当前 repeat 的 outer CV 切分 ----------
        cv_iter = _make_cv_iterator(
            cv_type,
            n_splits_actual,
            X,
            random_state=random_state + rep,
        )
        y_pred = np.zeros(n_sub, dtype=float)
        y_resid_true = np.zeros(n_sub, dtype=float)

        for fold_idx, (train_idx, test_idx) in enumerate(cv_iter):
            # ---------- 8. 在当前 outer training set 内运行 inner CV ----------
            # inner_seed 随 repeat/fold 改变，使每个 outer fold 的内层 KFold 可复现且
            # 不完全相同。inner CV 完全看不到 outer test fold。
            inner_seed = random_state + rep * 1000 + fold_idx
            (
                best_model_params,
                best_score,
                candidate_scores,
                inner_n_splits_actual,
            ) = _select_best_hyperparams_inner_cv(
                X=X[train_idx],
                y=y[train_idx],
                Covariates=Covariates[train_idx],
                param_candidates=param_candidates,
                candidate_model_params=candidate_model_params,
                model_type=model_type,
                inner_cv_type=inner_cv_type,
                inner_n_splits=inner_n_splits,
                random_state=inner_seed,
                selection_params=selection_params,
                standardize_features=standardize_features,
                scoring=scoring,
            )

            # ---------- 9. 用最佳参数重新准备 outer fold 特征 ----------
            # 这里重新使用完整 outer training fold 做协变量回归和选边，再预测
            # outer test fold；这是 nested CV 的最终评估层。
            prepared = _select_and_prepare_fold_features(
                X_train=X[train_idx],
                y_train=y[train_idx],
                Covariates_train=Covariates[train_idx],
                X_test=X[test_idx],
                y_test=y[test_idx],
                Covariates_test=Covariates[test_idx],
                model_type=model_type,
                standardize_features=standardize_features,
                **selection_params,
            )

            y_resid_true[test_idx] = prepared["y_resid_true_test"]
            # ---------- 10. 使用 inner CV 选出的最佳参数预测 outer test fold ----------
            # 保留 fitted model，并将最终模型系数与选边结果合并为 edge_records。
            # 这些系数来自 outer training fold 的重训模型，而不是 inner fold 中的
            # 临时模型。
            fold_pred, fitted_model = _fit_and_predict_prepared_fold(
                prepared,
                model_type=model_type,
                model_params=best_model_params,
                random_state=random_state + rep,
                return_model=True,
            )
            edge_records.append(
                _extract_fold_edge_record(
                    repeat_idx=rep,
                    fold_idx=fold_idx,
                    prepared_fold=prepared,
                    model=fitted_model,
                    model_type=model_type,
                )
            )

            y_pred[test_idx] = fold_pred

            # ---------- 11. 保存当前 outer fold 的调参过程摘要 ----------
            # candidate_scores 保留所有候选参数表现，best_params 是当前 outer fold
            # 真正用于最终测试折预测的参数。
            selected_param_records.append(
                {
                    "repeat": rep,
                    "fold": fold_idx,
                    "best_params": best_model_params.copy(),
                    "inner_score": best_score,
                    "scoring": scoring,
                    "inner_cv_type": str(inner_cv_type).lower(),
                    "inner_n_splits_actual": inner_n_splits_actual,
                    "candidate_scores": candidate_scores,
                }
            )

        # 当前 outer repeat 完成后，保存完整折外预测和折特异真实残差。
        y_pred_mat[rep] = y_pred
        y_resid_true_mat[rep] = y_resid_true

    # ---------- 12. 汇总 nested CV 输出 ----------
    y_pred_mean = y_pred_mat.mean(axis=0)
    y_resid_true_mean = y_resid_true_mat.mean(axis=0)

    return (
        y_pred_mean,
        y_resid_true_mean,
        y_pred_mat,
        y_resid_true_mat,
        edge_records,
        selected_param_records,
    )


def cpm_nested_cv_with_haufe(
    X,
    y,
    Covariates,
    param_grid,
    # ---------- 外层交叉验证参数 ----------
    n_splits=10,
    n_repeats=50,
    random_state=0,
    cv_type="kfold",
    # ---------- 内层交叉验证参数 ----------
    inner_n_splits=5,
    inner_cv_type="kfold",
    scoring="pearson_r",
    # ---------- 固定选边参数 ----------
    selection_method="pvalue",
    p_thresh=0.01,
    top_k=None,
    top_k_mode="total",
    max_edges_total=None,
    min_edges_total=0,
    fallback_if_too_few=False,
    max_edges_per_sign=None,
    # ---------- 基础模型参数 ----------
    ridge_alpha=1.0,
    lasso_alpha=1.0,
    elasticnet_alpha=1.0,
    l1_ratio=0.5,
    model_type="ridge",
    pls_components=None,
    pca_components=None,
    # ---------- 特征处理参数 ----------
    standardize_features=True,
    # ---------- 输出控制 ----------
    verbose=False,
):
    """
    使用 nested CV 选择模型超参数，并在 outer fold 内同步计算 selected-space Haufe。

    本函数与 ``cpm_nested_cv()`` 使用相同的 nested CV 流程和相同的前 6 个返回值。
    区别是：每个 outer fold 完成 inner CV 选参、并用最佳参数在完整 outer training
    fold 上重训模型后，立即基于该 fold 的 ``F_train`` 和 fitted model 计算
    selected-space Haufe feature。

    当前第一版只支持高维扩展 CPM 模型：``ridge``、``lasso``、``elasticnet``、
    ``pls`` 和 ``pcr``；暂不支持 ``model_type='linear'``。

    Returns
    -------
    与 ``cpm_nested_cv()`` 相同的前 6 个返回值，后面追加：

    haufe_results : dict
        selected-space Haufe 汇总结果。字段与
        ``cpm_fixed_params_cv_with_haufe()`` 返回的 ``haufe_results`` 一致。
    """
    # ---------- 1. 输入数据和基础模型参数验证 ----------
    X, y, Covariates, n_sub, n_edges = _validate_X_y_Covariates(
        X,
        y,
        Covariates,
        minimum_n_sub=3,
    )
    model_type, base_model_params = _validate_model_params(
        model_type=model_type,
        ridge_alpha=ridge_alpha,
        lasso_alpha=lasso_alpha,
        elasticnet_alpha=elasticnet_alpha,
        l1_ratio=l1_ratio,
        pls_components=pls_components,
        pca_components=pca_components,
    )
    _validate_haufe_supported_model_type(model_type)
    random_state = _validate_integer(random_state, "random_state")
    standardize_features = _validate_boolean(
        standardize_features,
        "standardize_features",
    )
    verbose = _validate_boolean(verbose, "verbose")

    # ---------- 2. 验证 inner CV 的评分指标 ----------
    scoring = str(scoring).lower()
    if scoring not in {"pearson_r", "r2", "neg_rmse", "neg_mae"}:
        raise ValueError("scoring must be 'pearson_r', 'r2', 'neg_rmse', or 'neg_mae'.")

    # ---------- 3. 验证 outer CV 设置 ----------
    cv_type, n_repeats_actual, n_splits_actual = _validate_cv_config(
        cv_type=cv_type,
        n_splits=n_splits,
        n_repeats=n_repeats,
        n_sub=n_sub,
    )
    if verbose and cv_type == "loocv" and n_repeats != 1:
        print("cv_type='loocv': n_repeats is ignored and set to 1.")

    # ---------- 4. 展开并验证待搜索的模型超参数候选 ----------
    param_candidates, candidate_model_params = _expand_param_grid(
        param_grid,
        model_type,
        base_model_params,
    )

    # ---------- 5. 汇总 fold 内固定选边参数 ----------
    selection_params = {
        "selection_method": selection_method,
        "p_thresh": p_thresh,
        "top_k": top_k,
        "top_k_mode": top_k_mode,
        "max_edges_total": max_edges_total,
        "min_edges_total": min_edges_total,
        "fallback_if_too_few": fallback_if_too_few,
        "max_edges_per_sign": max_edges_per_sign,
    }

    # ---------- 6. 初始化 outer CV 输出容器 ----------
    y_pred_mat = np.zeros((n_repeats_actual, n_sub), dtype=float)
    y_resid_true_mat = np.zeros((n_repeats_actual, n_sub), dtype=float)
    edge_records = []
    selected_param_records = []
    haufe_records = []

    for rep in range(n_repeats_actual):
        if verbose:
            print(f"Outer repeat {rep + 1}/{n_repeats_actual}")

        # ---------- 7. 当前 repeat 的 outer CV 切分 ----------
        cv_iter = _make_cv_iterator(
            cv_type,
            n_splits_actual,
            X,
            random_state=random_state + rep,
        )
        y_pred = np.zeros(n_sub, dtype=float)
        y_resid_true = np.zeros(n_sub, dtype=float)

        for fold_idx, (train_idx, test_idx) in enumerate(cv_iter):
            # ---------- 8. 在当前 outer training set 内运行 inner CV ----------
            inner_seed = random_state + rep * 1000 + fold_idx
            (
                best_model_params,
                best_score,
                candidate_scores,
                inner_n_splits_actual,
            ) = _select_best_hyperparams_inner_cv(
                X=X[train_idx],
                y=y[train_idx],
                Covariates=Covariates[train_idx],
                param_candidates=param_candidates,
                candidate_model_params=candidate_model_params,
                model_type=model_type,
                inner_cv_type=inner_cv_type,
                inner_n_splits=inner_n_splits,
                random_state=inner_seed,
                selection_params=selection_params,
                standardize_features=standardize_features,
                scoring=scoring,
            )

            # ---------- 9. 用最佳参数重新准备 outer fold 特征 ----------
            prepared = _select_and_prepare_fold_features(
                X_train=X[train_idx],
                y_train=y[train_idx],
                Covariates_train=Covariates[train_idx],
                X_test=X[test_idx],
                y_test=y[test_idx],
                Covariates_test=Covariates[test_idx],
                model_type=model_type,
                standardize_features=standardize_features,
                **selection_params,
            )

            y_resid_true[test_idx] = prepared["y_resid_true_test"]

            # ---------- 10. 使用 inner CV 选出的最佳参数预测 outer test fold ----------
            fold_pred, fitted_model = _fit_and_predict_prepared_fold(
                prepared,
                model_type=model_type,
                model_params=best_model_params,
                random_state=random_state + rep,
                return_model=True,
            )
            edge_records.append(
                _extract_fold_edge_record(
                    repeat_idx=rep,
                    fold_idx=fold_idx,
                    prepared_fold=prepared,
                    model=fitted_model,
                    model_type=model_type,
                )
            )

            # ---------- 10.5. 在最终 outer 模型上即时计算 selected-space Haufe ----------
            # Haufe 使用的是 outer training fold 的最终模型，而不是 inner fold 临时模型。
            haufe_records.append(
                _compute_fold_selected_haufe_record(
                    repeat_idx=rep,
                    fold_idx=fold_idx,
                    prepared_fold=prepared,
                    model=fitted_model,
                    model_type=model_type,
                )
            )

            y_pred[test_idx] = fold_pred

            # ---------- 11. 保存当前 outer fold 的调参过程摘要 ----------
            selected_param_records.append(
                {
                    "repeat": rep,
                    "fold": fold_idx,
                    "best_params": best_model_params.copy(),
                    "inner_score": best_score,
                    "scoring": scoring,
                    "inner_cv_type": str(inner_cv_type).lower(),
                    "inner_n_splits_actual": inner_n_splits_actual,
                    "candidate_scores": candidate_scores,
                }
            )

        y_pred_mat[rep] = y_pred
        y_resid_true_mat[rep] = y_resid_true

    # ---------- 12. 汇总 nested CV 输出和 selected-space Haufe map ----------
    y_pred_mean = y_pred_mat.mean(axis=0)
    y_resid_true_mean = y_resid_true_mat.mean(axis=0)
    haufe_results = _summarize_haufe_records(haufe_records, n_edges=n_edges)

    return (
        y_pred_mean,
        y_resid_true_mean,
        y_pred_mat,
        y_resid_true_mat,
        edge_records,
        selected_param_records,
        haufe_results,
    )


def _compute_regression_metrics(y_true_row, y_pred_row):
    """
    计算一组对应真实值与预测值的回归指标。

    Parameters
    ----------
    y_true_row : array-like, shape (n_sub,)
        一次 repeat 或平均预测对应的真实目标值。
    y_pred_row : array-like, shape (n_sub,)
        与 ``y_true_row`` 一一对应的预测值。

    Returns
    -------
    metrics : dict
        单组预测的回归指标字典，包含：

        - ``pearson_r`` : float
          Pearson 相关系数。若真实值或预测值方差近似为 0，则返回 ``np.nan``。
        - ``r2`` : float
          决定系数。若真实值方差近似为 0，则返回 ``np.nan``。
        - ``mae`` : float
          平均绝对误差。
        - ``rmse`` : float
          均方根误差。
    """
    y_true_row = np.asarray(y_true_row, dtype=float).ravel()
    y_pred_row = np.asarray(y_pred_row, dtype=float).ravel()

    # 误差定义为预测值减真实值；MAE/RMSE 都是“越小越好”的绝对误差指标。
    # RMSE 对大误差更敏感。
    errors = y_pred_row - y_true_row
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))

    # R² = 1 - 残差平方和 / 总平方和。若真实值几乎没有方差，分母接近 0，
    # R² 没有稳定含义，因此返回 NaN。
    ss_res = np.sum(errors**2)
    ss_total = np.sum((y_true_row - np.mean(y_true_row)) ** 2)
    r2 = np.nan if ss_total <= _EPS else 1.0 - ss_res / ss_total

    # Pearson r 衡量真实值和预测值的线性关联，要求两者都存在非零方差。
    # 若某一边几乎是常量，相关系数没有定义。
    if np.var(y_true_row) <= _EPS or np.var(y_pred_row) <= _EPS:
        pearson_r = np.nan
    else:
        pearson_r = stats.pearsonr(y_true_row, y_pred_row)[0]
        pearson_r = float(pearson_r)

    return {
        "pearson_r": pearson_r,
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
    }


def calculate_prediction_performance(y_resid_true_mat, y_pred_mat):
    """
    计算连续变量预测模型的交叉验证性能。

    本函数可以直接处理 ``cpm_fixed_params_cv()`` 或 ``cpm_nested_cv()`` 返回的
    ``y_resid_true_mat`` 与 ``y_pred_mat``。二维输入的每一行代表一次 repeat，
    函数会先逐行计算性能，再汇总跨 repeat 的均值和标准差；同时还会比较跨
    repeat 平均后的真实值与预测值。参数名故意与 CV 函数的返回值名称保持
    一致，以提醒这里评估的是折特异真实残差和折外预测残差，而不是原始 y。

    计算结果分为两层：``per_repeat`` 保留每次 repeat 各自的性能，用于观察
    不同折分带来的波动；``mean_prediction`` 先对每个被试的真实残差和预测值
    跨 repeat 取平均，再计算一次整体性能。

    Parameters
    ----------
    y_resid_true_mat : array-like, shape (n_sub,) or (n_repeats, n_sub)
        折特异的真实 y 残差，通常传入 ``cpm_fixed_params_cv()`` 或
        ``cpm_nested_cv()`` 返回值中的索引 3。不能传入原始 y 或全样本一次性
        残差化后的 y。
    y_pred_mat : array-like, shape (n_sub,) or (n_repeats, n_sub)
        与 ``y_resid_true_mat`` 逐元素对应的折外预测残差，通常传入
        ``cpm_fixed_params_cv()`` 或 ``cpm_nested_cv()`` 返回值中的索引 2。

    Returns
    -------
    performance : dict
        包含三个部分：

        - ``per_repeat`` : dict
          每次 repeat 的性能数组。包含 ``pearson_r``、``r2``、``mae`` 和
          ``rmse``，每个值均为 ``np.ndarray``，形状为 ``(n_repeats,)``。
        - ``summary`` : dict
          跨 repeat 的汇总指标。包含 ``n_repeats`` : int、``n_subjects`` : int，
          以及 ``pearson_r_mean``、``pearson_r_std``、``r2_mean``、``r2_std``、
          ``mae_mean``、``mae_std``、``rmse_mean``、``rmse_std`` 和
          ``pearson_r_fisher_mean``，这些指标值均为 float 或 ``np.nan``。
        - ``mean_prediction`` : dict
          对 repeat 取平均后，真实值和预测值之间的性能。结构同
          ``_compute_regression_metrics()`` 返回的 ``metrics``，各值均为 float
          或 ``np.nan``。

    Raises
    ------
    ValueError
        输入不是一维/二维数组、形状不一致、被试数不足或包含非有限值时抛出。

    Notes
    -----
    Pearson r 衡量线性关联，不反映预测值是否存在系统偏高或偏低；R²、MAE 和
    RMSE 会反映绝对预测误差。因此建议同时报告相关指标和误差指标。

    各 repeat 使用同一批被试，只是折分不同，因此并非相互独立的重复实验。
    ``summary`` 中的均值和标准差用于描述折分敏感性，不应直接解释为独立样本
    估计得到的标准误或置信区间。
    """
    # ---------- 1. 输入格式统一 ----------
    # 函数名中的 mat 表示推荐输入二维矩阵；同时也接受一维数组，以兼容单次 CV
    # 或用户手动传入的一组预测。
    y_resid_true_mat = np.asarray(y_resid_true_mat, dtype=float)
    y_pred_mat = np.asarray(y_pred_mat, dtype=float)

    # 单次 CV 或单组预测的形状通常是 (n_sub,)；这里统一提升为
    # (1, n_sub)，使后续代码都按“行 = repeat，列 = 被试”的二维矩阵处理。
    if y_resid_true_mat.ndim == 1:
        y_resid_true_mat = y_resid_true_mat[None, :]
    if y_pred_mat.ndim == 1:
        y_pred_mat = y_pred_mat[None, :]

    # ---------- 2. 输入合法性检查 ----------
    # 两个矩阵必须逐元素配对：同一行代表同一次 repeat，同一列代表同一被试。
    if y_resid_true_mat.ndim != 2 or y_pred_mat.ndim != 2:
        raise ValueError(
            "y_resid_true_mat and y_pred_mat must be 1D arrays or 2D arrays "
            "with shape (n_repeats, n_sub)."
        )
    if y_resid_true_mat.shape != y_pred_mat.shape:
        raise ValueError("y_resid_true_mat and y_pred_mat must have the same shape.")
    if y_resid_true_mat.shape[1] < 2:
        raise ValueError("At least 2 subjects are required to evaluate performance.")
    if not np.all(np.isfinite(y_resid_true_mat)):
        raise ValueError("y_resid_true_mat contains NaN or infinite values.")
    if not np.all(np.isfinite(y_pred_mat)):
        raise ValueError("y_pred_mat contains NaN or infinite values.")

    metric_names = ("pearson_r", "r2", "mae", "rmse")

    # ---------- 3. 逐 repeat 计算性能 ----------
    # 每行都是一次完整的折外预测；逐行评估可以观察不同随机折分带来的波动。
    # 每个指标都保留一个长度为 n_repeats 的数组，逐行记录每次 repeat 的性能。
    per_repeat = {
        name: np.empty(y_resid_true_mat.shape[0], dtype=float) for name in metric_names
    }

    # zip(...) 将真实残差矩阵和预测矩阵逐行配对；enumerate(...) 同时给出
    # 当前 repeat 的位置，用于把该 repeat 的各项指标写回 per_repeat。
    for repeat_idx, (y_resid_true_row, y_pred_row) in enumerate(
        zip(y_resid_true_mat, y_pred_mat)
    ):
        metrics = _compute_regression_metrics(y_resid_true_row, y_pred_row)
        for name in metric_names:
            per_repeat[name][repeat_idx] = metrics[name]

    # ---------- 4. 汇总 repeat 层面的性能分布 ----------
    # summary 描述 per_repeat 的均值和标准差；这些 repeat 共用同一批被试，因此它
    # 不是独立样本意义上的置信区间估计。
    summary = {
        "n_repeats": y_resid_true_mat.shape[0],
        "n_subjects": y_resid_true_mat.shape[1],
    }

    # summary 汇总当前保留的效应量和误差指标。
    for name in ("pearson_r", "r2", "mae", "rmse"):
        values = per_repeat[name]
        # 某些 repeat 可能因零方差得到 NaN；汇总时只使用有限值。
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            summary[f"{name}_mean"] = np.nan
            summary[f"{name}_std"] = np.nan
        else:
            summary[f"{name}_mean"] = float(np.mean(finite_values))
            summary[f"{name}_std"] = float(
                np.std(finite_values, ddof=1) if finite_values.size > 1 else 0.0
            )

    # 相关系数先做 Fisher z 转换再平均，比直接平均 r 更符合相关系数的分布性质。
    # arctanh 在 r = +/-1 处无穷大，因此先做极小幅度裁剪。
    finite_r = per_repeat["pearson_r"][np.isfinite(per_repeat["pearson_r"])]
    if finite_r.size == 0:
        summary["pearson_r_fisher_mean"] = np.nan
    else:
        clipped_r = np.clip(finite_r, -1.0 + _EPS, 1.0 - _EPS)
        summary["pearson_r_fisher_mean"] = float(
            np.tanh(np.mean(np.arctanh(clipped_r)))
        )

    # ---------- 5. 评估跨 repeat 平均后的最终预测 ----------
    # 如果最终报告或后续分析使用平均预测值，这个指标比 per_repeat 均值更贴近
    # 实际使用的 ensemble prediction。
    # mean_prediction 描述“先平均 repeat，再评估”的性能；它和 per_repeat 的均值
    # 不是同一个量，前者更接近最终 ensemble/平均预测的表现。
    mean_prediction = _compute_regression_metrics(
        np.mean(y_resid_true_mat, axis=0),
        np.mean(y_pred_mat, axis=0),
    )

    return {
        "per_repeat": per_repeat,
        "summary": summary,
        "mean_prediction": mean_prediction,
    }


def _validate_performance_metric(performance_metric):
    """
    验证并规范化预测性能指标名称。

    Parameters
    ----------
    performance_metric : str
        需要提取或检验的性能指标名称。

    Returns
    -------
    performance_metric : str
        规范化后的小写指标名称。

    Raises
    ------
    ValueError
        指标名称不属于 ``'pearson_r'``、``'r2'``、``'mae'`` 或 ``'rmse'`` 时抛出。
    """
    performance_metric = str(performance_metric).lower()
    valid_metrics = {"pearson_r", "r2", "mae", "rmse"}
    if performance_metric not in valid_metrics:
        raise ValueError(
            "performance_metric must be 'pearson_r', 'r2', 'mae', or 'rmse'."
        )
    return performance_metric


def _extract_performance_score(
    performance,
    performance_metric="pearson_r",
    use_mean_prediction=True,
):
    """
    从 ``calculate_prediction_performance()`` 的结果中提取一个目标性能分数。

    Parameters
    ----------
    performance : dict
        ``calculate_prediction_performance()`` 返回的性能字典。
    performance_metric : {'pearson_r', 'r2', 'mae', 'rmse'}, default='pearson_r'
        需要提取的性能指标。该值应已由调用方通过
        ``_validate_performance_metric()`` 验证；例如 ``cpm_permutation_test()``
        会在运行置换前提前完成该检查，避免无效指标等到置换循环中才报错。
    use_mean_prediction : bool, default=True
        若为 True，提取 ``mean_prediction`` 中的指标；若为 False，提取
        ``summary`` 中跨 repeat 平均后的指标。

    Returns
    -------
    score : float
        目标性能分数。
    score_source : str, {'mean_prediction', 'summary_mean'}
        分数来源。

    Raises
    ------
    KeyError
        ``performance`` 缺少所需字段，或 ``performance_metric`` 未经验证且不在
        性能字典中时抛出。
    ValueError
        布尔参数无效时抛出。
    """
    use_mean_prediction = _validate_boolean(
        use_mean_prediction,
        "use_mean_prediction",
    )

    if use_mean_prediction:
        score = performance["mean_prediction"][performance_metric]
        score_source = "mean_prediction"
    else:
        score = performance["summary"][f"{performance_metric}_mean"]
        score_source = "summary_mean"

    return float(score), score_source


def cpm_permutation_test(
    X,
    y,
    Covariates,
    n_permutations=1000,
    performance_metric="pearson_r",
    use_mean_prediction=True,
    # ---------- 外层交叉验证参数 ----------
    n_splits=10,
    n_repeats=50,
    random_state=0,
    cv_type="kfold",
    # ---------- nested CV 参数 ----------
    param_grid=None,
    inner_n_splits=5,
    inner_cv_type="kfold",
    scoring="pearson_r",
    # ---------- 选边参数 ----------
    selection_method="pvalue",
    p_thresh=0.01,
    top_k=None,
    top_k_mode="total",
    max_edges_total=None,
    min_edges_total=0,
    fallback_if_too_few=False,
    max_edges_per_sign=None,
    # ---------- 模型参数 ----------
    ridge_alpha=1.0,
    lasso_alpha=1.0,
    elasticnet_alpha=1.0,
    l1_ratio=0.5,
    model_type="ridge",
    pls_components=None,
    pca_components=None,
    # ---------- 特征处理参数 ----------
    standardize_features=True,
    # ---------- 输出控制 ----------
    return_permutation_performances=False,
    verbose=False,
):
    """
    运行 CPM 的置换 workflow，并返回置换零分布。

    本函数只负责生成 null distribution：每次打乱 y 与 X/Covariates 的对应关系，
    然后用置换后的 y 重新运行完整 CPM，并提取指定性能指标。它不运行真实 y 的
    observed CPM，也不计算 p 值。真实模型性能应先运行 ``cpm_fixed_params_cv()``
    或 ``cpm_nested_cv()``，再将返回值中的 ``y_resid_true_mat`` 和 ``y_pred_mat``
    传入 ``calculate_prediction_performance(y_resid_true_mat=cv_results[3],
    y_pred_mat=cv_results[2])``。p 值可用 ``calculate_permutation_p_value()``
    计算。

    若 ``param_grid is None``，每次置换调用 ``cpm_fixed_params_cv()``；若传入
    ``param_grid``，每次置换调用 ``cpm_nested_cv()``，即在每个 outer fold 内
    重新通过 inner CV 选择模型超参数。

    Parameters
    ----------
    X : array-like, shape (n_sub, n_edges)
        被试乘连接边的特征矩阵。
    y : array-like, shape (n_sub,)
        连续预测目标。每次置换会打乱该向量的被试顺序。
    Covariates : array-like or None, shape (n_sub, n_Covariates)
        需要控制的协变量，不包含截距；传入 None 表示只控制截距。
    n_permutations : int, default=1000
        置换次数。
    performance_metric : {'pearson_r', 'r2', 'mae', 'rmse'}, default='pearson_r'
        每次置换后需要提取的预测性能指标。
    use_mean_prediction : bool, default=True
        若为 True，置换分数来自 ``mean_prediction``；若为 False，置换分数来自
        ``summary`` 中跨 repeat 平均后的指标。
    n_splits, n_repeats, random_state, cv_type
        外层交叉验证参数，含义同 ``cpm_fixed_params_cv()``。
    param_grid : dict or list of dict or None, default=None
        nested CV 的模型超参数搜索范围。传入 None 时不运行 nested CV。
    inner_n_splits, inner_cv_type, scoring
        nested CV 的内层交叉验证参数，仅在 ``param_grid`` 非 None 时使用。
    selection_method, p_thresh, top_k, top_k_mode, max_edges_total,
    min_edges_total, fallback_if_too_few, max_edges_per_sign
        CPM 选边参数，含义同 ``select_edges()``。
    ridge_alpha, lasso_alpha, elasticnet_alpha, l1_ratio, model_type,
    pls_components, pca_components
        预测模型参数，含义同 ``cpm_fixed_params_cv()``。
    standardize_features : bool, default=True
        是否在每个 fold 内使用训练折均值和标准差标准化预测特征。
    return_permutation_performances : bool, default=False
        是否返回每次置换的完整性能字典。该输出可能较大，通常只在调试时需要。
    verbose : bool, default=False
        是否打印置换进度。

    Returns
    -------
    result : dict
        置换 workflow 结果，主要键包括：

        - ``null_scores`` : np.ndarray, shape (n_permutations,)
          每次置换得到的目标性能分数，即置换零分布。
        - ``performance_metric`` : str
          提取的性能指标名称。
        - ``score_source`` : str, {'mean_prediction', 'summary_mean'}
          置换分数来自平均预测性能，还是来自跨 repeat 汇总均值。
        - ``n_permutations`` : int
          请求运行的置换次数。
        - ``n_valid_permutations`` : int
          得到有限分数的置换次数。
        - ``n_invalid_permutations`` : int
          得到 ``NaN`` 或非有限分数的置换次数。
        - ``cv_mode`` : {'fixed', 'nested'}
          每次置换使用固定参数 CV 还是 nested CV。
        - ``permutation_performances`` : list of dict, optional
          仅当 ``return_permutation_performances=True`` 时返回。每个元素都是
          ``calculate_prediction_performance()`` 返回的 ``performance`` 字典。

    Raises
    ------
    ValueError
        输入格式、置换次数、性能指标或 CPM 参数无效时抛出。

    Notes
    -----
    本函数采用 label permutation，即直接打乱 y 相对于 X 和 Covariates 的对应
    关系。在包含强协变量结构的设计中，若需要严格保持协变量与 y 的关系，可能
    需要 Freedman-Lane 等更专门的置换方案。
    """
    # ---------- 1. 输入数据与置换参数验证 ----------
    # 本函数不运行真实 y 的 observed model，只为置换后的 y 生成 null scores。
    X, y, Covariates, _, _ = _validate_X_y_Covariates(
        X,
        y,
        Covariates,
        minimum_n_sub=3,
    )
    n_permutations = _validate_integer(
        n_permutations,
        "n_permutations",
        minimum=1,
    )
    random_state = _validate_integer(random_state, "random_state")
    use_mean_prediction = _validate_boolean(
        use_mean_prediction,
        "use_mean_prediction",
    )
    return_permutation_performances = _validate_boolean(
        return_permutation_performances,
        "return_permutation_performances",
    )
    verbose = _validate_boolean(verbose, "verbose")
    performance_metric = _validate_performance_metric(performance_metric)

    # ---------- 2. 整理每次置换都会复用的 CV 参数 ----------
    # permutation 只改变 y 的被试顺序；X、Covariates、CV 设置、选边规则和模型
    # 设置保持不变。置换函数只提取预测性能；每次 CV 临时产生的 edge_records
    # 不会保存到置换结果中。
    cv_common = {
        "X": X,
        "Covariates": Covariates,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "random_state": random_state,
        "cv_type": cv_type,
        "selection_method": selection_method,
        "p_thresh": p_thresh,
        "top_k": top_k,
        "top_k_mode": top_k_mode,
        "max_edges_total": max_edges_total,
        "min_edges_total": min_edges_total,
        "fallback_if_too_few": fallback_if_too_few,
        "max_edges_per_sign": max_edges_per_sign,
        "ridge_alpha": ridge_alpha,
        "lasso_alpha": lasso_alpha,
        "elasticnet_alpha": elasticnet_alpha,
        "l1_ratio": l1_ratio,
        "model_type": model_type,
        "pls_components": pls_components,
        "pca_components": pca_components,
        "standardize_features": standardize_features,
        "verbose": False,
    }
    use_nested_cv = param_grid is not None

    # ---------- 3. 初始化置换输出容器 ----------
    # null_scores 是最终用于计算 permutation p value 的零分布。可选的
    # permutation_performances 只建议调试时使用，因为它会保存完整性能字典。
    rng = np.random.default_rng(random_state)
    null_scores = np.empty(n_permutations, dtype=float)
    permutation_performances = [] if return_permutation_performances else None
    score_source = "mean_prediction" if use_mean_prediction else "summary_mean"

    for perm_idx in range(n_permutations):
        # ---------- 4. 生成一个 label permutation ----------
        # 这里打乱 y 相对于 X/Covariates 的对应关系；X 和 Covariates 的行顺序不变。
        if verbose and (
            perm_idx == 0
            or (perm_idx + 1) == n_permutations
            or (perm_idx + 1) % max(1, n_permutations // 10) == 0
        ):
            print(f"Permutation {perm_idx + 1}/{n_permutations}")

        y_perm = rng.permutation(y)

        # ---------- 5. 对置换后的 y 运行完整 CPM workflow ----------
        # 若 param_grid=None，使用固定超参数 CV；否则每次置换都执行 nested CV，
        # 包括在每个 outer fold 内重新选择超参数。
        if use_nested_cv:
            perm_cv_results = cpm_nested_cv(
                y=y_perm,
                param_grid=param_grid,
                inner_n_splits=inner_n_splits,
                inner_cv_type=inner_cv_type,
                scoring=scoring,
                **cv_common,
            )
        else:
            perm_cv_results = cpm_fixed_params_cv(y=y_perm, **cv_common)

        # ---------- 6. 从置换 CV 输出中提取目标性能分数 ----------
        # 与 observed model 一样，性能必须比较折特异真实残差和折外预测残差。
        perm_performance = calculate_prediction_performance(
            y_resid_true_mat=perm_cv_results[3],
            y_pred_mat=perm_cv_results[2],
        )
        null_scores[perm_idx], score_source = _extract_performance_score(
            perm_performance,
            performance_metric=performance_metric,
            use_mean_prediction=use_mean_prediction,
        )

        if return_permutation_performances:
            permutation_performances.append(perm_performance)

    # ---------- 7. 汇总置换零分布 ----------
    # p 值不在这里计算；调用 calculate_permutation_p_value() 时再把 observed_score
    # 与 null_scores 放在一起比较。
    finite_null_scores = null_scores[np.isfinite(null_scores)]
    result = {
        "null_scores": null_scores,
        "performance_metric": performance_metric,
        "score_source": score_source,
        "n_permutations": n_permutations,
        "n_valid_permutations": int(finite_null_scores.size),
        "n_invalid_permutations": int(n_permutations - finite_null_scores.size),
        "cv_mode": "nested" if use_nested_cv else "fixed",
    }

    if return_permutation_performances:
        result["permutation_performances"] = permutation_performances

    return result


def calculate_permutation_p_value(
    observed_score,
    null_scores,
    performance_metric="pearson_r",
    alternative="auto",
):
    """
    根据观测分数和置换分布计算置换检验 p 值。

    Parameters
    ----------
    observed_score : float
        真实 y 对应的观测预测性能。
    null_scores : array-like, shape (n_permutations,)
        置换 workflow 生成的零分布分数。
    performance_metric : {'pearson_r', 'r2', 'mae', 'rmse'}, default='pearson_r'
        性能指标名称，仅在 ``alternative='auto'`` 时用于推断检验方向。
    alternative : {'auto', 'greater', 'less'}, default='auto'
        置换检验方向。``'greater'`` 表示越大越好，``'less'`` 表示越小越好。
        ``'auto'`` 对 ``pearson_r`` 和 ``r2`` 使用 ``'greater'``，对 ``mae`` 和
        ``rmse`` 使用 ``'less'``。

    Returns
    -------
    result : dict
        置换 p 值结果字典，主要键包括：

        - ``p_value`` : float
          加 1 校正后的置换检验 p 值。
        - ``observed_score`` : float
          真实 y 对应的观测预测性能。
        - ``null_scores`` : np.ndarray, shape (n_permutations,)
          输入的置换零分布，保留原始顺序和非有限值。
        - ``performance_metric`` : str
          用于推断方向或记录结果的性能指标名称。
        - ``alternative`` : {'greater', 'less'}
          实际使用的检验方向。
        - ``n_permutations`` : int
          输入的置换分数总数。
        - ``n_valid_permutations`` : int
          参与 p 值计算的有限置换分数数量。
        - ``n_invalid_permutations`` : int
          被排除的非有限置换分数数量。
        - ``n_extreme`` : int
          置换分布中至少与观测分数同样极端的次数。

    Raises
    ------
    ValueError
        观测分数非有限、置换分数为空/全非有限，或检验方向无效时抛出。
    """
    performance_metric = _validate_performance_metric(performance_metric)
    observed_score = _validate_finite_float(observed_score, "observed_score")
    null_scores = np.asarray(null_scores, dtype=float).ravel()

    if null_scores.size == 0:
        raise ValueError("null_scores must contain at least one value.")

    alternative = str(alternative).lower()
    if alternative == "auto":
        alternative_actual = (
            "less" if performance_metric in {"mae", "rmse"} else "greater"
        )
    elif alternative in {"greater", "less"}:
        alternative_actual = alternative
    else:
        raise ValueError("alternative must be 'auto', 'greater', or 'less'.")

    finite_null_scores = null_scores[np.isfinite(null_scores)]
    if finite_null_scores.size == 0:
        raise ValueError(
            "All permutation scores are non-finite; permutation p value "
            "cannot be computed."
        )

    # n_extreme 统计置换分布中至少与真实模型同样好的次数；方向由指标决定。
    if alternative_actual == "greater":
        n_extreme = int(np.sum(finite_null_scores >= observed_score))
    else:
        n_extreme = int(np.sum(finite_null_scores <= observed_score))

    # 加 1 校正避免 p 值为 0；分母使用有效置换次数，排除无法定义的 null score。
    p_value = (n_extreme + 1) / (finite_null_scores.size + 1)

    return {
        "p_value": float(p_value),
        "observed_score": float(observed_score),
        "null_scores": null_scores,
        "performance_metric": performance_metric,
        "alternative": alternative_actual,
        "n_permutations": int(null_scores.size),
        "n_valid_permutations": int(finite_null_scores.size),
        "n_invalid_permutations": int(null_scores.size - finite_null_scores.size),
        "n_extreme": n_extreme,
    }


# -------------------------------------------------------------------------
# 多进程并行 API
# -------------------------------------------------------------------------
# 上述串行函数保持不变。以下函数仅并行执行相互独立的外层折或置换任务，
# 不改变数据处理流程、统计方法和返回结构。


def _run_fixed_outer_fold_parallel(
    X,
    y,
    Covariates,
    train_idx,
    test_idx,
    repeat_idx,
    fold_idx,
    model_type,
    model_params,
    selection_params,
    standardize_features,
    random_state,
):
    """运行固定参数交叉验证的单个外层折，并返回该折的结果。"""
    prepared = _select_and_prepare_fold_features(
        X_train=X[train_idx],
        y_train=y[train_idx],
        Covariates_train=Covariates[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        Covariates_test=Covariates[test_idx],
        model_type=model_type,
        standardize_features=standardize_features,
        **selection_params,
    )
    fold_pred, fitted_model = _fit_and_predict_prepared_fold(
        prepared,
        model_type=model_type,
        model_params=model_params,
        random_state=random_state + repeat_idx,
        return_model=True,
    )
    edge_record = _extract_fold_edge_record(
        repeat_idx=repeat_idx,
        fold_idx=fold_idx,
        prepared_fold=prepared,
        model=fitted_model,
        model_type=model_type,
    )
    return (
        repeat_idx,
        fold_idx,
        test_idx,
        fold_pred,
        prepared["y_resid_true_test"],
        edge_record,
    )


def cpm_fixed_params_cv_parallel(
    X,
    y,
    Covariates,
    n_splits=10,
    n_repeats=50,
    random_state=0,
    cv_type="kfold",
    selection_method="pvalue",
    p_thresh=0.01,
    top_k=None,
    top_k_mode="total",
    max_edges_total=None,
    min_edges_total=0,
    fallback_if_too_few=False,
    max_edges_per_sign=None,
    ridge_alpha=1.0,
    lasso_alpha=1.0,
    elasticnet_alpha=1.0,
    l1_ratio=0.5,
    model_type="ridge",
    pls_components=None,
    pca_components=None,
    standardize_features=True,
    verbose=False,
    n_jobs=1,
):
    """
    ``cpm_fixed_params_cv()`` 的多进程版本。

    本函数使用 joblib 的 ``loky`` 后端并行计算相互独立的外层折。返回值结构及
    ``edge_records`` 的顺序与串行函数一致。``n_jobs=1`` 时直接调用串行函数。
    """
    n_jobs = _validate_integer(n_jobs, "n_jobs", minimum=1)
    serial_kwargs = {
        "X": X,
        "y": y,
        "Covariates": Covariates,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "random_state": random_state,
        "cv_type": cv_type,
        "selection_method": selection_method,
        "p_thresh": p_thresh,
        "top_k": top_k,
        "top_k_mode": top_k_mode,
        "max_edges_total": max_edges_total,
        "min_edges_total": min_edges_total,
        "fallback_if_too_few": fallback_if_too_few,
        "max_edges_per_sign": max_edges_per_sign,
        "ridge_alpha": ridge_alpha,
        "lasso_alpha": lasso_alpha,
        "elasticnet_alpha": elasticnet_alpha,
        "l1_ratio": l1_ratio,
        "model_type": model_type,
        "pls_components": pls_components,
        "pca_components": pca_components,
        "standardize_features": standardize_features,
        "verbose": verbose,
    }
    if n_jobs == 1:
        return cpm_fixed_params_cv(**serial_kwargs)

    from joblib import Parallel, delayed

    X, y, Covariates, n_sub, _ = _validate_X_y_Covariates(
        X,
        y,
        Covariates,
        minimum_n_sub=3,
    )
    model_type, model_params = _validate_model_params(
        model_type=model_type,
        ridge_alpha=ridge_alpha,
        lasso_alpha=lasso_alpha,
        elasticnet_alpha=elasticnet_alpha,
        l1_ratio=l1_ratio,
        pls_components=pls_components,
        pca_components=pca_components,
    )
    random_state = _validate_integer(random_state, "random_state")
    standardize_features = _validate_boolean(
        standardize_features,
        "standardize_features",
    )
    verbose = _validate_boolean(verbose, "verbose")
    cv_type, n_repeats_actual, n_splits_actual = _validate_cv_config(
        cv_type=cv_type,
        n_splits=n_splits,
        n_repeats=n_repeats,
        n_sub=n_sub,
    )
    if verbose and cv_type == "loocv" and n_repeats != 1:
        print("cv_type='loocv'：忽略 n_repeats，实际重复次数设为 1。")

    selection_params = {
        "selection_method": selection_method,
        "p_thresh": p_thresh,
        "top_k": top_k,
        "top_k_mode": top_k_mode,
        "max_edges_total": max_edges_total,
        "min_edges_total": min_edges_total,
        "fallback_if_too_few": fallback_if_too_few,
        "max_edges_per_sign": max_edges_per_sign,
    }
    fold_specs = []
    for rep in range(n_repeats_actual):
        cv_iter = _make_cv_iterator(
            cv_type,
            n_splits_actual,
            X,
            random_state=random_state + rep,
        )
        fold_specs.extend(
            (rep, fold_idx, train_idx, test_idx)
            for fold_idx, (train_idx, test_idx) in enumerate(cv_iter)
        )

    if verbose:
        print(f"使用 n_jobs={n_jobs} 并行计算 {len(fold_specs)} 个外层折。")
    fold_results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        max_nbytes="100M",
        mmap_mode="r",
    )(
        delayed(_run_fixed_outer_fold_parallel)(
            X=X,
            y=y,
            Covariates=Covariates,
            train_idx=train_idx,
            test_idx=test_idx,
            repeat_idx=rep,
            fold_idx=fold_idx,
            model_type=model_type,
            model_params=model_params,
            selection_params=selection_params,
            standardize_features=standardize_features,
            random_state=random_state,
        )
        for rep, fold_idx, train_idx, test_idx in fold_specs
    )

    y_pred_mat = np.zeros((n_repeats_actual, n_sub), dtype=float)
    y_resid_true_mat = np.zeros((n_repeats_actual, n_sub), dtype=float)
    edge_records = []
    for rep, _, test_idx, fold_pred, y_true, edge_record in fold_results:
        y_pred_mat[rep, test_idx] = fold_pred
        y_resid_true_mat[rep, test_idx] = y_true
        edge_records.append(edge_record)

    return (
        y_pred_mat.mean(axis=0),
        y_resid_true_mat.mean(axis=0),
        y_pred_mat,
        y_resid_true_mat,
        edge_records,
    )


def _run_nested_outer_fold_parallel(
    X,
    y,
    Covariates,
    train_idx,
    test_idx,
    repeat_idx,
    fold_idx,
    param_candidates,
    candidate_model_params,
    model_type,
    inner_cv_type,
    inner_n_splits,
    selection_params,
    standardize_features,
    scoring,
    random_state,
):
    """运行 nested CV 的单个外层折，包括内层调参和外层预测。"""
    inner_seed = random_state + repeat_idx * 1000 + fold_idx
    (
        best_model_params,
        best_score,
        candidate_scores,
        inner_n_splits_actual,
    ) = _select_best_hyperparams_inner_cv(
        X=X[train_idx],
        y=y[train_idx],
        Covariates=Covariates[train_idx],
        param_candidates=param_candidates,
        candidate_model_params=candidate_model_params,
        model_type=model_type,
        inner_cv_type=inner_cv_type,
        inner_n_splits=inner_n_splits,
        random_state=inner_seed,
        selection_params=selection_params,
        standardize_features=standardize_features,
        scoring=scoring,
    )
    prepared = _select_and_prepare_fold_features(
        X_train=X[train_idx],
        y_train=y[train_idx],
        Covariates_train=Covariates[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        Covariates_test=Covariates[test_idx],
        model_type=model_type,
        standardize_features=standardize_features,
        **selection_params,
    )
    fold_pred, fitted_model = _fit_and_predict_prepared_fold(
        prepared,
        model_type=model_type,
        model_params=best_model_params,
        random_state=random_state + repeat_idx,
        return_model=True,
    )
    edge_record = _extract_fold_edge_record(
        repeat_idx=repeat_idx,
        fold_idx=fold_idx,
        prepared_fold=prepared,
        model=fitted_model,
        model_type=model_type,
    )
    param_record = {
        "repeat": repeat_idx,
        "fold": fold_idx,
        "best_params": best_model_params.copy(),
        "inner_score": best_score,
        "scoring": scoring,
        "inner_cv_type": str(inner_cv_type).lower(),
        "inner_n_splits_actual": inner_n_splits_actual,
        "candidate_scores": candidate_scores,
    }
    return (
        repeat_idx,
        fold_idx,
        test_idx,
        fold_pred,
        prepared["y_resid_true_test"],
        edge_record,
        param_record,
    )


def cpm_nested_cv_parallel(
    X,
    y,
    Covariates,
    param_grid,
    n_splits=10,
    n_repeats=50,
    random_state=0,
    cv_type="kfold",
    inner_n_splits=5,
    inner_cv_type="kfold",
    scoring="pearson_r",
    selection_method="pvalue",
    p_thresh=0.01,
    top_k=None,
    top_k_mode="total",
    max_edges_total=None,
    min_edges_total=0,
    fallback_if_too_few=False,
    max_edges_per_sign=None,
    ridge_alpha=1.0,
    lasso_alpha=1.0,
    elasticnet_alpha=1.0,
    l1_ratio=0.5,
    model_type="ridge",
    pls_components=None,
    pca_components=None,
    standardize_features=True,
    verbose=False,
    n_jobs=1,
):
    """
    ``cpm_nested_cv()`` 的多进程版本。

    每个工作进程负责一个完整的外层折，包括该折内部的 inner CV。返回值结构及
    外层折顺序与串行函数一致。``n_jobs=1`` 时直接调用 ``cpm_nested_cv()``。
    """
    n_jobs = _validate_integer(n_jobs, "n_jobs", minimum=1)
    serial_kwargs = {
        "X": X,
        "y": y,
        "Covariates": Covariates,
        "param_grid": param_grid,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "random_state": random_state,
        "cv_type": cv_type,
        "inner_n_splits": inner_n_splits,
        "inner_cv_type": inner_cv_type,
        "scoring": scoring,
        "selection_method": selection_method,
        "p_thresh": p_thresh,
        "top_k": top_k,
        "top_k_mode": top_k_mode,
        "max_edges_total": max_edges_total,
        "min_edges_total": min_edges_total,
        "fallback_if_too_few": fallback_if_too_few,
        "max_edges_per_sign": max_edges_per_sign,
        "ridge_alpha": ridge_alpha,
        "lasso_alpha": lasso_alpha,
        "elasticnet_alpha": elasticnet_alpha,
        "l1_ratio": l1_ratio,
        "model_type": model_type,
        "pls_components": pls_components,
        "pca_components": pca_components,
        "standardize_features": standardize_features,
        "verbose": verbose,
    }
    if n_jobs == 1:
        return cpm_nested_cv(**serial_kwargs)

    from joblib import Parallel, delayed

    X, y, Covariates, n_sub, _ = _validate_X_y_Covariates(
        X,
        y,
        Covariates,
        minimum_n_sub=3,
    )
    model_type, base_model_params = _validate_model_params(
        model_type=model_type,
        ridge_alpha=ridge_alpha,
        lasso_alpha=lasso_alpha,
        elasticnet_alpha=elasticnet_alpha,
        l1_ratio=l1_ratio,
        pls_components=pls_components,
        pca_components=pca_components,
    )
    random_state = _validate_integer(random_state, "random_state")
    standardize_features = _validate_boolean(
        standardize_features,
        "standardize_features",
    )
    verbose = _validate_boolean(verbose, "verbose")
    scoring = str(scoring).lower()
    if scoring not in {"pearson_r", "r2", "neg_rmse", "neg_mae"}:
        raise ValueError("scoring must be 'pearson_r', 'r2', 'neg_rmse', or 'neg_mae'.")
    cv_type, n_repeats_actual, n_splits_actual = _validate_cv_config(
        cv_type=cv_type,
        n_splits=n_splits,
        n_repeats=n_repeats,
        n_sub=n_sub,
    )
    if verbose and cv_type == "loocv" and n_repeats != 1:
        print("cv_type='loocv'：忽略 n_repeats，实际重复次数设为 1。")

    param_candidates, candidate_model_params = _expand_param_grid(
        param_grid,
        model_type,
        base_model_params,
    )
    selection_params = {
        "selection_method": selection_method,
        "p_thresh": p_thresh,
        "top_k": top_k,
        "top_k_mode": top_k_mode,
        "max_edges_total": max_edges_total,
        "min_edges_total": min_edges_total,
        "fallback_if_too_few": fallback_if_too_few,
        "max_edges_per_sign": max_edges_per_sign,
    }
    fold_specs = []
    for rep in range(n_repeats_actual):
        cv_iter = _make_cv_iterator(
            cv_type,
            n_splits_actual,
            X,
            random_state=random_state + rep,
        )
        fold_specs.extend(
            (rep, fold_idx, train_idx, test_idx)
            for fold_idx, (train_idx, test_idx) in enumerate(cv_iter)
        )

    if verbose:
        print(
            f"使用 n_jobs={n_jobs} 并行计算 "
            f"{len(fold_specs)} 个 nested CV 外层折。"
        )
    fold_results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        max_nbytes="100M",
        mmap_mode="r",
    )(
        delayed(_run_nested_outer_fold_parallel)(
            X=X,
            y=y,
            Covariates=Covariates,
            train_idx=train_idx,
            test_idx=test_idx,
            repeat_idx=rep,
            fold_idx=fold_idx,
            param_candidates=param_candidates,
            candidate_model_params=candidate_model_params,
            model_type=model_type,
            inner_cv_type=inner_cv_type,
            inner_n_splits=inner_n_splits,
            selection_params=selection_params,
            standardize_features=standardize_features,
            scoring=scoring,
            random_state=random_state,
        )
        for rep, fold_idx, train_idx, test_idx in fold_specs
    )

    y_pred_mat = np.zeros((n_repeats_actual, n_sub), dtype=float)
    y_resid_true_mat = np.zeros((n_repeats_actual, n_sub), dtype=float)
    edge_records = []
    selected_param_records = []
    for rep, _, test_idx, fold_pred, y_true, edge_record, param_record in fold_results:
        y_pred_mat[rep, test_idx] = fold_pred
        y_resid_true_mat[rep, test_idx] = y_true
        edge_records.append(edge_record)
        selected_param_records.append(param_record)

    return (
        y_pred_mat.mean(axis=0),
        y_resid_true_mat.mean(axis=0),
        y_pred_mat,
        y_resid_true_mat,
        edge_records,
        selected_param_records,
    )


def _run_single_permutation_parallel(
    permutation_idx,
    permutation_order,
    y,
    use_nested_cv,
    param_grid,
    inner_n_splits,
    inner_cv_type,
    scoring,
    cv_common,
    performance_metric,
    use_mean_prediction,
):
    """在单个工作进程中串行完成一次 CPM 置换。"""
    y_perm = y[permutation_order]
    if use_nested_cv:
        cv_results = cpm_nested_cv(
            y=y_perm,
            param_grid=param_grid,
            inner_n_splits=inner_n_splits,
            inner_cv_type=inner_cv_type,
            scoring=scoring,
            **cv_common,
        )
    else:
        cv_results = cpm_fixed_params_cv(y=y_perm, **cv_common)

    performance = calculate_prediction_performance(
        y_resid_true_mat=cv_results[3],
        y_pred_mat=cv_results[2],
    )
    score, score_source = _extract_performance_score(
        performance,
        performance_metric=performance_metric,
        use_mean_prediction=use_mean_prediction,
    )
    return permutation_idx, score, score_source, performance


def cpm_permutation_test_parallel(
    X,
    y,
    Covariates,
    n_permutations=1000,
    performance_metric="pearson_r",
    use_mean_prediction=True,
    n_splits=10,
    n_repeats=50,
    random_state=0,
    cv_type="kfold",
    param_grid=None,
    inner_n_splits=5,
    inner_cv_type="kfold",
    scoring="pearson_r",
    selection_method="pvalue",
    p_thresh=0.01,
    top_k=None,
    top_k_mode="total",
    max_edges_total=None,
    min_edges_total=0,
    fallback_if_too_few=False,
    max_edges_per_sign=None,
    ridge_alpha=1.0,
    lasso_alpha=1.0,
    elasticnet_alpha=1.0,
    l1_ratio=0.5,
    model_type="ridge",
    pls_components=None,
    pca_components=None,
    standardize_features=True,
    return_permutation_performances=False,
    verbose=False,
    n_jobs=1,
):
    """
    ``cpm_permutation_test()`` 的多进程版本。

    本函数先在主进程中使用与串行函数相同的随机数序列生成全部置换顺序，再将
    各次置换分发给不同工作进程。每次置换内部仍串行运行 CPM，避免嵌套进程池
    造成 CPU 过度占用。
    """
    n_jobs = _validate_integer(n_jobs, "n_jobs", minimum=1)
    serial_kwargs = {
        "X": X,
        "y": y,
        "Covariates": Covariates,
        "n_permutations": n_permutations,
        "performance_metric": performance_metric,
        "use_mean_prediction": use_mean_prediction,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "random_state": random_state,
        "cv_type": cv_type,
        "param_grid": param_grid,
        "inner_n_splits": inner_n_splits,
        "inner_cv_type": inner_cv_type,
        "scoring": scoring,
        "selection_method": selection_method,
        "p_thresh": p_thresh,
        "top_k": top_k,
        "top_k_mode": top_k_mode,
        "max_edges_total": max_edges_total,
        "min_edges_total": min_edges_total,
        "fallback_if_too_few": fallback_if_too_few,
        "max_edges_per_sign": max_edges_per_sign,
        "ridge_alpha": ridge_alpha,
        "lasso_alpha": lasso_alpha,
        "elasticnet_alpha": elasticnet_alpha,
        "l1_ratio": l1_ratio,
        "model_type": model_type,
        "pls_components": pls_components,
        "pca_components": pca_components,
        "standardize_features": standardize_features,
        "return_permutation_performances": return_permutation_performances,
        "verbose": verbose,
    }
    if n_jobs == 1:
        return cpm_permutation_test(**serial_kwargs)

    from joblib import Parallel, delayed

    X, y, Covariates, n_sub, _ = _validate_X_y_Covariates(
        X,
        y,
        Covariates,
        minimum_n_sub=3,
    )
    n_permutations = _validate_integer(
        n_permutations,
        "n_permutations",
        minimum=1,
    )
    random_state = _validate_integer(random_state, "random_state")
    use_mean_prediction = _validate_boolean(
        use_mean_prediction,
        "use_mean_prediction",
    )
    return_permutation_performances = _validate_boolean(
        return_permutation_performances,
        "return_permutation_performances",
    )
    verbose = _validate_boolean(verbose, "verbose")
    performance_metric = _validate_performance_metric(performance_metric)

    cv_common = {
        "X": X,
        "Covariates": Covariates,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "random_state": random_state,
        "cv_type": cv_type,
        "selection_method": selection_method,
        "p_thresh": p_thresh,
        "top_k": top_k,
        "top_k_mode": top_k_mode,
        "max_edges_total": max_edges_total,
        "min_edges_total": min_edges_total,
        "fallback_if_too_few": fallback_if_too_few,
        "max_edges_per_sign": max_edges_per_sign,
        "ridge_alpha": ridge_alpha,
        "lasso_alpha": lasso_alpha,
        "elasticnet_alpha": elasticnet_alpha,
        "l1_ratio": l1_ratio,
        "model_type": model_type,
        "pls_components": pls_components,
        "pca_components": pca_components,
        "standardize_features": standardize_features,
        "verbose": False,
    }
    use_nested_cv = param_grid is not None
    rng = np.random.default_rng(random_state)
    permutation_orders = np.asarray(
        [rng.permutation(n_sub) for _ in range(n_permutations)],
        dtype=int,
    )

    if verbose:
        print(f"使用 n_jobs={n_jobs} 并行计算 {n_permutations} 次置换。")
    permutation_results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        max_nbytes="100M",
        mmap_mode="r",
    )(
        delayed(_run_single_permutation_parallel)(
            permutation_idx=perm_idx,
            permutation_order=permutation_orders[perm_idx],
            y=y,
            use_nested_cv=use_nested_cv,
            param_grid=param_grid,
            inner_n_splits=inner_n_splits,
            inner_cv_type=inner_cv_type,
            scoring=scoring,
            cv_common=cv_common,
            performance_metric=performance_metric,
            use_mean_prediction=use_mean_prediction,
        )
        for perm_idx in range(n_permutations)
    )

    null_scores = np.empty(n_permutations, dtype=float)
    permutation_performances = (
        [None] * n_permutations if return_permutation_performances else None
    )
    score_source = "mean_prediction" if use_mean_prediction else "summary_mean"
    for perm_idx, score, score_source, performance in permutation_results:
        null_scores[perm_idx] = score
        if return_permutation_performances:
            permutation_performances[perm_idx] = performance

    finite_null_scores = null_scores[np.isfinite(null_scores)]
    result = {
        "null_scores": null_scores,
        "performance_metric": performance_metric,
        "score_source": score_source,
        "n_permutations": n_permutations,
        "n_valid_permutations": int(finite_null_scores.size),
        "n_invalid_permutations": int(n_permutations - finite_null_scores.size),
        "cv_mode": "nested" if use_nested_cv else "fixed",
    }
    if return_permutation_performances:
        result["permutation_performances"] = permutation_performances
    return result

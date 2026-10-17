"""
机器学习相关的通用工具函数。

本模块用于放置不限定于 CPM 的机器学习流程，例如 PCA embedding、分类器
辅助函数、交叉验证小工具等。
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .helpers import _validate_array


def make_pca_embeddings(X_z, n_embedding_components=50, random_state=2026):
    """
    基于标准化后的特征矩阵构造 PCA embedding。

    Parameters
    ----------
    X_z : array-like, shape (n_subjects, n_features)
        已预处理并按特征列 z-score 的矩阵。行对应被试，列对应特征。
    n_embedding_components : int, default=50
        每个 embedding 使用的 PC 数量。
        - `PCn_scaled` 使用 PC1-PCn；
        - `PCn_removePC1_scaled` 去掉 PC1，使用 PC2-PC(n+1)。
    random_state : int, default=2026
        randomized PCA 的随机种子。

    Returns
    -------
    Z_scaled : np.ndarray, shape (n_subjects, n_embedding_components)
        使用 PC1-PCn 的 PCA embedding。例如 `n_embedding_components=50` 时，
        该矩阵包含 PC1-PC50。行对应被试，列对应 PC score。
    Z_removePC1_scaled : np.ndarray, shape (n_subjects, n_embedding_components)
        去掉 PC1 后的 PCA embedding。例如 `n_embedding_components=50` 时，
        该矩阵包含 PC2-PC51。行对应被试，列对应 PC score。

        `Z_scaled` 和 `Z_removePC1_scaled` 都会对 PC score 做 z-score。这里的
        z-score 是“按列”进行的：对某一个 PC，在所有被试之间计算均值和标准差，
        然后用 `(score - mean) / std` 标准化。因此每个 PC score 列进入后续
        模型前均值约为 0、标准差约为 1，避免 PC1 因方差较大而天然占更大权重。
    pca : sklearn.decomposition.PCA
        已经用 `X_z` 训练好的 PCA 模型对象。它不是后续模型直接使用的数据；
        后续模型使用的是上面两个 `Z_*` 矩阵。保留 `pca` 的作用是
        记录 PCA 如何从原始特征空间映射到 PC 空间，例如每个 PC 的方向、每个
        PC 的方差解释比例，以及 PCA 中心化时使用的均值。
    pca_var : pandas.DataFrame
        PCA 方差解释表，每一行对应一个 PC，包含三列：

        - `PC`：PC 编号，从 1 开始；
        - `explained_variance_ratio`：该 PC 单独解释的输入特征方差比例；
        - `cumulative_explained_variance_ratio`：从 PC1 累加到当前 PC 后，
          总共解释的方差比例。
    """
    X_z = _validate_array(X_z, name="X_z", ndim=2)
    if X_z.shape[0] < 2:
        raise ValueError("X_z must contain at least 2 subjects and 1 feature.")

    n_embedding_components = int(n_embedding_components)
    if n_embedding_components < 1:
        raise ValueError("n_embedding_components must be >= 1.")

    # 本函数会同时构造两个 embedding：
    #   1) PCn_scaled：使用 PC1-PCn；
    #   2) PCn_removePC1_scaled：去掉 PC1，使用 PC2-PC(n+1)。
    #
    # 因此，如果 n_embedding_components=50，removePC1 版本实际需要 PC2-PC51，
    # PCA 至少要先拟合 51 个成分，而不是 50 个成分。
    #
    # PCA 成分数还受数据维度限制。X_z 的 shape 是 (n_subjects, n_features)。
    # PCA 会先对每个特征中心化，中心化会让样本维度损失 1 个自由度，因此最多
    # 只有 n_subjects - 1 个非零方差方向；同时，成分数也不能超过特征数。
    n_pca_components = min(
        n_embedding_components + 1,
        X_z.shape[0] - 1,
        X_z.shape[1],
    )
    if n_pca_components < n_embedding_components + 1:
        raise ValueError(
            "Not enough PCA components for removePC1 embedding: "
            f"need={n_embedding_components + 1}, available={n_pca_components}."
        )

    pca = PCA(
        n_components=n_pca_components,
        svd_solver="randomized",
        random_state=random_state,
    )
    scores = pca.fit_transform(X_z)

    # scores 的 shape 是 (n_subjects, n_pca_components)：
    #   - scores[:, :n] 取第 0 到 n-1 列，对应 PC1-PCn；
    #   - scores[:, 1:n+1] 取第 1 到 n 列，对应 PC2-PC(n+1)。
    #
    # StandardScaler().fit_transform(...) 会对每个 PC score 列分别做 z-score。
    # 也就是说，它不是对每个被试的一整行标准化，而是对 PC1、PC2、...
    # 分别在所有被试之间计算 mean/std 后标准化。
    Z_scaled = StandardScaler().fit_transform(scores[:, :n_embedding_components])
    Z_removePC1_scaled = StandardScaler().fit_transform(
        scores[:, 1 : n_embedding_components + 1]
    )

    pca_var = pd.DataFrame({
        "PC": np.arange(1, scores.shape[1] + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(
            pca.explained_variance_ratio_
        ),
    })

    return Z_scaled, Z_removePC1_scaled, pca, pca_var


def evaluate_binary_cluster_alignment(
    labels,
    y_binary,
    clustering_scores=None,
):
    """
    评估二元聚类结果与真实二分类标签的一致性。

    本函数用于二元聚类结果，例如 GMM 二分类输出。函数要求：

    - ``labels`` 只包含 0 和 1，分别表示 cluster 0 和 cluster 1；
    - ``y_binary`` 只包含 0 和 1，其中 1 是需要对齐的 positive 组；
    - 若提供 ``clustering_scores``，其 shape 必须为 `(n_subjects, 2)`，
    第 k 列对应 cluster k。（k=0或1）。

    无监督聚类的 cluster 编号没有天然方向：cluster 0 不一定是 negative-like，
    cluster 1 也不一定是 positive-like。因此，本函数先比较两个 cluster 内
    ``y_binary == 1`` 的比例；阳性比例更高的 cluster 被解释为 positive-like，
    另一个 cluster 被解释为 negative-like。随后再计算映射后的 alignment 指标。

    函数内部流程分为三层：

    1. 输入检查
       检查 ``labels`` 和 ``y_binary`` （1）是否逐被试对齐？（2）是否都只包含
       0/1？（3）是否都实际包含了两个类别？ 若提供
       ``clustering_scores``，还要求其 shape 为 ``(n_subjects, 2)``。

    2. Cluster 方向判定
       由于无监督 cluster 编号没有天然方向，函数用每个 cluster 内
       ``y_binary == 1`` 的比例决定哪个 cluster 是 positive-like。

    3. 指标计算
       计算映射后的 accuracy / balanced accuracy、列联表统计量、ARI / NMI。
       若提供 ``clustering_scores``，函数会取出 positive-like cluster 对应的
       分数列，并计算 ROC-AUC。

    Parameters
    ----------
    labels : array-like, shape (n_subjects,)
        无监督模型输出的二元 cluster label，必须只包含 0 和 1。这里的 0/1
        只是 cluster 编号，不代表 negative/positive 方向。
    y_binary : array-like, shape (n_subjects,)
        真实二分类标签，必须只包含 0 和 1，且两个类别都要出现。1 表示本函数
        要对齐的 positive 组。
    clustering_scores : array-like or None, default=None
        聚类算法输出的连续分数矩阵，行对应被试，列对应 cluster。例如 GMM 中
        可以传入 ``gmm.predict_proba(Z)``。函数会先判断哪个 cluster 是
        positive-like，再自动取出对应列作为 ``positive_score`` 并计算 ROC-AUC。

        为了保持函数逻辑清楚，本函数要求 ``clustering_scores`` 是二维数值矩阵，
        shape 为 ``(n_subjects, 2)``，且第 0 列对应 cluster 0，第 1 列对应
        cluster 1。
    Returns
    -------
    metrics : dict
        核心统计量字典。

        主要字段包括：

        - ``cluster_positive_like``：int，``y_binary == 1`` 比例更高的 cluster；
        - ``pos_rate_in_pos_like``：float，等于
          ``positive_rate[positive_like_cluster]``，即 positive-like cluster 内
          真实 positive 被试的比例；
        - ``pos_rate_in_neg_like``：float，等于
          ``positive_rate[negative_like_cluster]``，即 negative-like cluster 内
          真实 positive 被试的比例；
        - ``accuracy``：float，方向映射后的普通准确率，即
          ``pred_positive`` 与 ``y_binary`` 完全一致的被试比例；
        - ``balanced_accuracy``：float，方向映射后的平衡准确率，即分别
          计算 positive 组和 negative 组的正确率（sensitivity & specificity）后取平均；当两组样本量不平衡时，
          它比普通 accuracy 更不容易被大组主导；
        - ``positive_score_auc``：float，使用内部派生的 ``positive_score`` 计算
          的 ROC-AUC；
        - ``cramers_v``、``chi2_p``、``fisher_p``：float，列联表关联统计量；
        - ``ARI``、``NMI``：float，cluster label 与真实标签的无监督一致性指标。
    pred_positive : np.ndarray, shape (n_subjects,)
        将 positive-like cluster 映射为 1、negative-like cluster 映射为 0 后
        得到的二值标签。
    positive_score : np.ndarray or None, shape (n_subjects,)
        从 ``clustering_scores`` 中取出的 positive-like cluster 分数。若未提供
        ``clustering_scores``，则返回 ``None``。
    """
    # 1. 输入检查。
    # 本函数把聚类结果限定为二元数值 label：
    #   - labels 只能包含 0/1，表示 cluster 0 / cluster 1；
    #   - y_binary 只能包含 0/1，表示真实 negative / positive 标签；
    #   - labels 和 y_binary 都必须实际包含两个类别；
    #   - 二者必须逐被试对齐。
    #
    # 这样做的原因是 clustering_scores 使用列索引表达 cluster：第 0 列对应
    # cluster 0，第 1 列对应 cluster 1。若允许字符串 label 或任意编号，
    # positive_score 的列选择会变得不透明。
    labels = np.asarray(labels)
    y_binary = np.asarray(y_binary)
    if labels.shape[0] != y_binary.shape[0]:
        raise ValueError("labels and y_binary must have the same length.")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("labels must contain only 0 and 1.")
    if not np.all(np.isin(y_binary, [0, 1])):
        raise ValueError("y_binary must contain only 0 and 1.")
    labels = labels.astype(int)
    y_binary = y_binary.astype(int)

    unique_labels = np.unique(labels)
    if unique_labels.shape[0] != 2:
        raise ValueError("labels must contain both cluster 0 and cluster 1.")
    if np.unique(y_binary).shape[0] != 2:
        raise ValueError("y_binary must contain both 0 and 1.")

    # 2. Cluster 方向判定。
    # 无监督模型只给出 cluster 编号，不给出“哪个 cluster 对应 positive 组”的
    # 解释。因此这里分别统计 cluster 0 和 cluster 1 内 y_binary==1 的比例；
    # 阳性比例更高的 cluster 被解释为 positive-like。这个方向解释来自
    # y_binary，而不是 cluster 编号本身。
    positive_rate = {}

    for cluster in unique_labels:
        idx = labels == cluster
        positive_rate[cluster] = float(np.mean(y_binary[idx] == 1))

    # max(dict, key=dict.get) 的意思是：比较字典的 value，并返回 value 最大的 key。
    # 这里返回的是 y_binary==1 比例最高的 cluster 编号。
    positive_like_cluster = max(positive_rate, key=positive_rate.get)
    negative_like_cluster = (
        unique_labels[0]
        if unique_labels[1] == positive_like_cluster
        else unique_labels[1]
    )
    pos_rate_in_pos_like = float(positive_rate[positive_like_cluster])
    pos_rate_in_neg_like = float(positive_rate[negative_like_cluster])

    pred_positive = (labels == positive_like_cluster).astype(int)

    # 3. 指标计算。
    # clustering_scores 若存在，应是聚类算法的原始连续分数矩阵。例如 GMM 的
    # predict_proba(Z) 会给出 shape=(n_subjects, 2) 的 probability：
    # 第 0 列对应 cluster 0，第 1 列对应 cluster 1。
    #
    # 在完成方向判定后，positive-like cluster 对应的那一列才被解释为
    # positive_score。AUC 使用这个连续分数，不依赖固定分类阈值。
    positive_score = None
    if clustering_scores is not None:
        clustering_scores = _validate_array(
            clustering_scores,
            name="clustering_scores",
            ndim=2,
        )
        if clustering_scores.shape[0] != y_binary.shape[0]:
            raise ValueError(
                "clustering_scores must have the same number of rows as y_binary."
            )
        if clustering_scores.shape[1] != 2:
            raise ValueError("clustering_scores must have exactly 2 columns.")

        # 这里采用 GMM/sklearn 的常见约定：
        # clustering_scores[:, 0] 是属于 cluster 0 的 score；
        # clustering_scores[:, 1] 是属于 cluster 1 的 score。
        # 因此确定 positive-like cluster 后，直接取对应列即可。
        positive_col = int(positive_like_cluster)
        positive_score = clustering_scores[:, positive_col]

        auc = float(roc_auc_score(y_binary, positive_score))
    else:
        auc = np.nan

    # 列联表是后续 Cramer's V、chi-square test 和 Fisher exact test 的共同基础。
    # 行保留原始 cluster 编号；列使用 negative/positive 名称，便于直接阅读。
    crosstab_df = pd.crosstab(
        pd.Series(labels, name="cluster"),
        pd.Series(
            np.where(y_binary == 1, "positive", "negative"),
            name="target_group",
        ),
    )
    for col in ["negative", "positive"]:
        if col not in crosstab_df.columns:
            crosstab_df[col] = 0
    crosstab_df = crosstab_df[["negative", "positive"]].sort_index()

    # Cramer's V 基于同一个列联表计算，用来描述 cluster 与真实二分类标签的
    # 关联强度；chi2_p 是对应的 Pearson chi-square test p 值。
    crosstab_arr = crosstab_df.to_numpy(dtype=float)
    chi2, chi2_p, _, _ = stats.chi2_contingency(crosstab_arr)
    n_total = crosstab_arr.sum()
    n_rows, n_cols = crosstab_arr.shape
    denom = n_total * min(n_rows - 1, n_cols - 1)
    cramers_v = np.sqrt(chi2 / denom) if n_total > 0 and denom > 0 else np.nan

    fisher_p = (
        stats.fisher_exact(crosstab_df.values)[1]
        if crosstab_df.shape == (2, 2)
        else np.nan
    )

    metrics = {
        "cluster_positive_like": int(positive_like_cluster),
        "pos_rate_in_pos_like": pos_rate_in_pos_like,
        "pos_rate_in_neg_like": pos_rate_in_neg_like,
        # accuracy 和 balanced_accuracy 都是在完成 cluster 方向映射后计算的：
        # positive-like cluster 视为 1，negative-like cluster 视为 0。
        # accuracy 是总体预测正确比例；balanced accuracy 先分别计算两组正确率，
        # 再取平均，因此更适合 positive/negative 样本量不均衡的情况。
        "accuracy": float(accuracy_score(y_binary, pred_positive)),
        "balanced_accuracy": float(balanced_accuracy_score(y_binary, pred_positive)),
        "positive_score_auc": auc,
        "cramers_v": float(cramers_v),
        "chi2_p": float(chi2_p),
        "fisher_p": float(fisher_p) if np.isfinite(fisher_p) else np.nan,
        "ARI": float(adjusted_rand_score(y_binary, labels)),
        "NMI": float(normalized_mutual_info_score(y_binary, labels)),
    }

    return metrics, pred_positive, positive_score


def test_binary_cluster_alignment(
    labels,
    y_binary,
    clustering_scores=None,
    n_perm=5000,
    random_state=2026,
    n_jobs=1,
):
    """
    计算二元 cluster-label alignment，并用 label permutation 检验显著性。

    本函数是 ``evaluate_binary_cluster_alignment()`` 的 permutation 版本。
    它先在真实数据下计算 observed alignment 指标；随后固定聚类结果
    ``labels``，随机打乱真实标签 ``y_binary``，生成这些指标的 null
    distribution。

    因此，本函数回答的是：
    在 cluster 结构保持不变的前提下，如果真实二分类标签与 cluster 没有关系，
    当前 observed alignment 指标是否仍然可能由随机标签排列产生？

    Parameters
    ----------
    labels : array-like, shape (n_subjects,)
        无监督模型得到的二元 cluster label，必须只包含 0 和 1。permutation
        中固定不变。
    y_binary : array-like, shape (n_subjects,)
        真实二分类标签，必须只包含 0 和 1。permutation 中只打乱它。
    clustering_scores : array-like or None, default=None
        每个被试属于各个 cluster 的连续分数，例如
        ``gmm.predict_proba(Z)``。本函数会先判断真实数据下哪个 cluster 是
        positive-like，然后自动取出对应列作为 ``positive_score``。

        与 ``evaluate_binary_cluster_alignment()`` 一致，本参数可以为 ``None``。
        若为 ``None``，函数只对 label-based 指标做 permutation，不计算
        ``positive_score_auc`` 及其 permutation p 值。若提供该参数，则要求
        ``clustering_scores`` 是二维数值矩阵，shape 为 ``(n_subjects, 2)``，
        且第 0 列对应 cluster 0，第 1 列对应 cluster 1。
    n_perm : int, default=5000
        permutation 次数。
    random_state : int, default=2026
        随机种子起点。
    n_jobs : int, default=1
        joblib 并行数。设为 1 时串行运行；设为大于 1 时并行运行。
    Returns
    -------
    metrics : dict
        真实数据下的 alignment 指标，并附加每个被置换指标的 permutation p 值、
        null distribution 均值和标准差。大多数 alignment 指标越高表示对应关系
        越清楚，因此采用右尾检验。``pos_rate_in_neg_like`` 越低表示
        negative-like cluster 中混入的 positive 被试越少，因此采用左尾检验。
    pred_positive : np.ndarray, shape (n_subjects,)
        真实数据下将 positive-like cluster 映射为 1 后得到的二值标签。
    positive_score : np.ndarray or None, shape (n_subjects,)
        与真实 positive-like cluster 对齐后的连续分数，即从
        ``clustering_scores`` 中取出的对应列。若未提供 ``clustering_scores``，
        则返回 ``None``。
    perm_df : pandas.DataFrame, shape (n_perm, n_metrics)
        每次 permutation 得到的 null alignment 指标表。列名与被检验的
        observed 指标一致。

    """
    # evaluate_binary_cluster_alignment() 是本函数的 observed 计算入口：
    # 它负责检查 labels、y_binary 和 clustering_scores 是否符合二元聚类语义，
    # 并返回真实数据下的 alignment 指标。这里不重复写同一套输入检查逻辑。
    (
        metrics,
        pred_positive,
        positive_score,
    ) = evaluate_binary_cluster_alignment(
        labels,
        y_binary,
        clustering_scores=clustering_scores,
    )

    metric_names = [
        "balanced_accuracy",
        "cramers_v",
        "ARI",
        "NMI",
        "pos_rate_in_pos_like",
        "pos_rate_in_neg_like",
    ]
    if positive_score is not None:
        metric_names.insert(1, "positive_score_auc")

    higher_is_more_aligned = {name: True for name in metric_names}
    higher_is_more_aligned["pos_rate_in_neg_like"] = False

    observed_positive_like_cluster = metrics["cluster_positive_like"]
    seeds = [random_state + i for i in range(int(n_perm))]

    def permutation_once(seed):
        """完成一次 permutation：打乱 y_binary，重新计算 alignment 指标。"""
        rng = np.random.default_rng(seed)
        y_perm = rng.permutation(y_binary)

        perm_metrics, _, _ = evaluate_binary_cluster_alignment(labels, y_perm)

        if positive_score is not None:
            # positive_score 是 observed 数据下“属于 positive-like cluster”的分数。
            # permutation 后，y_perm 会改变每个 cluster 内的 positive 比例，因此
            # permutation 数据下的 positive-like cluster 可能与 observed 数据相反。
            #
            # 若方向不变，继续使用 positive_score；若方向反转，则使用 1-score。
            # 这样每次 permutation 的 AUC 都始终表示：
            # “当前 permutation 中 positive-like cluster 的连续分数”与 y_perm 的
            # 对应程度。
            if perm_metrics["cluster_positive_like"] == observed_positive_like_cluster:
                perm_score = positive_score
            else:
                perm_score = 1.0 - positive_score
            perm_metrics["positive_score_auc"] = float(roc_auc_score(y_perm, perm_score))

        return {name: perm_metrics[name] for name in metric_names}

    rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(permutation_once)(seed) for seed in seeds
    )
    perm_df = pd.DataFrame(rows)

    for name in metric_names:
        perm_values = perm_df[name].to_numpy(dtype=float)
        observed_value = float(metrics[name])
        if higher_is_more_aligned[name]:
            n_extreme = np.sum(perm_values >= observed_value)
        else:
            n_extreme = np.sum(perm_values <= observed_value)
        metrics[f"perm_p_{name}"] = float(
            (n_extreme + 1) / (len(perm_values) + 1)
        )
        metrics[f"perm_mean_{name}"] = float(np.mean(perm_values))
        metrics[f"perm_sd_{name}"] = float(np.std(perm_values, ddof=1))

    return metrics, pred_positive, positive_score, perm_df

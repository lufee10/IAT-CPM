import numpy as np
import pandas as pd


def fc_matrix_to_edges(fc_matrix, k=1):
    """
    从 FC matrix 中提取上三角 edge vector。

    该函数是 ``fc_edges_to_matrix()`` 的反向操作。默认 ``k=1``，即只提取
    主对角线以上的元素；这也是 CPM 中最常用的 FC edge 顺序。

    Parameters
    ----------
    fc_matrix : 2d-array, shape (n_rois, n_rois)
        ROI x ROI 的 FC matrix。
    k : int, default=1
        与 ``numpy.triu()`` 中的 k 含义一致。默认 ``k=1`` 表示排除主对角线；
        ``k=0`` 表示包含主对角线。

    Returns
    -------
    fc_edges : 1d-array
        FC matrix 的上三角元素。
    """
    fc_matrix = np.asarray(fc_matrix)
    if fc_matrix.ndim != 2 or fc_matrix.shape[0] != fc_matrix.shape[1]:
        raise ValueError("fc_matrix must be a square 2D array.")

    row_idx, col_idx = np.triu_indices(fc_matrix.shape[0], k=k)
    return fc_matrix[row_idx, col_idx]


def fc_edges_to_matrix(fc_edges, diagonal_value=1, verbose=True):
    """
    将 FC edge vector 还原为对称的 FC matrix。

    ``fc_edges`` 是 FC matrix 上三角的展开结果，通常不包含主对角线。
    它的顺序必须与 ``fc_matrix_to_edges(fc_matrix, k=1)`` 一致：按行读取
    上三角，先取第 0 行右侧元素，再取第 1 行右侧元素，依次类推。

    本函数常用于把 CPM/Haufe/edge-level 统计量从一维 edge 向量还原为
    ROI x ROI 矩阵，方便后续绘图或按 ROI/network 汇总。

    Parameters
    ----------
    fc_edges : 1d-array, shape (n_rois * (n_rois - 1) / 2,)
        FC 矩阵上三角 edge vector。
    diagonal_value : float, default=1
        输出 FC matrix 的对角线数值。相关矩阵通常可设为 1；若表示 edge-level
        统计量、模型权重或 Haufe 值，通常可设为 0。它只是“主对角线填充值”，
        不等于 ``np.triu_indices()`` 中控制上三角位置的 ``k``。
    verbose : bool, default=True
        是否打印运行提示。

    Returns
    -------
    fc_matrix : 2d-array, shape (n_rois, n_rois)
        对称 FC matrix。
    """
    fc_edges = np.asarray(fc_edges, dtype=float).ravel()
    n_edges = fc_edges.size
    n_rois = int((1 + np.sqrt(1 + 8 * n_edges)) / 2)
    if n_rois * (n_rois - 1) // 2 != n_edges:
        raise ValueError('Wrong! Check the length of "fc_edges"!')
    if verbose:
        print('Nice. The "fc_edges_to_matrix()" is operated normally.')

    fc_matrix = np.full((n_rois, n_rois), diagonal_value, dtype=float)
    row_idx, col_idx = np.triu_indices(n_rois, k=1)
    fc_matrix[row_idx, col_idx] = fc_edges
    fc_matrix[col_idx, row_idx] = fc_edges

    return fc_matrix


def edgeList(fcMatrix, k=1):
    """
    旧接口：从 FC matrix 中提取上三角 edge vector。

    该函数保留给旧脚本使用。新代码建议直接调用
    ``fc_matrix_to_edges(fc_matrix, k=k)``。默认 ``k=1``，与旧版一致，
    即排除主对角线，并按上三角行优先顺序展开 FC。

    Parameters
    ----------
    fcMatrix : 2d-array, shape (n_rois, n_rois)
        ROI x ROI 的 FC matrix。
    k : int, default=1
        与旧版 ``edgeList(fcMatrix, k=1)`` 一致，含义同 ``numpy.triu()``。

    Returns
    -------
    fc_edges : 1d-array
        FC matrix 上三角 edge vector。
    """
    return fc_matrix_to_edges(fcMatrix, k=k)


def assembleMatrix(*args, diagonal=1, verbose=True):
    """
    旧接口：将 FC edge vector 还原为对称 FC matrix。

    该函数保留给旧脚本使用。新代码建议直接调用
    ``fc_edges_to_matrix(fc_edges, diagonal_value=diagonal, verbose=verbose)``。

    注意：历史代码里存在两个旧版接口，本函数同时支持：

    - ``assembleMatrix(edgeList, diagonal=1, verbose=True)``：
      自动根据 edge 数量推断 ROI 数；
    - ``assembleMatrix(n, edgeList, verbose=True)``：
      显式给出 ROI 数，主对角线固定为 1。

    Parameters
    ----------
    *args
        支持两种形式：

        - ``(edgeList,)``；
        - ``(n, edgeList)``。
    diagonal : float, default=1
        仅用于 ``assembleMatrix(edgeList, diagonal=...)`` 形式，表示输出矩阵
        的主对角线填充值。
    verbose : bool, default=True
        是否打印运行提示。

    Returns
    -------
    fc_matrix : 2d-array, shape (n_rois, n_rois)
        对称 FC matrix。
    """
    if len(args) == 1:
        edgeList = args[0]
        return fc_edges_to_matrix(
            edgeList,
            diagonal_value=diagonal,
            verbose=verbose,
        )

    if len(args) != 2:
        raise TypeError(
            "assembleMatrix() supports either assembleMatrix(edgeList, "
            "diagonal=1, verbose=True) or assembleMatrix(n, edgeList, verbose=True)."
        )

    n, edgeList = args
    edgeList = np.asarray(edgeList, dtype=float).ravel()
    expected_edges = int(n) * (int(n) - 1) // 2
    if edgeList.size != expected_edges:
        raise ValueError(
            f'Wrong! For n={n}, "edgeList" should contain '
            f"{expected_edges} edges, got {edgeList.size}."
        )

    return fc_edges_to_matrix(edgeList, diagonal_value=1, verbose=verbose)


def modulesLinks(fcMatrix, kci):
    """
    旧接口：按 community pair 汇总 FC matrix 的连接总和。

    该函数保留给旧脚本使用。旧版返回 ``labelList, modulesLinks`` 两个
    numpy array；新代码建议使用 ``fc_matrix_module_pair_sum()``，返回更直观的
    pandas.DataFrame。

    注意：为了最大限度兼容旧结果，本函数保留旧版的“有方向标签”行为：
    ``kci2D = kci_i * 100 + kci_j``。因此 A-B 和 B-A 会作为不同标签分别输出。
    如果你的新分析需要无向 network pair，请使用 ``fc_matrix_module_pair_sum()``。

    Parameters
    ----------
    fcMatrix : 2d-array, shape (n_rois, n_rois)
        ROI x ROI 的 FC matrix。
    kci : 1d-array, shape (n_rois,)
        每个 ROI 所属 community/module 的编号。

    Returns
    -------
    labelList : 1d-array
        community pair 标签，例如 102 表示 community 1 到 community 2。
    modulesLinks : 1d-array
        每个 community pair 内 FC 值的总和。
    """
    fcMatrix = np.asarray(fcMatrix, dtype=float)
    kci = np.asarray(kci).ravel()
    if fcMatrix.ndim != 2 or fcMatrix.shape[0] != fcMatrix.shape[1]:
        raise ValueError("fcMatrix must be a square 2D array.")
    if kci.shape[0] != fcMatrix.shape[0]:
        raise ValueError("kci must have one community label per ROI.")

    nROIs = kci.shape[0]
    kci2D = (kci * 100)[:, np.newaxis] + kci
    for i in range(nROIs):
        kci2D[i, i] = 0

    labelList = []
    modules_links = []
    for v in np.sort(np.unique(kci2D)):
        if v != 0:
            index = kci2D == v
            links = np.sum(fcMatrix[index])
            labelList.append(v)
            modules_links.append(links)

    return np.array(labelList), np.array(modules_links)


def fc_matrix_module_pair_sum(fc_matrix, roi_label):
    """
    按 module/community pair 汇总 FC matrix 的连接总和。

    ``roi_label`` 给出每个 ROI 所属的 module/community 名称。每个矩阵元素
    ``fc_matrix[i, j]`` 会根据第 i、j 个 ROI 的标签，被归入对应的 module pair。
    函数返回每个 module pair 内 FC 值的总和，而不是平均值。

    注意：本函数按无向 FC 网络汇总。也就是说：

    - within-module：同一 module 内的每条 ROI-ROI FC 只计一次，主对角线上的
      自连接 ``fc_matrix[i, i]`` 不参与汇总；
    - between-module：A-B 和 B-A 被视为同一个 module pair，只计算一次，并把
      同一个结果同时填入输出矩阵的 ``[A, B]`` 和 ``[B, A]``；
    - 因此输出矩阵是对称的。该函数假设输入的 FC matrix 表示无向网络，通常
      应为对称矩阵。

    Parameters
    ----------
    fc_matrix : 2d-array, shape (n_rois, n_rois)
        ROI x ROI 的 FC matrix，可表示二值连接或加权连接。
    roi_label : 1d-array, shape (n_rois,)
        每个 ROI 所属的 module/community 名称，例如 ``"Vis"``、``"Default"``
        或 ``"HO14"``。标签顺序必须与 ``fc_matrix`` 的 ROI 顺序一致。

    Returns
    -------
    module_pair_sum : pandas.DataFrame, shape (n_networks, n_networks), dtype float
        行列均为 module/community 名称。每个单元格是对应 module pair 内
        FC matrix 数值的总和。行列顺序按 ``roi_label`` 中第一次出现的顺序排列。
    """
    fc_matrix = np.asarray(fc_matrix, dtype=float)
    roi_label = np.asarray(roi_label, dtype=object).ravel()
    if fc_matrix.ndim != 2 or fc_matrix.shape[0] != fc_matrix.shape[1]:
        raise ValueError("fc_matrix must be a square 2D array.")
    if roi_label.shape[0] != fc_matrix.shape[0]:
        raise ValueError("roi_label must have one label per ROI.")
    if pd.isna(roi_label).any():
        raise ValueError("roi_label cannot contain missing values.")

    labels = list(pd.unique(roi_label))
    out = pd.DataFrame(0.0, index=labels, columns=labels, dtype=float)
    upper_triangle = np.triu(np.ones_like(fc_matrix, dtype=bool), k=1)

    for i, label_i in enumerate(labels):
        row_mask = roi_label == label_i
        for label_j in labels[i:]:
            col_mask = roi_label == label_j

            if label_i == label_j:
                # within-module：只取完整矩阵上三角，避免同一条无向 FC 被
                # (roi_a, roi_b) 和 (roi_b, roi_a) 重复计算。
                block_mask = (
                    row_mask[:, np.newaxis]
                    & col_mask[np.newaxis, :]
                    & upper_triangle
                )
            else:
                # between-module：只取 label_i 行、label_j 列这一侧。对于无向
                # FC，另一侧 label_j 行、label_i 列包含的是同一批连接，不再重复
                # 求和；最后把同一个 sum 对称填入输出矩阵。
                block_mask = row_mask[:, np.newaxis] & col_mask[np.newaxis, :]

            block_sum = float(np.sum(fc_matrix[block_mask]))
            out.loc[label_i, label_j] = block_sum
            out.loc[label_j, label_i] = block_sum

    return out


def fc_edges_module_pair_sum(fc_edges, edge_label, roi_label):
    """
    按 module/community pair 汇总 FC edge vector 的连接总和。

    这是 edge-level 版本的 ``fc_matrix_module_pair_sum()``。它不需要先把
    ``fc_edges`` 还原成完整 FC matrix，而是先用 ``edge_label`` 找到每条 FC
    两端 ROI 的 index，再用 ``roi_label`` 把 ROI index 映射为 module/community
    名称，最后按 module pair 汇总。

    本函数同样按无向 FC 网络处理：

    - within-module：同一 module 内的每条 FC 在 ``fc_edges`` 中只出现一次，
      因此只计一次；
    - between-module：A-B 和 B-A 被视为同一个 module pair，计算一次后同时填入
      输出矩阵的 ``[A, B]`` 和 ``[B, A]``；
    - 输出矩阵是对称的。

    Parameters
    ----------
    fc_edges : 1d-array, shape (n_edges,)
        每条 FC 的数值。顺序必须与 ``edge_label`` 的行顺序一致。
    edge_label : pandas.DataFrame, shape (n_edges, 2)
        每条 FC 两端 ROI 的 index。推荐包含：

        - ``roi_i``：第一个 ROI 的 index；
        - ``roi_j``：第二个 ROI 的 index。

        这里的 index 必须能在 ``roi_label`` 的 index 中找到。若没有
        ``roi_i`` / ``roi_j`` 两列，则使用 DataFrame 的前两列。
    roi_label : pandas.Series or 1d-array, shape (n_rois,)
        每个 ROI 所属的 module/community 名称。若传入 ``pandas.Series``，
        其 index 就是合法的 ROI index；若传入普通 array-like，则使用位置
        ``0, 1, ..., n_rois - 1`` 作为 ROI index。
    Returns
    -------
    module_pair_sum : pandas.DataFrame, shape (n_modules, n_modules), dtype float
        行列均为 module/community 名称。每个单元格是对应无向 module pair 内
        FC edge 数值的总和。
    """
    fc_edges = np.asarray(fc_edges, dtype=float).ravel()
    if not isinstance(edge_label, pd.DataFrame):
        raise TypeError("edge_label must be a pandas.DataFrame.")
    if edge_label.shape[1] < 2:
        raise ValueError("edge_label must contain at least two columns.")

    if {"roi_i", "roi_j"}.issubset(edge_label.columns):
        roi_i_col, roi_j_col = "roi_i", "roi_j"
    else:
        roi_i_col, roi_j_col = edge_label.columns[:2]

    edge_roi_i = edge_label[roi_i_col].to_numpy()
    edge_roi_j = edge_label[roi_j_col].to_numpy()

    if edge_roi_i.shape[0] != fc_edges.shape[0]:
        raise ValueError("edge_label must have one row per edge.")
    if edge_roi_j.shape[0] != fc_edges.shape[0]:
        raise ValueError("edge_label must have one row per edge.")
    if pd.isna(edge_roi_i).any() or pd.isna(edge_roi_j).any():
        raise ValueError("edge_label cannot contain missing ROI indices.")
    if np.any(edge_roi_i == edge_roi_j):
        raise ValueError("edge_label cannot contain self-connections.")

    # edge_label 表示无向 FC，因此 (roi_a, roi_b) 和 (roi_b, roi_a) 是同一条
    # FC，不应该在 edge_label 中出现两次。
    #
    # 为了检查这种重复，先把每条 FC 的两个 ROI index 排成固定顺序：
    # - np.minimum(edge_roi_i, edge_roi_j) 得到较小的 ROI index；
    # - np.maximum(edge_roi_i, edge_roi_j) 得到较大的 ROI index。
    #
    # 例如原始两行 (0, 1) 和 (1, 0) 都会被标准化为 (0, 1)。
    # 这样一来，只要标准化后的 (roi_a, roi_b) 出现重复行，就说明同一条
    # 无向 FC 被重复提供了。若不报错，它会在 within/between-module sum 中
    # 被重复累加。
    edge_pairs = pd.DataFrame(
        {
            "roi_a": np.minimum(edge_roi_i, edge_roi_j),
            "roi_b": np.maximum(edge_roi_i, edge_roi_j),
        }
    )
    if edge_pairs.duplicated().any():
        raise ValueError("edge_label contains duplicated undirected ROI pairs.")

    if isinstance(roi_label, pd.Series):
        roi_label_series = roi_label.astype(object)
    else:
        roi_label_series = pd.Series(np.asarray(roi_label, dtype=object).ravel())
    if pd.isna(roi_label_series).any():
        raise ValueError("roi_label cannot contain missing values.")

    roi_index = pd.Index(np.concatenate([edge_roi_i, edge_roi_j]))
    missing_roi = roi_index.difference(roi_label_series.index)
    if len(missing_roi) > 0:
        raise ValueError(
            "edge_label contains ROI indices not found in roi_label.index: "
            + ", ".join(map(str, missing_roi[:10]))
        )

    edge_label_i = roi_label_series.loc[edge_roi_i].to_numpy(dtype=object)
    edge_label_j = roi_label_series.loc[edge_roi_j].to_numpy(dtype=object)
    labels = list(pd.unique(roi_label_series))

    out = pd.DataFrame(0.0, index=labels, columns=labels, dtype=float)

    for label_i, label_j, value in zip(edge_label_i, edge_label_j, fc_edges):
        out.loc[label_i, label_j] += value
        if label_i != label_j:
            out.loc[label_j, label_i] += value

    return out

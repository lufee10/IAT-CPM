"""
biox 底层 helper 函数。

本模块用于集中保存可以被多个分析脚本复用的基础函数。这里的函数只处理
通用的数据格式检查、被试轴对齐、列筛选和基础参数验证，不包含具体模型或
特定分析流程。
"""

import numpy as np



def _validate_array(X, name, ndim):
    """
    检查并规范化数值数组。

    本函数只处理普通数组格式：转换为 ``float``，整理成指定维度，并检查是否
    含有 NaN 或 Inf。它不解释变量在模型中的角色。

    Parameters
    ----------
    X : array-like
        待检查的数组。
    name : str
        变量名，仅用于生成清晰的错误信息，例如 ``"X"`` 或 ``"Covariates"``。
    ndim : {1, 2}
        期望输出维度。``ndim=1`` 时会将输入展平为一维向量；``ndim=2`` 时，
        若输入为一维数组，会被解释为单列矩阵。

    Returns
    -------
    X : np.ndarray
        规范化后的浮点数组。

    Raises
    ------
    ValueError
        当输入为 ``None``、维度不符合要求、没有列，或包含 NaN/Inf 时抛出。
    """
    if X is None:
        raise ValueError(f"{name} must not be None.")
    if ndim not in (1, 2):
        raise ValueError("ndim must be 1 or 2.")

    X = np.asarray(X, dtype=float)
    if ndim == 1:
        # ndim=1 用于“每个被试一个数值”的变量，例如 y。
        #
        # 接受：
        #   - shape=(n,)：标准一维向量；
        #   - shape=(n, 1)：表格工具常见的单列数组；
        #   - shape=(1, n)：单行向量。
        # 只有这些“确实只包含一个变量”的输入会被压成 shape=(n,)。
        if X.ndim == 1:
            pass
        elif X.ndim == 2 and min(X.shape) == 1:
            X = X.ravel()
        else:
            raise ValueError(
                f"{name} must be a 1D array or a single-row/single-column 2D array."
            )

    elif ndim == 2:
        # ndim=2 用于“每个被试一个或多个变量”的矩阵，例如 X。
        #
        # 接受：
        #   - shape=(n, p) 且 p >= 1：标准二维矩阵；
        #   - shape=(n,)：单个变量的简写，会转换为 shape=(n, 1)。
        if X.ndim == 2:
            if X.shape[1] < 1:
                raise ValueError(f"{name} must contain at least one column.")
        elif X.ndim == 1:
            X = X[:, None]
        else:
            raise ValueError(
                f"{name} must be 2D, or 1D shorthand for one column; "
                f"got shape {X.shape}."
            )

    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    return X


def _check_subject_alignment(**arrays):
    """
    检查多个 subject-level 数组的被试轴是否对齐。

    所有 subject-level 数组的第一维都应表示被试数。这里把第一个数组的
    ``shape[0]`` 作为参照，并检查其余数组是否具有相同的第一维。

    Parameters
    ----------
    **arrays : dict
        名称到数组的映射。每个数组都应具有第一维，且第一维表示被试数。
        参数名只用于生成报错信息，例如 ``y.shape[0] must match X.shape[0]``。

    Returns
    -------
    n_subjects : int
        共同的样本数。

    Raises
    ------
    ValueError
        当没有传入数组，或任意两个数组第一维不一致时抛出。
    """
    if not arrays:
        raise ValueError("At least one array is required.")

    n_subjects = None
    first_name = None
    for name, arr in arrays.items():
        arr = np.asarray(arr)
        if n_subjects is None:
            n_subjects = arr.shape[0]
            first_name = name
        elif arr.shape[0] != n_subjects:
            raise ValueError(
                f"{name}.shape[0] must match {first_name}.shape[0] "
                f"({arr.shape[0]} != {n_subjects})."
            )

    return n_subjects


def _select_finite_variance_columns(X, eps=1e-8):
    """
    选择 finite 且方差不接近 0 的列。

    Parameters
    ----------
    X : array-like, shape (n_subjects, n_features)
        待筛选的二维特征矩阵。
    eps : float, default=1e-8
        方差/标准差阈值。列标准差必须大于该值才会被保留。

    Returns
    -------
    keep : np.ndarray, shape (n_features,), dtype bool
        布尔 mask。True 表示该列所有值均为 finite，且列标准差大于 ``eps``。
    """
    if X is None:
        raise ValueError("X must not be None.")

    # 这个函数的目的就是识别并排除含 NaN/Inf 的列，因此不能调用
    # _validate_array()；后者会在看到 NaN/Inf 时直接报错。
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    elif X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X.shape[1] < 1:
        raise ValueError("X must contain at least one column.")

    finite_cols = np.isfinite(X).all(axis=0)
    col_sd = np.zeros(X.shape[1], dtype=float)
    col_sd[finite_cols] = X[:, finite_cols].std(axis=0)

    return finite_cols & (col_sd > eps)


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

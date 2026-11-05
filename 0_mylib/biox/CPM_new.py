"""
CPM 工具函数的新版本草稿。

本模块用于逐步重组 ``CPM.py`` 和 ``CPM_ext.py`` 中重复的基础逻辑。
底层输入检查函数集中放在 ``helpers.py``；本文件只保留 CPM 分析本身需要的
步骤，包括输入组织、协变量残差化、fold 内选边、特征构建、模型拟合/预测、
nested CV 调参辅助函数，以及 selected-space Haufe 记录。
"""

from itertools import product

import numpy as np
import scipy.stats as stats

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.pipeline import Pipeline

from .helpers import (
    _check_subject_alignment,
    _select_finite_variance_columns,
    _validate_array,
    _validate_boolean,
    _validate_finite_float,
    _validate_integer,
)

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


# ---------------------------------------------------------------------
# Layer 1. CPM 通用配置、残差化与 nested CV 调参 helper
# ---------------------------------------------------------------------


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
        协变量矩阵，不包含截距；传入 None 或零列矩阵表示无协变量。
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
    X = _validate_array(X, name="X", ndim=2)
    y = _validate_array(y, name="y", ndim=1)

    if Covariates is None:
        Covariates = np.empty((X.shape[0], 0), dtype=float)
    else:
        # Covariates 允许零列矩阵，因为它是“无协变量”的内部标准格式；
        # _validate_array(..., ndim=2) 会拒绝零列矩阵，因此这里只在确有
        # 协变量列时复用它做完整检查。
        Covariates = np.asarray(Covariates, dtype=float)
        if Covariates.ndim == 1:
            Covariates = Covariates[:, None]
        elif Covariates.ndim != 2:
            raise ValueError(
                "Covariates must be None, a 1D array, or a 2D array."
            )
        if Covariates.shape[1] > 0:
            Covariates = _validate_array(Covariates, name="Covariates", ndim=2)
        elif not np.all(np.isfinite(Covariates)):
            raise ValueError("Covariates contains NaN or infinite values.")

    n_sub = _check_subject_alignment(X=X, y=y, Covariates=Covariates)
    n_columns = X.shape[1]

    if minimum_n_sub is not None and n_sub < minimum_n_sub:
        raise ValueError(f"At least {minimum_n_sub} subjects are required.")

    return X, y, Covariates, n_sub, n_columns


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


def _create_prediction_model(model_type, model_params, F_train, random_state):
    """
    为当前 fold 创建一个全新的预测模型对象。

    模型参数应已在调用前由 ``_validate_model_params()`` 验证。本函数只负责
    根据 ``model_type`` 和当前 fold 的有效特征数创建尚未拟合的 sklearn 模型，
    不执行 ``fit()``。

    交叉验证中，每个测试折只能使用对应训练折拟合出的模型。sklearn 模型在
    ``fit()`` 后会保存当前训练集学到的状态，例如回归系数、PLS 权重或 PCA
    主成分，因此不能在不同 fold 之间复用同一个已拟合模型对象。

    对 PLS/PCR 来说，模型结构还依赖当前 fold 的有效特征数。选边结果和常量
    特征筛除都可能让 ``F_train.shape[1]`` 在不同 fold 中变化，因此合法成分数
    需要在当前 fold 内确定。

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
        Lasso 和 ElasticNet 使用的随机种子；其他模型会忽略该参数。

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
      并要求 ``n_splits <= n_sub``。实际切分由后续 ``_create_cv_iterator()``
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


def _create_cv_iterator(cv_type, n_splits_actual, X, random_state=None):
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


def _design_matrix(Covariates, n_subjects):
    """
    构造包含截距的设计矩阵。

    本函数只负责一个清楚的数学步骤：把已经检查并整理好的 Covariates
    加上截距列，得到 ``C = [1, Covariates]``。原始输入的类型、维度、
    NaN/Inf 和被试数一致性检查应放在对外入口函数中完成，例如
    ``fast_partCorr_train()``。

    ``Covariates=None`` 表示没有协变量列；此时设计矩阵只包含截距。

    Parameters
    ----------
    Covariates : np.ndarray or None, shape (n_subjects, n_Covariates)
        已经检查过的协变量矩阵，不包含截距。传入 ``None`` 表示没有协变量列。
    n_subjects : int
        被试数。

    Returns
    -------
    C : np.ndarray, shape (n_subjects, n_Covariates + 1)
        设计矩阵。第一列为截距，后续列为 Covariates。
    """
    if Covariates is None:
        Covariates = np.empty((n_subjects, 0), dtype=float)

    return np.column_stack([np.ones(n_subjects), Covariates])


def _residualize_with_C(data_2d, C):
    """
    使用已经构造好的设计矩阵，对二维数据矩阵做残差化。

    本函数只负责线性代数计算：给定 ``data_2d`` 和 ``C``，估计
    ``data_2d ~ C`` 的回归系数，并返回残差。它不检查原始 Covariates，也不
    构造截距列。

    调用本函数前应保证 ``data_2d`` 和 ``C`` 均为二维数组，且二者第一维都
    对应同一批被试。

    Parameters
    ----------
    data_2d : np.ndarray, shape (n_subjects, n_variables)
        待残差化的二维数据矩阵。X 可以直接传入；y 需要先转成单列矩阵。
    C : np.ndarray, shape (n_subjects, n_Covariates + 1)
        已经包含截距列的设计矩阵。

    Returns
    -------
    data_resid : np.ndarray, shape (n_subjects, n_variables)
        残差化后的二维数据矩阵。
    beta : np.ndarray, shape (n_Covariates + 1, n_variables)
        每个变量对设计矩阵的回归系数。第一行是截距。
    """
    # C 的 Moore-Penrose 伪逆只需要计算一次。
    # 若 data_2d 有多列，例如 X 的多条 FC edge，下面的矩阵乘法会同时拟合
    # 每一列的线性模型：
    #
    #     data_2d = C @ beta
    #
    # beta 的每一列对应 data_2d 中一个变量的回归系数。
    C_pinv = np.linalg.pinv(C)
    beta = C_pinv @ data_2d

    # 残差 = 原始数据 - 由 C 解释的线性部分。
    # 在偏相关中，后续真正参与相关计算的是这些残差，而不是原始 X/y。
    data_resid = data_2d - C @ beta

    return data_resid, beta


def _grid_value_list(value, name):
    """
    将 param_grid 中某个参数的候选值统一为列表。

    nested CV 展开参数网格前，需要先把每个参数的候选值整理成相同格式。
    用户可能传入单个值、list、tuple 或 ``np.ndarray``；本函数统一返回
    list。例如 ``1.0`` 会变成 ``[1.0]``，``np.array([0.1, 1.0])`` 会变成
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
    展开参数网格，并生成每个候选方案对应的模型参数。

    本函数为 nested CV 生成需要逐一评估的候选参数方案。若 ``param_grid`` 是
    dict，则对各参数候选值做笛卡尔积；若是 list-of-dict，则认为用户已经手动
    给出了候选组合。随后，每个候选方案会覆盖 ``base_model_params`` 中的同名
    参数，并重新执行模型参数验证。

    例如 ElasticNet 的参数网格
    ``{"elasticnet_alpha": [0.1, 1.0], "l1_ratio": [0.2, 0.8]}`` 会展开为
    四个候选组合。展开前会先检查参数名是否属于当前模型，避免把 Ridge 的
    ``ridge_alpha`` 之类无关参数放进 ElasticNet 的 inner CV。

    Parameters
    ----------
    param_grid : dict or list of dict
        nested CV 的候选超参数。dict 会按键对应的候选值做笛卡尔积；list of dict
        表示用户已经手动指定候选组合。
    model_type : {'linear', 'ridge', 'lasso', 'elasticnet', 'pls', 'pcr'}
        当前预测模型类型，用于限制 ``param_grid`` 中允许出现的参数名。
    base_model_params : dict
        当前模型的基础参数。候选方案只覆盖同名键；未出现在候选方案中的参数
        会继续使用基础参数。

    Returns
    -------
    param_candidates : list of dict
        从 ``param_grid`` 展开的原始候选参数列表。每个元素只包含该候选方案
        显式覆盖的键。
    candidate_model_params : list of dict
        每个候选方案覆盖 ``base_model_params`` 后、重新验证得到的模型参数。
        该列表与 ``param_candidates`` 一一对应，可直接传给模型构建函数。

    Raises
    ------
    ValueError
        ``param_grid`` 类型错误、候选列表为空、包含当前模型不支持的参数名，
        或合并后的模型参数不合法时抛出。
    """
    model_type = str(model_type).lower()
    allowed = set(_TUNABLE_PARAMS.get(model_type, set()))

    if isinstance(param_grid, dict):
        # set(param_grid) 是用户提供的参数名；减去 allowed 后，剩下的就是
        # 当前模型不支持的参数。
        unknown = set(param_grid) - allowed
        if unknown:
            raise ValueError(
                "param_grid contains unsupported parameter(s) for "
                f"model_type='{model_type}': {sorted(unknown)}."
            )

        if len(param_grid) == 0:
            candidates = [{}]
        else:
            # keys 保存参数名顺序；value_lists 保存每个参数对应的候选值列表。
            # 例如 ElasticNet 中：
            # keys = ["elasticnet_alpha", "l1_ratio"]
            # value_lists = [[0.1, 1.0], [0.2, 0.8]]
            keys = list(param_grid)
            value_lists = [_grid_value_list(param_grid[key], key) for key in keys]

            # product(*value_lists) 生成所有候选值组合；zip(keys, values) 再把
            # 每组候选值配回参数名。若 values = (0.1, 0.2)，则
            # dict(zip(keys, values)) 得到
            # {"elasticnet_alpha": 0.1, "l1_ratio": 0.2}。
            candidates = [dict(zip(keys, values)) for values in product(*value_lists)]

    elif isinstance(param_grid, (list, tuple)):
        # list-of-dict 表示用户已经手动指定候选组合，不再做笛卡尔积展开。
        candidates = []
        for candidate in param_grid:
            if not isinstance(candidate, dict):
                raise ValueError("Each item in param_grid must be a dict.")
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
        # candidate 只覆盖一个候选组合中的同名参数。例如 candidate =
        # {"l1_ratio": 0.8} 时，其他基础参数会从 base_model_params 保留。
        merged = base_model_params.copy()
        merged.update(candidate)

        # 合并后重新验证，可以确保每个实际进入模型的候选参数都满足类型和取值
        # 范围要求，例如 l1_ratio 必须落在 [0, 1]。
        _, merged = _validate_model_params(model_type=model_type, **merged)
        candidate_model_params.append(merged)

    return candidates, candidate_model_params


def _score_predictions(y_true, y_pred, scoring):
    """
    计算 nested CV 内层验证集的调参分数；分数越大越好。

    本函数用于比较候选超参数的 inner out-of-fold 预测表现。为了让选择逻辑统一，
    所有指标都转换成“越大越好”：相关和 R² 直接使用原值；RMSE 和 MAE 这类
    误差指标返回负值，即 ``neg_rmse`` 和 ``neg_mae``。

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        内层验证样本的真实目标残差。
    y_pred : array-like, shape (n_samples,)
        与 ``y_true`` 对应的预测值。
    scoring : {'pearson_r', 'r2', 'neg_rmse', 'neg_mae'}
        内层 CV 用于选择超参数的指标。

    Returns
    -------
    score : float
        当前候选参数的验证分数。若 Pearson r 或 R² 因零方差无法定义，则返回
        ``np.nan``。

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

    本函数只接触当前 outer training set，不接触 outer test set。它先把 outer
    training set 再切成 inner train 和 inner validation；每个 inner train 内
    独立完成 CPM 残差化、偏相关选边和特征构建；随后对同一份 inner fold 特征
    评估所有候选模型参数。候选分数基于拼接后的 inner out-of-fold 预测计算。

    这里复用 ``fit_cpm_fold_model_with_haufe()`` 的拟合和预测逻辑，但传入
    ``compute_haufe=False``。inner CV 的目的只是选择超参数，不需要在每个候选
    参数和每个 inner fold 中计算 Haufe。

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
    random_state : int or None
        内层 KFold 切分和随机模型使用的随机种子。
    selection_params : dict
        固定的 CPM 选边参数，会传递给 ``prepare_cpm_fold_features()``。
    standardize_features : bool
        是否在每个 inner fold 内对训练特征和验证特征进行训练集标准化。
    scoring : {'pearson_r', 'r2', 'neg_rmse', 'neg_mae'}
        内层 CV 的调参指标。

    Returns
    -------
    best_model_params : dict
        inner out-of-fold 分数最高的模型参数。若所有候选分数均为 ``np.nan``，
        则返回第一个候选参数。
    best_score : float
        最佳候选参数对应的内层验证分数；若所有候选均无法评分，则为 ``np.nan``。
    candidate_scores : list of dict
        每个候选参数的原始候选值、完整模型参数和内层验证分数。每个 dict 包含
        ``candidate_params``、``model_params`` 和 ``score``。
    n_splits_actual : int
        内层 CV 实际使用的折数；LOOCV 时等于当前 outer training set 的样本量。

    Raises
    ------
    ValueError
        候选参数列表为空、两个候选列表长度不一致，或内层 CV 设置无效时抛出。
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
        cv_iter = _create_cv_iterator("loocv", n_splits_actual, X)
    elif inner_cv_type == "kfold":
        n_splits_actual = _validate_integer(
            inner_n_splits,
            "inner_n_splits",
            minimum=2,
        )
        if n_splits_actual > n_sub:
            raise ValueError("inner_n_splits cannot be larger than outer train size.")
        cv_iter = _create_cv_iterator(
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
        prepared = prepare_cpm_fold_features(
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
        true_parts.append(prepared["y_resid_test"])

        for candidate_idx, model_params in enumerate(candidate_model_params):
            # inner CV 只需要预测值用于调参，因此复用统一的 fold 拟合函数，
            # 但关闭 Haufe 计算。edge/haufe records 会被返回，但这里不保存。
            fold_random_state = (
                None if random_state is None else random_state + inner_fold_idx
            )
            fold_result = fit_cpm_fold_model_with_haufe(
                repeat_idx=0,
                fold_idx=inner_fold_idx,
                prepared_fold=prepared,
                model_type=model_type,
                model_params=model_params,
                random_state=fold_random_state,
                compute_haufe=False,
            )
            pred = fold_result["y_pred_test"]

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


# ---------------------------------------------------------------------
# Layer 2. CPM fold 特征准备、模型拟合与 Haufe 记录
# ---------------------------------------------------------------------


def fast_partCorr_train(X, y, Covariates):
    """
    计算训练集中 X 的每一列与 y 之间的偏相关系数，并控制 Covariates。

    该版本复用通用数组校验函数和残差化底层函数。X 和 y 共用同一个设计矩阵
    ``C = [1, Covariates]``。原始输入检查集中放在本函数入口处，后续 helper
    只处理已经整理好的数组。
    返回值与旧版 ``biox.CPM.fast_partCorr_train()`` 保持一致。

    Parameters
    ----------
    X : array-like, shape (n_subjects, n_features)
        自变量矩阵，每一列为一个待分析的连接边或其他预测特征。
    y : array-like, shape (n_subjects,)
        因变量，例如行为得分。
    Covariates : array-like or None, shape (n_subjects, n_Covariates)
        需要控制的协变量矩阵，不包含截距。传入 ``None`` 表示只控制截距。

    Returns
    -------
    r_vals : np.ndarray, shape (n_features,)
        X 每一列与 y 在控制 Covariates 后的偏相关系数。
    p_vals : np.ndarray, shape (n_features,)
        对应的双尾 p 值。
    X_resid : np.ndarray, shape (n_subjects, n_features)
        X 的残差矩阵。
    y_resid : np.ndarray, shape (n_subjects,)
        y 的残差向量。
    beta_X : np.ndarray, shape (n_Covariates + 1, n_features)
        X 各列对设计矩阵的回归系数。
    beta_y : np.ndarray, shape (n_Covariates + 1,)
        y 对设计矩阵的回归系数。

    Raises
    ------
    ValueError
        输入形状不一致、包含非有限值，或训练样本不足以提供有效残差自由度时抛出。
    """
    # ----- 0. 输入格式统一与完整性检查 -----
    X, y, Covariates, n_subjects, n_features = _validate_X_y_Covariates(
        X,
        y,
        Covariates,
    )

    # ----- 1. 构建设计矩阵 C = [1, Covariates] -----
    # 第一列全 1，对应线性模型中的截距项；后续列是需要控制的协变量。
    C = _design_matrix(Covariates, n_subjects)

    # ----- 2. 分别残差化 X 和 y -----
    # 对每条 FC edge 拟合 X_edge ~ C，并对 y 拟合 y ~ C。
    # 得到的 X_resid 和 y_resid 均已剔除截距和 Covariates 的线性影响。
    X_resid, beta_X = _residualize_with_C(X, C)
    y_resid_2d, beta_y_2d = _residualize_with_C(y[:, None], C)
    y_resid = y_resid_2d.ravel()
    beta_y = beta_y_2d[:, 0]

    # ----- 3. 计算残差之间的相关系数，即偏相关系数 -----
    # 分子：X_resid 每一列与 y_resid 的点积。
    num = X_resid.T @ y_resid

    # 分母：sqrt(sum(X_resid^2) * sum(y_resid^2))。
    # ss_X 是每条 edge 的残差平方和；ss_y 是 y 残差平方和。
    ss_X = (X_resid**2).sum(axis=0)
    ss_y = y_resid @ y_resid
    den = np.sqrt(ss_X * ss_y)

    # 仅在分母有效时计算相关。若某条 edge 或 y 的残差没有变异，
    # 相关系数没有定义，保留为 NaN，后续选边函数会自然排除它。
    r_vals = np.divide(
        num,
        den,
        out=np.full(n_features, np.nan, dtype=float),
        where=den > _EPS,
    )
    r_vals = np.clip(r_vals, -1.0, 1.0)

    # ----- 4. 将偏相关系数转换为双尾 p 值 -----
    # 设计矩阵的秩通常等于协变量数 + 1；若协变量完全共线，秩会更低。
    rank_C = np.linalg.matrix_rank(C)

    # 偏相关自由度 = 样本数 - 设计矩阵秩 - 1。
    df = n_subjects - rank_C - 1
    if df <= 0:
        # 自由度不足说明样本数相对协变量过少；继续返回 NaN 会掩盖问题，
        # 因此直接报错。
        raise ValueError(
            "Insufficient residual degrees of freedom: "
            f"n_subjects={n_subjects}, rank(design)={rank_C}, df={df}."
        )

    # 只为 t 值计算把 |r|=1 裁到浮点安全范围；返回的 r_vals 仍保留在 [-1, 1]。
    r_for_t = np.clip(r_vals, -1.0 + _EPS, 1.0 - _EPS)
    t = r_for_t * np.sqrt(df / (1 - r_for_t**2))

    # 双尾检验：2 * survival function，即 2 * (1 - CDF)。
    p_vals = 2 * stats.t.sf(np.abs(t), df)

    # 返回偏相关、p 值、残差和回归系数，保持与旧版 CPM.py 一致。
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
    top_k_mode : {'total', 'per_sign', 'positive', 'negative'}, default='total'
        ``total`` 在全部有效边中选择；``per_sign`` 分别从正、负边中各选择；
        ``positive`` 只从正边中选择；``negative`` 只从负边中选择。
    max_edges_total : int or None, default=None
        最终返回的最大边数，若超过则截断。
    min_edges_total : int, default=0
        当 fallback_if_too_few=True 时，最少需要返回的边数。
    fallback_if_too_few : bool, default=False
        初始结果不足时，是否回退到当前候选池中 p 值最小的
        ``min_edges_total`` 条有效边。若 ``top_k_mode`` 为 ``positive`` 或
        ``negative``，fallback 也只在对应方向内补足。

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
    总边数上限在 fallback 后执行，因此最终边数可能低于 ``min_edges_total``。
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
        if top_k_mode not in {"total", "per_sign", "positive", "negative"}:
            raise ValueError(
                "top_k_mode must be 'total', 'per_sign', 'positive', or 'negative'."
            )

    # NaN/Inf 的效应量或 p 值不能参与选择；r=0 的边没有正负方向，也无法进入
    # 后续正/负 CPM strength，因此在这里一并排除。
    valid_mask = np.isfinite(r_vals) & np.isfinite(p_vals) & (r_vals != 0)
    valid_idx = np.flatnonzero(valid_mask)
    fallback_pool = valid_idx

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

        elif top_k_mode == "positive":
            # 只在正相关边中按 p 值升序选择 top_k 条。
            pos_all = valid_idx[r_vals[valid_idx] > 0]
            fallback_pool = pos_all
            idx = pos_all[np.argsort(p_vals[pos_all], kind="stable")[:top_k]]

        elif top_k_mode == "negative":
            # 只在负相关边中按 p 值升序选择 top_k 条。
            neg_all = valid_idx[r_vals[valid_idx] < 0]
            fallback_pool = neg_all
            idx = neg_all[np.argsort(p_vals[neg_all], kind="stable")[:top_k]]

    # ---------- 2. 边数不足时执行 fallback ----------
    # fallback 仍然只从有效候选池中选择，不会为了凑数引入 NaN/Inf 或 r=0 的边。
    # 对 positive/negative 模式，候选池已经限定为相应方向，因此不会跨方向补足。
    if fallback_if_too_few and idx.size < min_edges_total:
        used_fallback = True
        sorted_valid = fallback_pool[np.argsort(p_vals[fallback_pool], kind="stable")]
        idx = sorted_valid[:min_edges_total]

    # ---------- 3. 总边数硬上限 ----------
    # 放在最后执行，确保 fallback 不会突破 max_edges_total。
    if max_edges_total is not None:
        idx = idx[:max_edges_total]

    # 返回索引数组（整数类型）以及是否使用了回退的标记。
    return np.asarray(idx, dtype=int), used_fallback


def prepare_cpm_fold_features(
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
    model_type,
    standardize_features,
):
    """
    在单个交叉验证 fold 中完成边选择，并构造训练/测试特征。

    本函数的核心原则是：所有需要“从数据中估计规则”的步骤都只使用训练折。
    这些规则包括协变量残差化系数、偏相关选边结果和常量特征筛除规则。若启用
    Z-score 标准化，均值和标准差也只从训练折的 ``F_train`` 中估计。测试折不
    参与这些规则的估计；它只使用训练折得到的 ``beta_X``、``beta_y``、选中边、
    保留特征列和标准化参数，生成 ``F_test`` 和 ``y_resid_test``。这样后续
    模型拟合和预测性能评估才保持折外、无数据泄漏。

    返回的 ``F_train``、``feature_edge_idx`` 和标准化参数也为 selected-space
    Haufe 分析保留必要信息。对于高维 CPM 模型，每一列 ``F_train`` 都对应
    ``feature_edge_idx`` 中的一条原始 FC edge；对于标准 CPM 的 ``linear``
    模型，输入是 ``pos_strength`` / ``neg_strength`` 汇总特征，因此不能直接
    进行 edge-level Haufe 映射。

    Parameters
    ----------
    X_train, X_test : np.ndarray
        当前 fold 的训练集和测试集连接边矩阵，形状分别为
        ``(n_train, n_edges)`` 和 ``(n_test, n_edges)``。
    y_train, y_test : np.ndarray
        当前 fold 的训练集和测试集连续目标变量。
    Covariates_train, Covariates_test : array-like or None
        当前 fold 的训练集和测试集协变量矩阵；传入 ``None`` 表示无协变量。
    selection_method, p_thresh, top_k, top_k_mode, max_edges_total,
    min_edges_total, fallback_if_too_few
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
        - ``y_resid_test`` : np.ndarray, shape (n_test,)
          测试折目标残差，由训练折估计的 ``beta_y`` 作用于测试折计算得到。
        - ``pos_idx`` / ``neg_idx`` : np.ndarray
          ``select_edges()`` 选中的正边和负边索引。
        - ``feature_edge_idx`` : np.ndarray, shape (n_features_kept,)
          实际进入高维模型的 edge 索引。``model_type='linear'`` 使用 network
          strength，不把单条 edge 作为模型特征，因此该字段为空数组。
        - ``feature_names`` : np.ndarray, shape (n_features_kept,)
          实际进入模型的特征名称，与 ``F_train`` / ``F_test`` 的列一一对应。
          该字段主要用于结果解释：标准 CPM 中通常是 ``pos_strength`` 和/或
          ``neg_strength``；高维模型中通常是 ``edge_12``、``edge_98`` 这类
          与原始 edge 编号对应的名称。
        - ``standardize_features`` : bool
          ``F_train`` / ``F_test`` 是否按训练折均值和标准差做过 Z-score 标准化。
        - ``feature_mean`` / ``feature_sd`` : np.ndarray, shape (n_features_kept,) or (0,)
          若 ``standardize_features=True`` 且存在有效特征，分别保存标准化前
          ``F_train`` 每列的训练折均值和标准差；否则为空数组。这两个字段用于
          记录 Haufe 等后续分析对应的特征尺度。
        - ``used_fallback`` : bool
          本 fold 是否触发 fallback 选边。
        - ``has_features`` : bool
          当前 fold 是否最终保留了可用于建模的非恒定特征。该字段主要用于
          后续流程判断：若没有 edge 通过筛选，或筛除常量特征后没有特征剩余，
          则 ``has_features=False``，预测函数应跳过模型拟合并使用默认预测。

    Raises
    ------
    ValueError
        ``fast_partCorr_train()`` 返回值形状异常，或下游选边参数无效时抛出。
    """
    X_train, y_train, Covariates_train, _, n_edges = _validate_X_y_Covariates(
        X_train,
        y_train,
        Covariates_train,
    )
    X_test, y_test, Covariates_test, _, n_test_edges = _validate_X_y_Covariates(
        X_test,
        y_test,
        Covariates_test,
    )
    if n_test_edges != n_edges:
        raise ValueError("X_test must contain the same number of columns as X_train.")
    if Covariates_test.shape[1] != Covariates_train.shape[1]:
        raise ValueError(
            "Covariates_test must contain the same number of columns as "
            "Covariates_train."
        )

    model_type = str(model_type).lower()
    if model_type not in _SUPPORTED_MODELS:
        raise ValueError(
            "model_type must be one of: "
            "'linear', 'ridge', 'lasso', 'elasticnet', 'pls', or 'pcr'."
        )
    standardize_features = _validate_boolean(
        standardize_features,
        "standardize_features",
    )

    (
        r_vals,
        p_vals,
        X_resid_train,
        y_resid_train,
        beta_X,
        beta_y,
    ) = fast_partCorr_train(X_train, y_train, Covariates_train)

    C_test = _design_matrix(Covariates_test, X_test.shape[0])
    y_resid_test = y_test - C_test @ beta_y

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
    )

    if idx.size > 0:
        pos_idx = idx[r_vals[idx] > 0]
        neg_idx = idx[r_vals[idx] < 0]
    else:
        pos_idx = np.array([], dtype=int)
        neg_idx = np.array([], dtype=int)

    # 两种情况都会导致“当前 fold 无法建模”：
    # 1) 没有任何正/负边通过选边；
    # 2) 入选边构成的特征在训练折中没有有效方差。
    # 这里先统一无效 fold 的返回结构，避免多个提前返回分支维护不同字段。
    empty_prepared_fold = {
        "F_train": None,
        "F_test": None,
        "y_resid_train": y_resid_train,
        "y_resid_test": y_resid_test,
        "pos_idx": pos_idx,
        "neg_idx": neg_idx,
        "feature_edge_idx": np.array([], dtype=int),
        "feature_names": np.array([], dtype=object),
        "standardize_features": standardize_features,
        "feature_mean": np.array([], dtype=float),
        "feature_sd": np.array([], dtype=float),
        "used_fallback": used_fallback,
        "has_features": False,
    }

    if pos_idx.size + neg_idx.size == 0:
        # 没有任何正/负边可进入 CPM 特征构建；后续预测函数会识别
        # has_features=False 并返回全 0 预测，避免空特征矩阵导致模型报错。
        return empty_prepared_fold

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

    keep = _select_finite_variance_columns(F_train, eps=_EPS)

    if not np.any(keep):
        # 入选特征若在训练折中全是常量，就无法为模型提供可学习信息；此时该
        # fold 视为没有有效特征。
        return empty_prepared_fold

    F_train = F_train[:, keep]
    F_test = F_test[:, keep]
    # feature_names 必须和 F_train/F_test 的列保持一一对应；高维模型的
    # feature_edge_idx 也同步筛除常量边，便于之后将 coef 对齐到原始 edge。
    feature_names = feature_names[keep]
    if model_type != "linear":
        feature_edge_idx = feature_edge_idx[keep]

    feature_mean = np.array([], dtype=float)
    feature_sd = np.array([], dtype=float)
    if standardize_features:
        feature_mean = np.mean(F_train, axis=0)
        feature_sd = np.std(F_train, axis=0, ddof=0)

        F_train = (F_train - feature_mean) / feature_sd
        F_test = (F_test - feature_mean) / feature_sd

    return {
        "F_train": F_train,
        "F_test": F_test,
        "y_resid_train": y_resid_train,
        "y_resid_test": y_resid_test,
        "pos_idx": pos_idx,
        "neg_idx": neg_idx,
        "feature_edge_idx": feature_edge_idx,
        "feature_names": feature_names,
        "standardize_features": standardize_features,
        "feature_mean": feature_mean,
        "feature_sd": feature_sd,
        "used_fallback": used_fallback,
        "has_features": True,
    }


def fit_cpm_fold_model_with_haufe(
    repeat_idx,
    fold_idx,
    prepared_fold,
    model_type,
    model_params,
    random_state,
    compute_haufe=True,
    eps=1e-12,
):
    """
    拟合单个 CPM fold 模型，并返回预测、edge record 和可选 Haufe record。

    本函数接收 ``prepare_cpm_fold_features()`` 的输出。前一步已经完成协变量
    残差化、偏相关选边、特征构建、常量特征筛除和可选标准化；本函数只负责
    当前 fold 的模型层操作：

    1. 用 ``F_train`` 和 ``y_resid_train`` 拟合预测模型；
    2. 用 ``F_test`` 预测当前测试折，得到 ``y_pred_test``；
    3. 从 fitted model 中提取 ``coef``，并与选边信息合并成 ``edge_record``；
    4. 若 ``compute_haufe=True``，用 fitted model 在训练折上的预测值计算
       selected-space Haufe。

    高维 CPM 模型（``ridge``、``lasso``、``elasticnet``、``pls`` 和 ``pcr``）
    的每一列 ``F_train`` 都对应一条 selected edge，因此可以计算 edge-level
    Haufe。标准 CPM 的 ``linear`` 模型使用 ``pos_strength`` / ``neg_strength``
    汇总特征，仍会正常拟合、预测并返回 ``edge_record``，但
    ``haufe_record['valid_haufe']`` 会设为 ``False``。

    若当前 fold 没有有效特征，函数不拟合模型，返回全 0 测试折预测、空
    ``edge_record`` 和无效的 ``haufe_record``。因为 CPM 预测的是 y 的残差，
    全 0 预测表示“预测为训练残差中心附近”。

    Parameters
    ----------
    repeat_idx : int
        当前 outer repeat 的零基编号。
    fold_idx : int
        当前 outer fold 的零基编号。
    prepared_fold : dict
        ``prepare_cpm_fold_features()`` 返回的 fold 特征字典。
    model_type : {'linear', 'ridge', 'lasso', 'elasticnet', 'pls', 'pcr'}
        当前 fold 使用的预测模型类型。``linear`` 可以拟合和预测，但不计算
        edge-level Haufe。
    model_params : dict
        已通过 ``_validate_model_params()`` 验证的模型参数，只包含当前模型实际
        使用的键。
    random_state : int
        Lasso 和 ElasticNet 使用的随机种子；其他模型会忽略该参数。
    compute_haufe : bool, default=True
        是否计算 selected-space Haufe。outer CV 的最终模型通常应设为 True；
        inner CV 调参只需要预测值，可设为 False 以复用同一套拟合/预测逻辑并
        跳过 Haufe 计算。
    eps : float, default=1e-12
        判断近零方差时使用的数值阈值。

    Returns
    -------
    fold_result : dict
        当前 fold 的模型结果。主要键包括：

        - ``y_pred_test`` : np.ndarray, shape (n_test,)
          当前测试折的预测值，目标尺度为训练折协变量残差。
        - ``edge_record`` : dict
          当前 fold 的选边结果、模型特征名称和模型系数。
        - ``haufe_record`` : dict
          当前 fold 的 selected-space Haufe 结果。若 ``compute_haufe=False``，
          该记录会保留标准字段，但 ``valid_haufe=False`` 且
          ``haufe_reason='not_requested'``。

    edge_record : dict
        ``fold_result['edge_record']`` 的主要键包括：

        - ``repeat`` / ``fold`` : int
          当前记录对应的 outer repeat 和 outer fold 零基编号。
        - ``model_type`` : str
          当前 fold 使用的预测模型类型。
        - ``feature_names`` : np.ndarray, shape (n_features_kept,)
          与 ``coef`` 一一对应的模型特征名称。
        - ``coef`` : np.ndarray, shape (n_features_kept,)
          当前 fold 模型在实际输入特征空间中的系数。PCR 的系数会从 PCA 成分
          空间回投到当前 fold 的 selected feature space。
        - ``edge_idx`` : np.ndarray, shape (n_features_kept,) or (0,)
          高维 CPM 中与 ``coef`` 对应的原始 edge 索引。
        - ``pos_idx`` / ``neg_idx`` : np.ndarray
          当前 fold 被 CPM 选中的正边和负边索引。
        - ``n_pos`` / ``n_neg`` : int
          当前 fold 被 CPM 选中的正边数和负边数。
        - ``standardize_features`` : bool
          当前 fold 的模型特征是否经过训练折 Z-score 标准化。
        - ``feature_mean`` / ``feature_sd`` : np.ndarray, shape (n_features_kept,) or (0,)
          若启用标准化，保存标准化前训练折特征的均值和标准差。
        - ``used_fallback`` / ``has_features`` : bool
          当前 fold 是否触发 fallback，以及是否存在有效建模特征。

    haufe_record : dict
        ``fold_result['haufe_record']`` 的主要键包括：

        - ``repeat`` / ``fold`` : int
          当前记录对应的 outer repeat 和 outer fold 零基编号。
        - ``model_type`` : str
          当前 fold 使用的预测模型类型。
        - ``edge_idx`` : np.ndarray, shape (n_selected_features,)
          当前 fold 实际进入模型、且成功计算 Haufe 的 edge 编号。
        - ``haufe_slope`` : np.ndarray, shape (n_selected_features,)
          ``Cov(F_e, yhat) / Var(yhat)``。
        - ``haufe_corr`` : np.ndarray, shape (n_selected_features,)
          ``Corr(F_e, yhat)``，范围限制在 [-1, 1]。
        - ``n_selected_features`` : int
          当前 fold 成功计算 Haufe 的 selected edge 数。
        - ``valid_haufe`` : bool
          当前 fold 是否成功得到 Haufe 值。
        - ``haufe_reason`` : str
          Haufe 状态说明。``"computed"`` 表示成功计算；``"linear_strength_features"``
          表示标准 CPM 只有 network-strength 特征，不能映射回单条 edge；
          ``"not_requested"`` 表示本次调用主动跳过 Haufe；
          其他取值说明当前 fold 没有有效 Haufe 结果的原因。

    Raises
    ------
    ValueError
        模型系数数量与 fold 特征数不一致时抛出。
    """
    model_type = str(model_type).lower()
    compute_haufe = _validate_boolean(compute_haufe, "compute_haufe")
    fold_id = {
        "repeat": int(repeat_idx),
        "fold": int(fold_idx),
        "model_type": model_type,
    }

    y_resid_test = prepared_fold["y_resid_test"]
    feature_edge_idx = np.asarray(prepared_fold["feature_edge_idx"], dtype=int)
    feature_names = np.asarray(prepared_fold["feature_names"], dtype=object)

    # 默认返回用于处理“无有效特征”的 fold。此时不拟合模型，预测值设为 0；
    # 这对应“测试折目标残差预测为训练残差中心附近”的基线预测。
    coef = np.array([], dtype=float)
    y_pred_test = np.zeros(y_resid_test.shape[0], dtype=float)
    if not compute_haufe:
        haufe_reason = "not_requested"
    elif model_type == "linear":
        haufe_reason = "linear_strength_features"
    else:
        haufe_reason = "no_valid_features"
    haufe_record = {
        **fold_id,
        "edge_idx": np.array([], dtype=int),
        "haufe_slope": np.array([], dtype=float),
        "haufe_corr": np.array([], dtype=float),
        "n_selected_features": 0,
        "valid_haufe": False,
        "haufe_reason": haufe_reason,
    }

    if prepared_fold["has_features"]:
        F_train = prepared_fold["F_train"]
        F_test = prepared_fold["F_test"]

        # 每个 fold 都创建并拟合一个新模型；模型状态只能来自当前训练折。
        model = _create_prediction_model(
            model_type=model_type,
            model_params=model_params,
            F_train=F_train,
            random_state=random_state,
        )
        model.fit(F_train, prepared_fold["y_resid_train"])
        y_pred_test = np.asarray(model.predict(F_test)).ravel()

        # 记录模型在“实际输入特征空间”中的系数。PCR 是唯一需要回投的情况：
        # 线性回归先在 PCA 成分空间拟合，因此要乘以 PCA components 才能回到
        # 当前 fold 的 selected feature space。
        if model_type == "pcr":
            pca = model.named_steps["pca"]
            reg = model.named_steps["linear_regression"]
            component_coef = np.asarray(reg.coef_, dtype=float).ravel()
            coef = pca.components_.T @ component_coef
        else:
            coef = np.asarray(model.coef_, dtype=float).ravel()

        n_features = feature_names.size
        if coef.size != n_features:
            raise ValueError(
                "Model coefficients must have the same length as fold features; "
                f"got {coef.size}, expected {n_features}."
            )
        coef = coef.astype(float, copy=False)

        # Haufe 只在明确需要时计算。inner CV 调参阶段只需要预测值，因此可通过
        # compute_haufe=False 跳过这一段，避免在每个候选参数上重复计算 Haufe。
        #
        # 对标准 CPM linear 模型，即使 compute_haufe=True，也会保留默认的无效
        # Haufe record，因为它的模型特征是 pos/neg strength 汇总，不是一条条
        # 原始 edge。
        if compute_haufe and model_type != "linear":
            edge_idx = feature_edge_idx.ravel()
            F_train_haufe = np.asarray(F_train, dtype=np.float64)

            if edge_idx.size == 0:
                haufe_record["haufe_reason"] = "no_edge_level_features"
            elif F_train_haufe.ndim != 2 or F_train_haufe.shape[1] != edge_idx.size:
                haufe_record["haufe_reason"] = "feature_edge_mismatch"
            else:
                # selected-space Haufe 使用训练折中真正进入模型的特征 F_train，
                # 并考察这些特征与模型预测值 yhat_train 的关系。
                yhat_train = np.asarray(
                    model.predict(F_train_haufe),
                    dtype=np.float64,
                ).ravel()
                yhat_dm = yhat_train - yhat_train.mean()
                yhat_ss = float(yhat_dm @ yhat_dm)

                if not (np.isfinite(yhat_ss) and yhat_ss > eps):
                    haufe_record["haufe_reason"] = "zero_prediction_variance"
                else:
                    F_dm = F_train_haufe - F_train_haufe.mean(axis=0, keepdims=True)
                    numerator = F_dm.T @ yhat_dm
                    f_ss = np.sum(F_dm**2, axis=0)
                    valid = np.isfinite(f_ss) & (f_ss > eps)

                    if not np.any(valid):
                        haufe_record["haufe_reason"] = "no_valid_feature_variance"
                    else:
                        edge_idx = edge_idx[valid]
                        numerator = numerator[valid]
                        f_ss = f_ss[valid]

                        # slope 给出 Cov(F_e, yhat) / Var(yhat)；corr 给出同一关系
                        # 的标准化版本。二者都只在当前 fold 的 selected features
                        # 上计算，后续再由 _summarize_cpm_haufe_records() 汇总回
                        # 全部 edge 空间。
                        haufe_slope = numerator / yhat_ss
                        haufe_corr = numerator / np.sqrt(
                            np.maximum(f_ss * yhat_ss, eps)
                        )
                        haufe_corr = np.clip(haufe_corr, -1.0, 1.0)

                        haufe_record = {
                            **fold_id,
                            "edge_idx": edge_idx.astype(int, copy=False),
                            "haufe_slope": haufe_slope.astype(float, copy=False),
                            "haufe_corr": haufe_corr.astype(float, copy=False),
                            "n_selected_features": int(edge_idx.size),
                            "valid_haufe": True,
                            "haufe_reason": "computed",
                        }

    # edge_record 无论 Haufe 是否有效都会返回。它保存当前 fold 的选边、
    # 模型特征和模型系数，是后续查看 stable edges / selected coefficients
    # 的基础记录。
    edge_record = {
        **fold_id,
        "feature_names": feature_names.copy(),
        "coef": coef.copy(),
        "edge_idx": feature_edge_idx.copy(),
        "pos_idx": prepared_fold["pos_idx"].copy(),
        "neg_idx": prepared_fold["neg_idx"].copy(),
        "n_pos": int(prepared_fold["pos_idx"].size),
        "n_neg": int(prepared_fold["neg_idx"].size),
        "standardize_features": bool(prepared_fold["standardize_features"]),
        "feature_mean": prepared_fold["feature_mean"].copy(),
        "feature_sd": prepared_fold["feature_sd"].copy(),
        "used_fallback": bool(prepared_fold["used_fallback"]),
        "has_features": bool(prepared_fold["has_features"]),
    }

    return {
        "y_pred_test": y_pred_test,
        "edge_record": edge_record,
        "haufe_record": haufe_record,
    }


def _summarize_cpm_haufe_records(haufe_records, n_edges):
    """
    汇总 fold-level selected-space Haufe records。

    ``fit_cpm_fold_model_with_haufe()`` 每次返回一个 ``fold_result``，其中的
    ``haufe_record`` 只包含当前 fold 实际进入模型的 selected edges。该函数把
    所有 fold-level Haufe 结果累加回长度为 ``n_edges`` 的全脑 edge vector，
    得到跨 fold 的 selected-space Haufe map。

    汇总规则为：

    - 某条 edge 在某个 fold 被选中且成功计算 Haufe 时，累加该 fold 的
      ``haufe_slope`` 和 ``haufe_corr``；
    - 没有在该 fold 进入模型的 edge，对该 fold 贡献 0；
    - 最后除以成功计算 Haufe 的 fold 数 ``n_valid_folds``。

    因此，返回的 Haufe 均值是 selected-space across-fold mean：它同时反映
    edge 被选中的稳定性和被选中后与模型预测的关系。

    Parameters
    ----------
    haufe_records : list of dict
        每个元素为 ``fit_cpm_fold_model_with_haufe()`` 返回的
        ``fold_result['haufe_record']``。
    n_edges : int
        原始 FC edge 总数，用于确定输出全脑向量长度。

    Returns
    -------
    haufe_results : dict
        跨 fold 汇总后的 Haufe 结果。主要键包括：

        - ``fold_records`` : list of dict
          原始 fold-level Haufe records。
        - ``n_valid_folds`` : int
          成功计算 Haufe 的 fold 数。
        - ``selected_slope_sum`` / ``selected_corr_sum`` : np.ndarray, shape (n_edges,)
          每条 edge 在有效 Haufe folds 中累加的 slope/correlation。
        - ``selected_count`` : np.ndarray, shape (n_edges,)
          每条 edge 在有效 Haufe folds 中被实际用于 Haufe 计算的次数。
        - ``haufe_slope_selected_mean_all_folds`` : np.ndarray, shape (n_edges,)
          每条 edge 的 selected-space Haufe slope 均值。
        - ``haufe_corr_selected_mean_all_folds`` : np.ndarray, shape (n_edges,)
          每条 edge 的 selected-space Haufe correlation 均值。
        - ``haufe_slope_abs_selected_mean_all_folds`` : np.ndarray, shape (n_edges,)
          ``haufe_slope_selected_mean_all_folds`` 的绝对值。
        - ``haufe_corr_abs_selected_mean_all_folds`` : np.ndarray, shape (n_edges,)
          ``haufe_corr_selected_mean_all_folds`` 的绝对值。
        - ``haufe_selection_count`` : np.ndarray, shape (n_edges,)
          等同于 ``selected_count``，便于按 Haufe 语境读取。
        - ``haufe_selection_freq`` : np.ndarray, shape (n_edges,)
          ``selected_count / n_valid_folds``；若没有有效 Haufe fold，则全为 0。

    Raises
    ------
    ValueError
        任一有效 Haufe record 中 ``edge_idx``、``haufe_slope`` 和 ``haufe_corr``
        长度不一致时抛出。
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
            raise ValueError(
                "Each Haufe record must align edge_idx, haufe_slope, and haufe_corr."
            )

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


# ---------------------------------------------------------------------
# Layer 3. CPM 交叉验证主流程
# ---------------------------------------------------------------------


def cpm_nested_cv_with_haufe(
    X,
    y,
    Covariates,
    param_grid=None,
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
    使用固定参数或 nested CV 参数优选完成 CPM 预测，并记录 selected-space Haufe。

    本函数是 CPM_new 当前的统一 CV 主流程。它同时支持两种使用方式：

    1. fixed-parameters 模式：当 ``param_grid=None``、``param_grid={}``，或
       ``param_grid`` 展开后只有一个候选组合时，不运行 inner CV，直接使用固定
       模型参数完成 outer CV；
    2. nested-CV 模式：当 ``param_grid`` 展开后包含多个候选组合时，才在每个
       outer training set 内运行 inner CV，选择当前 outer fold 的最佳模型参数。

    无论使用哪种模式，outer CV 都是最终评估层。每个 outer fold 都会在训练折内
    重新执行 CPM 选边、特征构建、模型拟合和测试折预测，并同步保存
    edge coefficient record。高维 CPM 模型还会计算 selected-space Haufe record。

    需要注意的是，本函数只优选模型层面的超参数，例如 ``ridge_alpha``、
    ``lasso_alpha``、``elasticnet_alpha``、``l1_ratio``、``pls_components`` 或
    ``pca_components``。CPM 选边规则，例如 ``p_thresh``、``top_k`` 和
    ``top_k_mode``，在整个 CV 中保持固定；但具体选中的边仍会在每个 outer
    训练折内重新估计；nested-CV 模式下，inner 训练折也会独立重新选边。

    ``model_type='linear'`` 也可以运行。此时模型是标准 CPM 的
    ``pos_strength`` / ``neg_strength`` 线性回归，没有 edge-level Haufe 映射；
    因此每个 fold 的 ``haufe_record['valid_haufe']`` 会是 ``False``，原因记录为
    ``"linear_strength_features"``。

    Parameters
    ----------
    X : array-like, shape (n_sub, n_edges)
        被试乘连接边的特征矩阵。每一列通常是一条 FC edge。
    y : array-like, shape (n_sub,)
        连续预测目标。模型实际拟合和评估的是训练折内去除协变量后的 y 残差。
    Covariates : array-like or None, shape (n_sub, n_Covariates)
        需要控制的协变量，不包含截距；传入 ``None`` 表示只控制截距。
    param_grid : dict, list of dict, or None, default=None
        待优选的模型超参数范围。``None`` 或空 dict 表示不调参，直接使用固定
        基础模型参数。非空 dict 会展开为笛卡尔积；list of dict 表示用户已经
        手动指定候选组合。若展开后只有一个候选组合，也会按 fixed-parameters
        模式处理，不运行 inner CV。允许的键取决于 ``model_type``：

        - ``'ridge'``：``ridge_alpha``；
        - ``'lasso'``：``lasso_alpha``；
        - ``'elasticnet'``：``elasticnet_alpha``、``l1_ratio``；
        - ``'pls'``：``pls_components``；
        - ``'pcr'``：``pca_components``；
        - ``'linear'``：无可调模型超参数，通常传入空 dict。

    n_splits : int, default=10
        外层 K 折交叉验证的折数，仅在 ``cv_type='kfold'`` 时使用。
    n_repeats : int, default=50
        外层 K 折随机划分的重复次数。LOOCV 无随机划分，因此会固定为 1。
    random_state : int, default=0
        外层 CV、内层 CV 以及随机模型使用的基础随机种子。
    cv_type : {'kfold', 'loocv'}, default='kfold'
        外层交叉验证方案。
    inner_n_splits : int, default=5
        内层 K 折 CV 的折数，仅在 ``inner_cv_type='kfold'`` 时使用。
    inner_cv_type : {'kfold', 'loocv'}, default='kfold'
        内层交叉验证方案。
    scoring : {'pearson_r', 'r2', 'neg_rmse', 'neg_mae'}, default='pearson_r'
        内层 CV 的调参指标。仅 nested-CV 模式使用；fixed-parameters 模式会忽略
        该参数。所有指标均按“越大越好”选择最佳参数。
    selection_method : {'pvalue', 'topk'}, default='pvalue'
        训练折内的边选择方法，具体规则由 ``select_edges()`` 实现。
    p_thresh : float, default=0.01
        p 值选边阈值，仅在 ``selection_method='pvalue'`` 时使用。
    top_k : int or None, default=None
        Top-K 模式的选边数。
    top_k_mode : {'total', 'per_sign', 'positive', 'negative'}, default='total'
        Top-K 选边范围：全部边、正负边分别选择、只选正边或只选负边。
    max_edges_total : int or None, default=None
        ``select_edges()`` 返回的最大总边数。
    min_edges_total : int, default=0
        启用 fallback 时希望保留的最少总边数。
    fallback_if_too_few : bool, default=False
        初始选边不足时，是否回退为按 p 值选择 ``min_edges_total`` 条边。
    ridge_alpha, lasso_alpha, elasticnet_alpha : float
        当前模型的基础正则参数；若对应参数不在 ``param_grid`` 中，会保留这些
        基础值。
    l1_ratio : float, default=0.5
        ElasticNet 的 L1/L2 混合比例，可在 ``param_grid`` 中覆盖。
    model_type : {'linear', 'ridge', 'lasso', 'elasticnet', 'pls', 'pcr'}, default='ridge'
        预测模型类型。``linear`` 使用标准 CPM strength；其他模型使用入选 edge
        作为高维特征。
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
        每个 outer fold 的选边、特征名称和模型系数记录。该记录来自最终 outer
        training fold 的重训模型，而不是 inner CV 中的临时模型。
    selected_param_records : list of dict
        每个 outer fold 的参数记录。``tuning_mode`` 标记该 fold 使用
        ``"fixed"`` 还是 ``"nested"``。fixed-parameters 模式下
        ``inner_score=np.nan``、``inner_n_splits_actual=0``、
        ``candidate_scores=[]``；nested-CV 模式下则记录最佳 inner score 和所有
        候选参数的 inner score。
    haufe_results : dict
        selected-space Haufe 汇总结果，由 ``_summarize_cpm_haufe_records()`` 生成。
        对 ``model_type='linear'``，``n_valid_folds`` 通常为 0，因为标准 CPM 的
        strength 特征不能直接映射回单条 edge。

    Raises
    ------
    ValueError
        输入形状、外层/内层 CV 参数、模型参数、``param_grid`` 或 fold 内残差自由度
        无效时抛出。
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
    random_state = _validate_integer(random_state, "random_state")
    standardize_features = _validate_boolean(
        standardize_features,
        "standardize_features",
    )
    verbose = _validate_boolean(verbose, "verbose")

    # ---------- 2. 验证 outer CV 设置 ----------
    cv_type, n_repeats_actual, n_splits_actual = _validate_cv_config(
        cv_type=cv_type,
        n_splits=n_splits,
        n_repeats=n_repeats,
        n_sub=n_sub,
    )
    if verbose and cv_type == "loocv" and n_repeats != 1:
        print("cv_type='loocv': n_repeats is ignored and set to 1.")

    # ---------- 3. 决定使用 fixed-parameters 还是 nested-CV 调参 ----------
    # 没有参数网格时，直接使用基础模型参数；这相当于固定参数 CV。
    # 若用户给了非空 param_grid，则先展开候选组合；只有候选数超过 1 时才值得
    # 运行 inner CV。单一候选组合没有“选择”问题，按固定参数处理即可。
    if param_grid is None or (isinstance(param_grid, dict) and len(param_grid) == 0):
        param_candidates = [{}]
        candidate_model_params = [base_model_params.copy()]
    else:
        param_candidates, candidate_model_params = _expand_param_grid(
            param_grid,
            model_type,
            base_model_params,
        )

    if len(candidate_model_params) == 1:
        tuning_mode = "fixed"
        scoring_used = None
    else:
        tuning_mode = "nested"
        scoring_used = str(scoring).lower()
        if scoring_used not in {"pearson_r", "r2", "neg_rmse", "neg_mae"}:
            raise ValueError(
                "scoring must be 'pearson_r', 'r2', 'neg_rmse', or 'neg_mae'."
            )

    # ---------- 4. 固定 CPM 选边设置 ----------
    selection_params = {
        "selection_method": selection_method,
        "p_thresh": p_thresh,
        "top_k": top_k,
        "top_k_mode": top_k_mode,
        "max_edges_total": max_edges_total,
        "min_edges_total": min_edges_total,
        "fallback_if_too_few": fallback_if_too_few,
    }

    # ---------- 5. 初始化 outer CV 输出容器 ----------
    y_pred_mat = np.zeros((n_repeats_actual, n_sub), dtype=float)
    y_resid_true_mat = np.zeros((n_repeats_actual, n_sub), dtype=float)
    edge_records = []
    selected_param_records = []
    haufe_records = []

    for rep in range(n_repeats_actual):
        if verbose:
            print(f"Outer repeat {rep + 1}/{n_repeats_actual}")

        # 当前 repeat 的 outer CV 切分。outer test fold 只用于最终预测评估，
        # 不参与 inner CV 的超参数选择。
        cv_iter = _create_cv_iterator(
            cv_type,
            n_splits_actual,
            X,
            random_state=random_state + rep,
        )
        y_pred = np.zeros(n_sub, dtype=float)
        y_resid_true = np.zeros(n_sub, dtype=float)

        for fold_idx, (train_idx, test_idx) in enumerate(cv_iter):
            # ---------- 6. 确定当前 outer fold 使用的模型参数 ----------
            if tuning_mode == "nested":
                # inner CV 只在 outer training set 内选择参数，不接触 outer test。
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
                    scoring=scoring_used,
                )
            else:
                # fixed-parameters 模式不运行 inner CV；所有 outer folds 使用同一套
                # 已验证参数。若 param_grid 只有一个候选，则使用该候选。
                best_model_params = candidate_model_params[0].copy()
                best_score = np.nan
                candidate_scores = []
                inner_n_splits_actual = 0

            # ---------- 7. 用最终参数重新训练 outer fold，并预测 outer test ----------
            prepared = prepare_cpm_fold_features(
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

            fold_result = fit_cpm_fold_model_with_haufe(
                repeat_idx=rep,
                fold_idx=fold_idx,
                prepared_fold=prepared,
                model_type=model_type,
                model_params=best_model_params,
                random_state=random_state + rep,
            )

            y_resid_true[test_idx] = prepared["y_resid_test"]
            y_pred[test_idx] = fold_result["y_pred_test"]
            edge_records.append(fold_result["edge_record"])
            haufe_records.append(fold_result["haufe_record"])

            # ---------- 8. 保存当前 outer fold 的参数记录 ----------
            selected_param_records.append(
                {
                    "repeat": rep,
                    "fold": fold_idx,
                    "tuning_mode": tuning_mode,
                    "best_params": best_model_params.copy(),
                    "inner_score": best_score,
                    "scoring": scoring_used,
                    "inner_cv_type": (
                        str(inner_cv_type).lower()
                        if tuning_mode == "nested"
                        else None
                    ),
                    "inner_n_splits_actual": inner_n_splits_actual,
                    "candidate_scores": candidate_scores,
                }
            )

        y_pred_mat[rep] = y_pred
        y_resid_true_mat[rep] = y_resid_true

    # ---------- 9. 汇总预测、真实残差和 Haufe map ----------
    y_pred_mean = y_pred_mat.mean(axis=0)
    y_resid_true_mean = y_resid_true_mat.mean(axis=0)
    haufe_results = _summarize_cpm_haufe_records(haufe_records, n_edges=n_edges)

    return (
        y_pred_mean,
        y_resid_true_mean,
        y_pred_mat,
        y_resid_true_mat,
        edge_records,
        selected_param_records,
        haufe_results,
    )

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker

from .fcmat import fc_matrix_to_edges


def _load_func_with_sform_fallback(filepath):
    """加载功能像，并为缺少 sform 的图像补充可用的空间信息。

    该函数用于处理一种常见情况：功能像已有可用的 qform，但
    ``sform_code`` 为 0，导致部分下游程序提示图像缺少 sform。

    处理规则如下：

    - 如果 ``sform_code > 0``，直接返回原图；
    - 如果 ``sform_code == 0``、但 ``qform_code > 0``，新建一个内存中的
      NIfTI 图像，将原 qform 同时设置为新图像的 qform 和 sform；
    - 如果两个 code 都为 0，则报错，不对图像空间作猜测。

    该函数只补充 NIfTI header 中的 sform 信息，不执行图像配准或重采样，
    也不会把修改后的图像自动保存到磁盘。

    Parameters
    ----------
    filepath : str or os.PathLike
        NIfTI 功能像的文件路径。

    Returns
    -------
    nibabel.nifti1.Nifti1Image
        如果原图已有 sform，返回原图；否则返回用 qform 补充 sform 后的
        新图像对象。

    Raises
    ------
    ValueError
        当 ``sform_code`` 和 ``qform_code`` 都为 0 时抛出。

    Notes
    -----
    ``code > 0`` 只表示相应变换在 header 中被启用。本函数不判断该变换
    是否与图谱正确对齐，也不检查同时存在的 sform 和 qform 是否一致。

    新建图像时会将功能像数据载入内存，但不改变体素值。
    """
    img = nib.load(filepath)

    # coded=True 同时返回仿射矩阵和对应的空间 code。code=0 表示该 form
    # 未被 header 正式启用，即使相关矩阵字段中可能仍保存有数值。
    sform, sform_code = img.header.get_sform(coded=True)
    qform, qform_code = img.header.get_qform(coded=True)
    sform_code = int(sform_code)
    qform_code = int(qform_code)

    # NiBabel 通常优先使用已启用的 sform。此时保持原图不变，避免对已经完整
    # 保存空间信息的预处理结果做不必要的重建。
    if sform_code > 0:
        return img

    if qform_code > 0:
        print(
            "   Warning: sform_code=0; copying the enabled qform to sform "
            f"(qform_code={qform_code})."
        )

        # 原文件已有一套被启用的 qform，因此这里只让 sform 与 qform 使用同一
        # 体素到世界坐标映射。该操作不改变体素数组，也不进行空间重采样。
        fixed_header = img.header.copy()
        fixed_img = nib.Nifti1Image(
            np.asanyarray(img.dataobj),
            affine=qform,
            header=fixed_header,
        )

        # 显式设置两套 form 及其 code，避免新建图像时 header 的自动处理导致
        # qform 或 sform 的 code 与原始有效 qform 不一致。
        fixed_img.set_qform(qform, code=qform_code)
        fixed_img.set_sform(qform, code=qform_code)
        return fixed_img

    # 两套空间变换都未启用时，图像数组本身不足以确定其真实物理位置。此时
    # 报错比猜测一个模板 affine 更安全，后者可能产生无报错但解剖错位的 ROI。
    raise ValueError(
        "sform_code and qform_code are both 0; the image's true spatial "
        "location cannot be confirmed. Check the preprocessing output instead "
        "of assigning an atlas or template affine."
    )


def extract_roi_timeseries(data_dir, labels_img, output_dir, suffix="timeseries"):
    """使用离散标签图谱批量提取 ROI 时间序列。

    图谱中的每个非零标签值定义一个 ROI。函数将每幅功能像重采样到标签图谱
    的网格，计算每个 ROI 内所有体素的平均信号，并将结果保存为
    ``ROI × timepoint`` 数组。

    Parameters
    ----------
    data_dir : str or os.PathLike
        保存预处理后功能像的目录。函数读取该目录下所有 ``.nii`` 和
        ``.nii.gz`` 文件，并按文件名排序处理。
    labels_img : str or os.PathLike
        离散标签图谱的 NIfTI 文件路径。图谱中的 0 视为背景，每个非零标签值
        视为一个 ROI。
    output_dir : str or os.PathLike
        ROI 时间序列的保存目录。每个功能像对应一个
        ``<功能像文件名>_<suffix>.npy`` 文件。
    suffix : str, default="timeseries"
        添加在功能像文件名之后的输出标识。默认生成
        ``<功能像文件名>_timeseries.npy``。

    Returns
    -------
    successful_files : list of str
        成功完成提取并保存的功能像文件名。
    failed_files : list of tuple
        失败记录；每项为 ``(功能像文件名, 错误信息)``。

    Notes
    -----
    ``NiftiLabelsMasker.fit_transform()`` 返回 ``timepoint × ROI``，保存前会
    转置为 ``ROI × timepoint`` 并转换为 ``float32``。

    本函数不执行功能像到标准空间的配准。它假设输入功能像已经完成所需的
    预处理，并且与标签图谱处于同一解剖空间；
    ``_load_func_with_sform_fallback()`` 只在缺少 sform 时用已启用的 qform
    补充空间 header。
    """
    data_dir = Path(data_dir)
    labels_img = Path(labels_img)
    output_dir = Path(output_dir)

    if not isinstance(suffix, str) or not suffix.strip():
        raise ValueError("suffix must be a non-empty string.")
    suffix = suffix.strip()

    if not data_dir.is_dir():
        raise NotADirectoryError(f"Functional image directory not found: {data_dir}")
    if not labels_img.is_file():
        raise FileNotFoundError(f"Label atlas not found: {labels_img}")

    # 图谱中的非零整数是实际 ROI 标签；0 是背景，不计入预期 ROI 数量。
    atlas_loaded = nib.load(labels_img)
    atlas_values = np.unique(np.asanyarray(atlas_loaded.dataobj))
    atlas_values = atlas_values[atlas_values != 0]
    expected_n_rois = len(atlas_values)

    # resampling_target="labels" 保持标签图谱网格不变，将每名被试的功能像
    # 重采样到图谱网格。这样可避免把图谱重采样到较粗功能像网格时丢失小 ROI。
    masker = NiftiLabelsMasker(
        labels_img=atlas_loaded,
        background_label=0,
        resampling_target="labels",
        strategy="mean",
        standardize=False,
        detrend=False,
        reports=False,
        verbose=1,
    )

    functional_files = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and (path.name.endswith(".nii") or path.name.endswith(".nii.gz"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Number of non-background atlas ROIs: {expected_n_rois}")
    print(f"Number of functional files: {len(functional_files)}")
    print(f"Output directory: {output_dir}")

    successful_files = []
    failed_files = []

    for filepath in functional_files:
        print(f"\nProcessing: {filepath.name}")

        try:
            func_img = _load_func_with_sform_fallback(filepath)

            voxel_and_time_size = func_img.header.get_zooms()
            print(f"   Functional shape: {func_img.shape}")
            print(f"   Voxel size: {voxel_and_time_size[:3]}")
            print(f"   Orientation: {nib.aff2axcodes(func_img.affine)}")
            if func_img.ndim >= 4:
                _, time_unit = func_img.header.get_xyzt_units()
                time_unit = time_unit or "unknown time unit"
                print(f"   TR: {voxel_and_time_size[3]} {time_unit}")
            else:
                print("   TR: N/A (image is not 4D)")

            # 每名被试直接调用 fit_transform。masker 会将当前功能像重采样到
            # 标签图谱网格，然后返回 timepoint × ROI 的平均时间序列。
            time_series = masker.fit_transform(func_img)
            n_timepoints, n_rois = time_series.shape
            print(f"   Extracted {n_rois} ROIs, {n_timepoints} timepoints")

            # ROI 数量减少通常表示图谱重采样或空间对齐出现问题，因此不保存
            # 不完整结果，避免后续被试之间的 ROI 列发生错位。
            if n_rois != expected_n_rois:
                raise RuntimeError(
                    f"Extracted {n_rois} ROIs; expected {expected_n_rois}."
                )
            if not np.all(np.isfinite(time_series)):
                raise RuntimeError("Extracted time series contains NaN or Inf.")

            if filepath.name.endswith(".nii.gz"):
                subject_name = filepath.name[:-7]
            else:
                subject_name = filepath.stem

            save_path = output_dir / f"{subject_name}_{suffix}.npy"
            np.save(save_path, time_series.T.astype(np.float32))
            print(f"   Saved to: {save_path}")
            successful_files.append(filepath.name)

        except Exception as error:
            print(f"   Error: {error}")
            failed_files.append((filepath.name, str(error)))

    print("\nROI time-series extraction summary:")
    print(f"  Successful: {len(successful_files)}")
    print(f"  Failed: {len(failed_files)}")

    if failed_files:
        print("Failed files:")
        for filename, error in failed_files:
            print(f"  {filename}: {error}")

    return successful_files, failed_files


def align_subjects_with_timeseries(
    df,
    timeseries_dir,
    func_nifti_dir=None,
    subject_id_col="ID",
    timeseries_suffix="timeseries",
    numeric_columns=(),
    categorical_columns=(),
):
    """整理被试信息表，并与 ROI 时间序列及可选功能像进行匹配。

    函数按照两个阶段筛选被试：

    1. 检查被试 ID 以及指定的数值变量和类别变量。因 ID 缺失、
       数值变量为 ``NaN``/正负无穷，或类别变量缺失而排除的被试，
       记录在 ``excluded_invalid_values`` 中。
    2. 对第一阶段保留的被试匹配 ROI 时间序列。如果提供
       ``func_nifti_dir``，同时匹配功能像。因所需文件缺失而排除的
       被试，记录在 ``excluded_inputs`` 中。

    ``df_profile`` 只包含通过上述两阶段检查的被试。三个返回表均保留
    输入 ``df`` 的列；文件路径和文件存在状态只用于函数内部检查，
    不写入返回结果。函数不会修改传入的 ``df``。

    Parameters
    ----------
    df : pandas.DataFrame
        被试信息表。必须包含 ``subject_id_col`` 指定的 ID 列，以及
        ``numeric_columns`` 和 ``categorical_columns`` 中指定的列。
        返回结果保持该表中的被试顺序。
    timeseries_dir : str or os.PathLike
        ROI 时间序列所在目录。文件名应为
        ``<subject_id>_<timeseries_suffix>.npy``。
    func_nifti_dir : str or os.PathLike or None, default=None
        可选的功能像目录。提供目录时，文件名应为 ``<subject_id>.nii``
        或 ``<subject_id>.nii.gz``；只有同时找到 ROI 时间序列和功能像的
        被试才会保留。设为 ``None`` 时，不检查功能像。
    subject_id_col : str, default="ID"
        ``df`` 中的被试 ID 列名。
    timeseries_suffix : str, default="timeseries"
        ROI 时间序列的文件名后缀。例如，取默认值时，被试
        ``S01`` 对应的文件名为 ``S01_timeseries.npy``。
    numeric_columns : sequence of str, default=()
        后续分析需要的数值列。函数先尝试将这些列转换为数值；
        无法转换的值、``NaN`` 和正负无穷值均视为无效。
    categorical_columns : sequence of str, default=()
        后续分析需要的类别列。缺失值视为无效；函数不修改类别的
        原始编码或数据类型。

    Returns
    -------
    df_profile : pandas.DataFrame
        可直接进入后续分析的被试表：指定变量均有效，且所需文件
        均已匹配。只包含输入 ``df`` 的列。
    excluded_inputs : pandas.DataFrame
        被试变量有效，但缺少所需文件的记录。新增的
        ``exclusion_reason`` 列说明缺少 ROI 时间序列、功能像，
        或两者均缺失。
    excluded_invalid_values : pandas.DataFrame
        因 ID 或指定变量无效而排除的被试记录。新增的
        ``invalid_columns`` 列列出该被试的所有无效列。

    Raises
    ------
    KeyError
        ``df`` 缺少 ID 列，或缺少指定的数值列或类别列。
    ValueError
        ``df`` 不是 DataFrame，数值列与类别列重叠，时间序列
        后缀为空，被试 ID 重复，或同一 ID 对应多个功能像。
    NotADirectoryError
        ROI 时间序列目录不存在，或提供的功能像目录不存在。

    Notes
    -----
    函数只检查文件名和文件是否存在，不读取 NIfTI 或 NPY 的内容。
    因此，它不检查功能像的空间信息，也不检查时间序列的形状、
    数据类型或数组内部的非有限值。

    如果同一被试同时存在 ``<ID>.nii`` 和 ``<ID>.nii.gz``，函数
    无法判断应该使用哪一个文件，因此会报错，而不会自动选择。
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")

    timeseries_dir = Path(timeseries_dir)
    if not timeseries_dir.is_dir():
        raise NotADirectoryError(f"ROI time-series directory not found: {timeseries_dir}")
    if not isinstance(timeseries_suffix, str) or not timeseries_suffix.strip():
        raise ValueError("timeseries_suffix must be a non-empty string.")
    timeseries_suffix = timeseries_suffix.strip()

    numeric_columns = list(numeric_columns)
    categorical_columns = list(categorical_columns)
    overlapping_columns = sorted(set(numeric_columns) & set(categorical_columns))
    if overlapping_columns:
        raise ValueError(
            "Columns cannot be both numeric and categorical: "
            f"{overlapping_columns}"
        )

    # 复制输入表：后续的ID和数值类型转换只作用于函数内部，
    # 不改动调用者手中的原始DataFrame。
    df_checked = df.reset_index(drop=True).copy()
    original_columns = list(df_checked.columns)
    if subject_id_col not in df_checked.columns:
        raise KeyError(f"Column not found in df: {subject_id_col}")

    for column in [*numeric_columns, *categorical_columns]:
        if column not in df_checked.columns:
            raise KeyError(f"Column not found in df: {column}")

    # 缺失ID的行无法与时间序列或功能像匹配。先单独保留这些记录，
    # 便于在运行摘要中打印原始行号和其他被试信息。
    missing_subject_id = df_checked[subject_id_col].isna()
    missing_subject_records = df_checked.loc[missing_subject_id].copy()
    missing_subject_records.insert(
        0,
        "df_row",
        np.flatnonzero(missing_subject_id),
    )

    # ID缺失时无法通过被试编号定位记录，因此在继续筛选前
    # 立即打印整行信息。df_row是输入表重置索引后的行号，从0开始。
    if missing_subject_id.any():
        print("Subjects with missing IDs:")
        print(missing_subject_records.to_string(index=False))

    df_checked.loc[~missing_subject_id, subject_id_col] = (
        df_checked.loc[~missing_subject_id, subject_id_col].astype(str)
    )

    # 文件名是字符串，因此先把非缺失ID统一转为字符串，再检查重复。
    # 重复ID会使同一文件对应多行被试资料，函数不猜测应保留哪一行。
    duplicated_subjects = (
        ~missing_subject_id
        & df_checked[subject_id_col].duplicated(keep=False)
    )
    if duplicated_subjects.any():
        duplicated_ids = df_checked.loc[duplicated_subjects, subject_id_col].tolist()
        raise ValueError(f"Duplicated IDs in df: {duplicated_ids}")

    for column in numeric_columns:
        df_checked[column] = pd.to_numeric(df_checked[column], errors="coerce")

    # 建立一张与df_checked行对齐的布尔表：行表示被试，列表示
    # 待检查变量，True表示该被试在当前变量上存在无效值。
    invalid_values = pd.DataFrame(index=df_checked.index)
    invalid_values[subject_id_col] = missing_subject_id

    # 数值列之前已转换为数值类型。np.isfinite对普通有限数返回True，
    # 对NaN和正负无穷返回False；取反后，True统一表示“无效”。
    for column in numeric_columns:
        invalid_values[column] = ~np.isfinite(
            df_checked[column].to_numpy(dtype=float)
        )

    # 类别列不做数值转换，只将缺失值标记为无效。
    for column in categorical_columns:
        invalid_values[column] = df_checked[column].isna()

    # 横向检查每一名被试：只要任意待检查变量为True，该被试就需要因变量
    # 无效而排除。这样无需手动逐列合并多个布尔数组。
    has_invalid_values = invalid_values.any(axis=1)

    # 保存变量无效的被试，并将该行所有为True的列名合并为排除原因。
    # 例如Age和group均无效时，invalid_columns为"Age; group"。
    excluded_invalid_values = df_checked.loc[
        has_invalid_values,
        original_columns,
    ].copy()
    if has_invalid_values.any():
        excluded_invalid_values["invalid_columns"] = invalid_values.loc[
            has_invalid_values
        ].apply(
            lambda row: "; ".join(row.index[row]),
            axis=1,
        )
    else:
        # 没有被试因变量无效而排除时，仍保留空的invalid_columns列，
        # 使返回表的列结构在不同数据集中保持一致。
        excluded_invalid_values["invalid_columns"] = pd.Series(dtype=str)

    # 只有ID和指定变量均有效的被试才继续检查影像文件。由此可以将
    # “DataFrame中的值无效”和“所需文件缺失”分别记录在两个排除表中。
    df_file_check = df_checked.loc[
        ~has_invalid_values,
        original_columns,
    ].copy()

    # 为信息完整的被试检查ROI时间序列文件。
    # map会依次读取df_file_check中的每个ID，按
    # <subject_id>_<timeseries_suffix>.npy的规则生成完整路径。
    timeseries_paths = df_file_check[subject_id_col].map(
        lambda subject_id: timeseries_dir
        / f"{subject_id}_{timeseries_suffix}.npy"
    )

    # Path.is_file逐个检查路径是否存在且是普通文件。转换后的布尔数组
    # 与df_file_check的行顺序完全一致：True表示该被试有ROI时间序列。
    has_roi_timeseries = timeseries_paths.map(Path.is_file).to_numpy(dtype=bool)

    # complete_inputs用于累积“所有要求的文件都存在”这一最终条件。
    # 此时还没有加入功能像条件，所以它暂时等于has_roi_timeseries。
    # 使用.copy()是为了后续更新complete_inputs时，保持has_roi_timeseries
    # 不变，以便最后准确记录每名被试究竟缺少哪一类文件。
    complete_inputs = has_roi_timeseries.copy()

    # 功能像匹配是可选步骤。func_nifti_dir为None时，不把功能像纳入
    # 被试筛选；has_func_nifti也保持为None，表示“未检查”。
    has_func_nifti = None
    n_func_files = None
    if func_nifti_dir is not None:
        func_dir = Path(func_nifti_dir)
        if not func_dir.is_dir():
            raise NotADirectoryError(
                f"Functional NIfTI directory not found: {func_dir}"
            )

        # 扫描目录并建立“被试ID -> 功能像路径”的对应关系。
        # 这个字典同时用于检查重复文件和判断被试是否有功能像。
        func_path_by_id = {}
        for filepath in sorted(func_dir.iterdir()):
            # 跳过子目录和隐藏文件。
            if not filepath.is_file() or filepath.name.startswith("."):
                continue

            # 去掉.nii或.nii.gz后缀，剩余的文件主名作为被试ID。
            if filepath.name.endswith(".nii.gz"):
                subject_id = filepath.name[:-7]
            elif filepath.name.endswith(".nii"):
                subject_id = filepath.name[:-4]
            else:
                continue

            # 同一ID只能对应一个功能像。若同时存在.nii和.nii.gz等重复文件，
            # 函数无法判断应使用哪一个，因此报错并列出两个路径。
            if subject_id in func_path_by_id:
                raise ValueError(
                    f"Multiple functional NIfTI files matched ID {subject_id!r}:\n"
                    f"  {func_path_by_id[subject_id]}\n"
                    f"  {filepath}"
                )
            func_path_by_id[subject_id] = filepath

        # 按ID判断每名被试是否有功能像。与ROI时间序列的检查结果
        # 合并后，complete_inputs表示两类文件都已找到。
        has_func_nifti = (
            df_file_check[subject_id_col]
            .isin(func_path_by_id)
            .to_numpy(dtype=bool)
        )
        complete_inputs &= has_func_nifti
        n_func_files = len(func_path_by_id)

    df_profile = (
        df_file_check.loc[complete_inputs, original_columns]
        .reset_index(drop=True)
        .copy()
    )
    excluded_inputs = (
        df_file_check.loc[~complete_inputs, original_columns]
        .reset_index(drop=True)
        .copy()
    )

    # 为因文件不完整而排除的被试标注具体原因。如果两类文件都缺失，
    # exclusion_reason会同时保留两个原因。
    if not excluded_inputs.empty:
        excluded_positions = np.flatnonzero(~complete_inputs)
        exclusion_reasons = []
        for position in excluded_positions:
            reasons = []
            if not has_roi_timeseries[position]:
                reasons.append("missing_roi_timeseries")
            if has_func_nifti is not None and not has_func_nifti[position]:
                reasons.append("missing_func_nifti")
            exclusion_reasons.append("; ".join(reasons))
        excluded_inputs["exclusion_reason"] = exclusion_reasons

    excluded_invalid_values = excluded_invalid_values.reset_index(drop=True)

    print("Subject input preparation summary:")
    print(f"  Input subjects: {len(df)}")
    print(f"  Subjects excluded for invalid df values: {len(excluded_invalid_values)}")
    if n_func_files is not None:
        print(f"  Functional NIfTI files: {n_func_files}")
    print(f"  Matched subjects: {len(df_profile)}")
    print(f"  Subjects excluded for missing files: {len(excluded_inputs)}")

    if not excluded_invalid_values.empty:
        print("Invalid DataFrame value details:")
        print(
            excluded_invalid_values[
                [subject_id_col, "invalid_columns"]
            ].to_string(index=False)
        )

    if not excluded_inputs.empty:
        print("Excluded input details:")
        print(
            excluded_inputs[
                [subject_id_col, "exclusion_reason"]
            ].to_string(index=False)
        )

    return df_profile, excluded_inputs, excluded_invalid_values


def timeSeriesGp(dataDir, masker):

# This batch function extracts ROI-level time series from each .nii file in
# dataDir. Each .nii file is treated as one subject. The returned 3D array has
# shape (subjects, ROIs, time points).
#
# Dimension order:
#   - subjects: determined by sorted file names in dataDir;
#   - ROIs: determined by the atlas labels used by NiftiLabelsMasker;
#   - time points: determined by the temporal dimension of each 4D .nii file.
#
# NiftiLabelsMasker.fit_transform() returns an array with shape
# (time points, ROIs). The ".T" below transposes it to (ROIs, time points), so
# each row represents one ROI. This orientation is required by fcMatrixGp(),
# because np.corrcoef(e) treats rows as variables.


    '''
    Parameters
    ----------
    dataDir : str
        The absolute path for the directory of .nii files.
    masker : str
        The absolult path for the atlas (one .nii file).

    Returns
    -------
    timeSeriesGp: 3d-array
       Array with shape (subjects, ROIs, time points). Each element along the
       first dimension corresponds to one subject/file. Within each subject,
       each row corresponds to one ROI and each column corresponds to one time
       point. The subject order follows sorted file names in dataDir; the ROI
       order follows the atlas labels used by NiftiLabelsMasker.

    '''

    masker = NiftiLabelsMasker(labels_img=masker)

    files = os.listdir(dataDir) 
    files = np.sort(files)  # To collect and sort the absolute path for each .nii file.
    timeSeriesGp = []
    for file in files:
        filepath = os.path.join(dataDir, file)
        print(file) # To print the "file" for the convenience of checking data.
        timeSeriesGp.append(masker.fit_transform(filepath).T) # Don't forget do the transformation!
        
    return np.array(timeSeriesGp)



def fcMatrixGp(timeSeriesGp, fisherZ=True): 

# This function is a batch function that constructs the FC-matrix for each element of "timeSeriesGp", which 
#   was further transformed into Fisher's Z-value by default (or determined by the parameter of "fisherZ").


    '''
    Parameters
    ----------
    timeSeriesGp : 3d-array
        Each element refers to one subject, where each row contains the time series for one ROI.
    fisherZ : bool
        Determine whether to transform into Fisher's Z-value.

    Returns
    -------
    fcMatrixGp: 3d-array
        Each element refers to the FC-matrix for one subject.

    '''

    fcMatrixGp = []
    for e in timeSeriesGp:
        fcMatrixGp.append(np.corrcoef(e))
    if fisherZ:
        fcMatrixGp = np.arctanh(fcMatrixGp)

    return np.array(fcMatrixGp)


def compute_fc_from_timeseries_dirs(
    df,
    roi_timeseries_sources,
    subject_id_col="ID",
    file_template="sub-{subject_id}_{run_label}.npy",
    drop_initial_timepoints=0,
    expected_n_timepoints=None,
    fisher_z=True,
    return_edges=True,
):
    """
    从 ROI time series 文件计算 FC，支持多个 time series 来源、多 run、ROI subset，
    并可选择返回 FC edge vector 或完整 FC matrix。这里的“多个 time series
    来源”通常指不同 atlas/parcellation，例如 cortical Schaefer 和
    subcortical HO。

    函数内部的基本流程分为三层：

    1. 被试层
       1.1 按 ``df`` 行顺序逐被试处理；
       1.2 成功完成 FC 计算的被试进入 ``df_fc``，且 ``df_fc`` 与返回的 FC
           数据严格按行对齐。

    2. Time series 层
       2.1 对当前被试，逐 time series 来源、逐 run 读取文件。这里的来源
           通常是不同 atlas/parcellation 的 ROI time series 目录；
       2.2 删除每个 run 开头指定数量的 time points；
       2.3 同一来源内，将一个或多个 run 沿时间维拼接；
       2.4 若该来源指定 ``roi_index``，仅保留这些 ROI；
       2.5 将当前被试所有来源的 time series 沿 ROI 维拼接。例如
           Schaefer200 ``(200, 740)`` 与 HO14 ``(14, 740)`` 拼接后为
           ``(214, 740)``。

    3. FC 层
       3.1 对最终 ROI x timepoints 矩阵计算 Pearson FC；
       3.2 根据 ``fisher_z`` 决定是否做 Fisher-z 转换；
       3.3 根据 ``return_edges`` 决定返回 edge vector 还是完整 FC matrix。

    Parameters
    ----------
    df : pandas.DataFrame
        被试信息表。函数会按照 ``df`` 的当前行顺序逐被试读取 time series。
        返回的 ``df_fc`` 与 ``FC_data`` 严格按行对齐。
    roi_timeseries_sources : list of dict
        每个 dict 描述一个 ROI time series 来源。这里的“来源”通常指一个
        atlas/parcellation 对应的一组 ROI time series 文件，例如 Schaefer200
        cortical ROIs 或 HO14 subcortical ROIs。至少需要包含：

        - ``"data_dir"``：保存该来源 ROI time series ``.npy`` 文件的目录；

        可选字段包括：

        - ``"name"``：time series 来源名称，只用于排除记录，便于定位问题文件；
        - ``"run_labels"``：该来源需要读取的 run 名称。若省略，默认读取
          ``("func1", "func2")``；若为 ``None``，表示该来源每个被试只有一个
          time series 文件；
        - ``"roi_index"``：需要保留的 ROI 下标。若为 ``None`` 或省略，则保留
          该来源中的全部 ROI。

        例如 Schaefer200 + HO14 可以写成：

        ``[
            {"name": "Schaefer200", "data_dir": SCHAEFER_DIR, "run_labels": ("func1", "func2")},
            {"name": "HO14", "data_dir": HO_DIR, "run_labels": ("func1", "func2"), "roi_index": HO_RAW_INDEX},
        ]``

    subject_id_col : str, default="ID"
        ``df`` 中保存被试 ID 的列名。
    file_template : str, default="sub-{subject_id}_{run_label}.npy"
        由被试 ID 和 run 名称生成文件名的模板。模板可使用：

        - ``{subject_id}``：当前被试 ID；
        - ``{run_label}``：当前 run 名称；

    drop_initial_timepoints : int, default=0
        每个 run 开头需要删除的时间点数。例如 ``drop_initial_timepoints=5`` 表示
        每个 run 都先删除最初 5 个 time points，再进行 run 拼接和 FC 计算。
        该参数常用于去掉 fMRI run 开头的 steady-state 不稳定时间点。
    expected_n_timepoints : int or None, default=None
        若不是 ``None``，则要求每个来源的所有 run 拼接后时间点数等于该值。
        不满足的被试会被跳过。
    fisher_z : bool, default=True
        是否对 FC correlation matrix 做 Fisher-z 转换。
    return_edges : bool, default=True
        是否把每个被试的 FC matrix 向量化为 edge vector。

        - ``True``：返回 shape 为 ``(n_subjects, n_edges)`` 的 FC edge matrix；
        - ``False``：返回 shape 为 ``(n_subjects, n_rois, n_rois)`` 的 FC matrix group。

    Returns
    -------
    df_fc : pandas.DataFrame
        成功计算 FC 的被试表，行顺序与第二个返回值完全一致。
    FC_data : ndarray
        若 ``return_edges=True``，返回 FC edge matrix；若 ``return_edges=False``，
        返回 FC matrix group。若 ``fisher_z=True``，这里保存 Fisher-z FC。

    Notes
    -----
    函数会直接打印数据准备摘要，包括输入被试数、成功被试数、排除被试数，
    以及被试排除原因和详细排除记录。摘要只用于运行时检查，不作为返回值增加
    后续代码负担。

    本函数假设每个 ``.npy`` time series 的形状是 ``(n_rois, n_timepoints)``。
    多个 run 会沿时间维 ``axis=1`` 拼接；多个来源，即不同 atlas/parcellation
    的 ROI time series，会沿 ROI 维 ``axis=0`` 拼接。

    Examples
    --------
    1) Schaefer200 + HO14，两个 run，返回 FC edges：

    ``df_fc, FC_edges = compute_fc_from_timeseries_dirs(
        df,
        roi_timeseries_sources=[
            {"name": "Schaefer200", "data_dir": SCHAEFER_DIR, "run_labels": ("func1", "func2")},
            {"name": "HO14", "data_dir": HO_DIR, "run_labels": ("func1", "func2"), "roi_index": HO_RAW_INDEX},
        ],
        subject_id_col="ID",
        file_template="sub-{subject_id}_{run_label}.npy",
        expected_n_timepoints=740,
        return_edges=True,
    )``

    2) 单个来源，只有一个 run，文件名不含 run label：

    ``df_fc, FC_edges = compute_fc_from_timeseries_dirs(
        df,
        roi_timeseries_sources=[{"name": "Schaefer400", "data_dir": SCHAEFER400_DIR, "run_labels": None}],
        file_template="sub-{subject_id}.npy",
        expected_n_timepoints=370,
    )``

    3) 返回完整 FC matrix group，而不是 edge vector：

    ``df_fc, FC_matrices = compute_fc_from_timeseries_dirs(
        df,
        roi_timeseries_sources=[{"name": "Schaefer200", "data_dir": SCHAEFER_DIR}],
        return_edges=False,
    )``
    """
    # 0. 输入检查。这里尽早拦截明显错误，避免主循环跑到深处才报出难懂的异常。
    if subject_id_col not in df.columns:
        raise KeyError(f"Column not found: {subject_id_col}")
    if len(roi_timeseries_sources) == 0:
        raise ValueError("roi_timeseries_sources must contain at least one source.")
    if drop_initial_timepoints < 0:
        raise ValueError("drop_initial_timepoints must be >= 0.")

    for i, source in enumerate(roi_timeseries_sources):
        if "data_dir" not in source:
            raise KeyError(f"roi_timeseries_sources[{i}] must contain 'data_dir'.")

    fc_list = []
    keep_row_idx = []
    excluded_subjects = []

    # 1.1 被试层：严格按照 df 当前行顺序逐被试处理。
    for row_idx in range(df.shape[0]):
        subject_id = str(df.iloc[row_idx][subject_id_col])
        source_ts_list = []
        # skip_subject 决定当前被试是否进入最终 df_fc / FC_data。
        # 只要该被试任一来源/任一 run 出现缺失文件、time series 形状错误或
        # 时间点数不符合要求，就设为 True；该被试随后不会进入 keep_row_idx，
        # 因而不会出现在 df_fc 中，也不会有对应的 FC_data 行。
        skip_subject = False
        exclusion = None

        for source in roi_timeseries_sources:
            run_ts_list = []
            # 每个来源通常对应一个 atlas/parcellation，例如 cortical Schaefer
            # 或 subcortical HO。不同来源可以有自己的 run 结构；run_labels=None
            # 表示该来源每个被试只有一个文件，例如 sub-001.npy。
            # 这里转换成 [None] 以复用同一套循环。
            source_run_labels = source.get("run_labels", ("func1", "func2"))
            if source_run_labels is None:
                run_labels_to_read = [None]
            else:
                run_labels_to_read = list(source_run_labels)
            if len(run_labels_to_read) == 0:
                raise ValueError("source run_labels must contain at least one run label.")

            # 2.1 Time series 层：对当前被试的当前来源，逐 run 读取文件。
            # “当前来源”通常就是某个 atlas/parcellation 的 time series 目录。
            for run_label in run_labels_to_read:
                # file_template 统一负责文件命名。多 run 时常用 {run_label}；
                # 单 run 时可设置 run_labels=None，并使用不含 {run_label} 的模板。
                file_name = file_template.format(
                    subject_id=subject_id,
                    run_label="" if run_label is None else run_label,
                )
                file_path = Path(source["data_dir"]) / file_name

                if not file_path.exists():
                    # 当前被试缺少任一必需文件时，不能得到完整 ROI time series，
                    # 因此排除整个被试。
                    exclusion = {
                        "subject_id": subject_id,
                        "reason": "missing",
                        "source": source.get("name", Path(source["data_dir"]).name),
                        "run_label": run_label,
                        "file": str(file_path),
                    }
                    skip_subject = True
                    break

                ts = np.load(file_path)
                if ts.ndim != 2:
                    # 每个 .npy 必须是 ROI x timepoints 的二维矩阵；否则无法
                    # 与其他 run/source 拼接，也无法计算 FC。
                    exclusion = {
                        "subject_id": subject_id,
                        "reason": "bad_timeseries_shape",
                        "source": source.get("name", Path(source["data_dir"]).name),
                        "run_label": run_label,
                        "file": str(file_path),
                        "shape": tuple(ts.shape),
                    }
                    skip_subject = True
                    break
                # 2.2 每个 run 分别删除开头指定数量的 time points，再进入拼接。
                if drop_initial_timepoints > 0:
                    ts = ts[:, drop_initial_timepoints:]
                run_ts_list.append(ts)

            if skip_subject:
                break

            # 2.3 同一来源内，将一个或多个 run 沿时间维拼接：
            #   (n_rois, t1), (n_rois, t2) -> (n_rois, t1 + t2)
            ts_source = np.concatenate(run_ts_list, axis=1)
            if expected_n_timepoints is not None and ts_source.shape[1] != expected_n_timepoints:
                # 若设置 expected_n_timepoints，则每个来源在 run 拼接后的时间点数
                # 必须一致；不一致通常表示 run 缺失、截断或前处理输出异常。
                exclusion = {
                    "subject_id": subject_id,
                    "reason": "bad_timepoints",
                    "source": source.get("name", Path(source["data_dir"]).name),
                    "observed_timepoints": int(ts_source.shape[1]),
                    "expected_timepoints": int(expected_n_timepoints),
                }
                skip_subject = True
                break

            # 2.4 若当前来源指定 roi_index，仅保留这些 ROI。
            roi_index = source.get("roi_index", None)
            if roi_index is not None:
                ts_source = ts_source[np.asarray(roi_index, dtype=int), :]
            source_ts_list.append(ts_source)

        if skip_subject:
            excluded_subjects.append(exclusion)
            continue

        # 2.5 将当前被试所有来源沿 ROI 维拼接，得到统一的
        # ROI x timepoints 矩阵。例如 Schaefer200 cortical ROIs 与 HO14
        # subcortical ROIs 可拼接为 (200, 740) + (14, 740) -> (214, 740)。
        ts_all = np.concatenate(source_ts_list, axis=0)

        # 3.1 np.corrcoef(ts_all) 把每一行当作一个变量，因此这里得到 ROI x ROI FC。
        fc = np.corrcoef(ts_all)

        # 相关系数非常接近 +/-1 时，arctanh 会得到 inf。先轻微截断，避免
        # Fisher-z 后产生无穷值。
        fc = np.clip(fc, -0.999999, 0.999999)
        # 3.2 根据 fisher_z 决定是否做 Fisher-z 转换。
        if fisher_z:
            fc = np.arctanh(fc)

        # 3.3 CPM/GMM 等模型通常使用 edge vector；可视化或 network-level 汇总有时
        # 更适合保留完整 FC matrix，因此这里用 return_edges 控制输出形式。
        if return_edges:
            fc_list.append(fc_matrix_to_edges(fc, k=1))
        else:
            fc_list.append(fc)
        keep_row_idx.append(row_idx)

    # 1.2/1.3 只有成功完成 FC 计算的被试会进入 df_fc；df_fc 与 FC_data
    # 严格按行对齐，即 df_fc.iloc[i] 对应 FC_data[i]。
    df_fc = df.iloc[keep_row_idx].reset_index(drop=True).copy()
    FC_data = np.asarray(fc_list, dtype=np.float32)

    # 4. 运行摘要。
    # 这部分不参与 FC 计算，只用于检查本次数据准备是否符合预期。
    # 因为 ROI 数、edge 数、fisher_z、return_edges 等信息都能从输入参数或
    # 返回数据直接判断，所以这里不重复输出；只报告真正由本次读取/筛选决定的
    # 被试保留情况。
    info = {
        "n_input_subjects": int(df.shape[0]),
        "n_kept_subjects": int(df_fc.shape[0]),
        "n_excluded_subjects": int(df.shape[0] - df_fc.shape[0]),
    }

    print("FC data preparation summary:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 若有被试被排除，再输出两层排除信息：
    #   1) 按 reason 汇总数量，快速判断主要问题是缺文件、时间点数异常，
    #      还是 time series 形状错误；
    #   2) 逐被试打印详细记录，方便直接定位具体 subject/source/run/file。
    if excluded_subjects:
        reason_counts = {}
        for item in excluded_subjects:
            reason_counts[item["reason"]] = reason_counts.get(item["reason"], 0) + 1

        print("Excluded subjects by reason:")
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count}")

        print("Excluded subject details:")
        for item in excluded_subjects:
            details = ", ".join(f"{key}={value}" for key, value in item.items())
            print(f"  {details}")

    return df_fc, FC_data

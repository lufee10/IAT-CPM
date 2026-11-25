# IAT-CPM

## Directory Structure

```text
IAT-CPM/
├── 0_mylib/
│   └── biox/
├── 1_Inputs/
├── 2_Computations/
└── 3_Visualizations/
```

- `0_mylib/`: Python utility modules used in the project. Its `biox/` subdirectory contains the Python package imported as `biox` in the scripts.
- `1_Inputs/`: Basic input data, including:
  - `HBN_func_HarvardOxfordSubcortical/`: Time series extracted from preprocessed MRI data using the Harvard–Oxford subcortical atlas.
  - `HBN_func_Schaefer200/`: Time series extracted from preprocessed MRI data using the Schaefer 200-parcel atlas.
  - `量表数据汇总_with_MeanEnorm.csv`: Questionnaire data corresponding to the MRI data.
- `2_Computations/`: Scripts for computational analyses and intermediate data.
- `3_Visualizations/`: Scripts for visualization and intermediate data. All input data come from `1_Inputs/` and `2_Computations/`.

## Analysis Scripts in `2_Computations/`

The table below summarizes each notebook and its corresponding manuscript sections.

| Notebook | Purpose and manuscript sections |
| :--- | :--- |
| [CPM_comparingBehav.ipynb](2_Computations/CPM_comparingBehav.ipynb) | Characterizes between-range differences in demographic, socioeconomic, and psychopathological characteristics.<br><br>**Methods 2.5.1 · Results 3.1** |
| [CPM_Validation_<br>rangeRestriction.ipynb](2_Computations/CPM_Validation_rangeRestriction.ipynb) | Provides a complementary assessment of range-dependent phenotypic associations by modeling each characteristic as a function of continuous PIU severity, using segmented regressions with a fixed knot at PIU = 30.<br><br>**Methods 2.5.1 · Results 3.1** |
| [CPM_GMM.ipynb](2_Computations/CPM_GMM.ipynb) | Tests whether independently derived clusters of covariate-adjusted whole-brain functional connectivity align with predefined PIU-severity ranges using PCA, Gaussian mixture modeling, and label permutations.<br><br>**Methods 2.5.2 · Results 3.2** |
| [CPM_GMM_validation.ipynb](2_Computations/CPM_GMM_validation.ipynb) | Evaluates the robustness of connectomic differentiation across PCA dimensionalities and repeated stratified participant subsamples.<br><br>**Methods 2.5.2 · Results 3.2** |
| [CPM_nestedCV-LOOCV.ipynb](2_Computations/CPM_nestedCV-LOOCV.ipynb) | Estimates connectome–behavior mappings separately within each PIU-severity range using nested CPM–Ridge, with outer leave-one-out cross-validation and inner selection of the ridge penalty.<br><br>**Methods 2.6.1 · Results 3.3** |
| [CPM_nestedCV-<br>Permutation.ipynb](2_Computations/CPM_nestedCV-<br>Permutation.ipynb) | Assesses the statistical significance of within-range prediction by comparing observed performance with permutation-derived null distributions.<br><br>**Methods 2.6.2 · Results 3.3** |
| [CPM_nestedCV_<br>crossGroup.ipynb](2_Computations/CPM_nestedCV_<br>crossGroup.ipynb) | Assesses bidirectional cross-range generalizability by applying transformations and models fitted within the source range to participants in the target range without target-range fitting or tuning.<br><br>**Methods 2.6.3 · Results 3.4** |
| [CPM_classic_Haufe.ipynb](2_Computations/CPM_classic_Haufe.ipynb) | Computes classical Haufe-transformed patterns and permutation-based summaries to characterize the organization of predictive connectivity at the edge and network levels.<br><br>**Methods 2.7.2 · Results 3.6** |

**Notes:** `lowIAT` (PIU ≤ 30) and `highIAT` (PIU > 30) denote severity ranges, not diagnoses. Outputs are stored in the corresponding subdirectories. Additional analyses for Sections 2.6.4 and 2.7.1 are included in `3_Visualizations/`. Section numbers refer to the manuscript *Range-Dependent Connectome–Behavior Mappings of Problematic Internet Use in Youth*.

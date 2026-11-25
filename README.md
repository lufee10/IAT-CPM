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

## Computational Analyses

The notebooks in `2_Computations/` implement the multilevel analytical framework described in the manuscript, from phenotypic and connectomic differentiation to within-range prediction, cross-range generalization, and the organization of predictive features.

### Phenotypic differentiation

**`CPM_comparingBehav.ipynb`**

Characterizes between-range differences in demographic, socioeconomic, and psychopathological characteristics.

*Manuscript correspondence:* **Section 2.5.1:** Phenotypic differentiation; **Section 3.1:** PIU severity exhibits distinct phenotypic embedding across ranges.

**`CPM_Validation_rangeRestriction.ipynb`**

Provides a complementary assessment of range-dependent phenotypic associations by modeling each characteristic as a function of continuous PIU severity, using segmented regressions with a fixed knot at PIU = 30.

*Manuscript correspondence:* **Section 2.5.1:** Phenotypic differentiation, specifically the reverse-direction segmented regression analysis; **Section 3.1:** PIU severity exhibits distinct phenotypic embedding across ranges.

### Connectomic differentiation

**`CPM_GMM.ipynb`**

Tests whether independently derived clusters of covariate-adjusted whole-brain functional connectivity align with predefined PIU-severity ranges using PCA, Gaussian mixture modeling, and label permutations.

*Manuscript correspondence:* **Section 2.5.2:** Connectomic differentiation; **Section 3.2:** Global FC organization varies across PIU severity ranges.

**`CPM_GMM_validation.ipynb`**

Evaluates the robustness of connectomic differentiation across PCA dimensionalities and repeated stratified participant subsamples.

*Manuscript correspondence:* **Section 2.5.2:** Connectomic differentiation, specifically the robustness analyses; **Section 3.2:** Global FC organization varies across PIU severity ranges.

### Within-range prediction and cross-range generalization

**`CPM_nestedCV-LOOCV.ipynb`**

Estimates connectome–behavior mappings separately within each PIU-severity range using nested CPM–Ridge, with outer leave-one-out cross-validation and inner selection of the ridge penalty.

*Manuscript correspondence:* **Section 2.6.1:** Nested CPM–Ridge prediction within PIU severity ranges; **Section 3.3:** Connectome-based models predict PIU severity within both ranges.

**`CPM_nestedCV-Permutation.ipynb`**

Assesses the statistical significance of within-range prediction by comparing observed performance with permutation-derived null distributions.

*Manuscript correspondence:* **Section 2.6.2:** Permutation inference for within-range prediction; **Section 3.3:** Connectome-based models predict PIU severity within both ranges.

**`CPM_nestedCV_crossGroup.ipynb`**

Assesses bidirectional cross-range generalizability by applying transformations and models fitted within the source range to participants in the target range without target-range fitting or tuning.

*Manuscript correspondence:* **Section 2.6.3:** Cross-range generalization of CPM–Ridge models; **Section 3.4:** Predictive mappings fail to generalize across severity ranges.

### Multiscale organization of predictive features

**`CPM_classic_Haufe.ipynb`**

Computes classical Haufe-transformed patterns and permutation-based summaries to characterize the organization of predictive connectivity at the edge and network levels.

*Manuscript correspondence:* **Section 2.7.2:** Haufe transformation and statistical inference; **Section 3.6:** Range-specific predictive features converge on FPN-centered organization.

Throughout the code, `lowIAT` and `highIAT` denote lower-severity (PIU ≤ 30) and higher-severity (PIU > 30) ranges rather than diagnostic groups. Corresponding subdirectories store analysis outputs and intermediate data. Additional analyses supporting **Section 2.6.4 (Validation analyses)**, **Section 2.7.1 (Edge-selection size and cross-range overlap)**, and their associated results are also implemented in `3_Visualizations/`.

Section numbering follows the manuscript *Range-Dependent Connectome–Behavior Mappings of Problematic Internet Use in Youth*.


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


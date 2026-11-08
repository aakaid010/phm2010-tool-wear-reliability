# Milling Tool-Wear Prediction and Reliability Diagnostics

Research notebooks by **Kazi Ahsan Ahmed Akaid** investigating milling tool-wear prediction and the reliability of predictions on held-out tools.

The project examines prediction accuracy, empirical prediction intervals, ensemble disagreement, explanation stability, and retrospective warning timing. It includes analyses using PHM2010, NASA Milling, and a separate nine-tool milling dataset.

## Repository contents

| Notebook | Description |
| --- | --- |
| [PHM2010 Data Exploration](phm2010-01-data-exploration.ipynb) | Data inspection, wear-label exploration, statistical and spectral feature extraction, and a Random Forest baseline. |
| [PHM2010 Deep Learning](phm2010-02-deep-learning.ipynb) | Signal-window preparation, CNN–BiLSTM–attention models, ensemble evaluation, empirical prediction intervals, explanation diagnostics, and warning-timing analysis. |
| [NASA Milling Validation](nasa-milling-independent-dataset-validation.ipynb) | Separate case-held-out evaluation using sensor-based and run/condition Random Forest baselines. |
| [Nine-Tool Independent Validation](wear-reliability-nine-tool-independent-validatio.ipynb) | Independent milling dataset analysis, including feature extraction, Random Forest ensembles, empirical intervals, and selective prediction. |

## Research objectives

- Evaluate tool-wear prediction on held-out cutters.
- Examine feature-based Random Forest and neural signal models.
- Assess empirical prediction-interval coverage and width.
- Investigate relationships between ensemble disagreement, prediction error, and explanation stability.
- Explore retrospective warning timing under illustrative wear thresholds.
- Examine related evaluation approaches on additional milling datasets.

## Datasets and methods

### PHM2010

The main analysis uses labeled cutters **C1, C4, and C6** from the PHM Society 2010 challenge dataset.

The recordings include force, vibration, and acoustic-emission signals.

The notebooks include:

- Statistical and spectral feature extraction.
- Random Forest regression.
- CNN–BiLSTM–attention regression.
- A five-seed neural ensemble.
- Cutter-held-out evaluation.
- Empirical prediction intervals.
- Explanation-stability diagnostics.
- Retrospective warning analysis.

The notebooks retain earlier exploratory cells and multiple iterations of the analysis.

Source data: [PHM Society 2010 Data Challenge](https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/)

### NASA Milling

The NASA notebook performs a separate case-held-out evaluation using newly trained Random Forest models.

It compares sensor-based prediction with a run/condition baseline and evaluates empirical prediction intervals. Its saved execution output reports results for **16 held-out cases**.

NASA case IDs are evaluation groups and should not be interpreted as verified individual cutter identities.

This analysis trains models on NASA data. It does not transfer PHM2010-trained model weights.

### Nine-tool milling dataset

The independent nine-tool analysis contains **6,418 recordings** from nine machine/tool groups across three machines.

| Partition | Tools | Recordings |
| --- | --- | ---: |
| Source | M1T1, M1T2, M2T4, M2T5, M3T7, M3T8 | 4,243 |
| Held-out test | M1T3, M2T6, M3T9 | 2,175 |

The notebook includes:

- Dataset inventory and HDF5 inspection.
- Tool grouping using machine and tool metadata.
- Source/test split generation.
- Statistical and spectral force-signal features.
- Random Forest models and ensembles.
- Standard and adaptive empirical prediction intervals.
- Selective prediction using ensemble disagreement.
- Figure and table generation from saved held-out predictions.

Models are trained separately on this dataset. PHM2010-trained neural weights are not transferred.

## Software requirements

The notebooks use Python and the following packages:

| Package | Purpose |
| --- | --- |
| NumPy | Numerical operations |
| pandas | Data loading and tabular analysis |
| SciPy | Signal processing and statistical features |
| Matplotlib | Figures and visualizations |
| scikit-learn | Random Forest models and evaluation metrics |
| PyTorch | Neural models and training |
| h5py | HDF5 recording access |
| joblib | Model persistence |
| tqdm | Progress reporting |
| PyArrow | Parquet support for the NASA dataset |

Jupyter or a compatible notebook environment is required for local execution. IPython is used for notebook display and download links.

A complete, pinned environment has not yet been validated.

## Running the notebooks

The notebooks were developed for **Kaggle** and use:

- `/kaggle/input` for source datasets.
- `/kaggle/working` for generated files.

Source datasets must be obtained separately and attached to the notebook environment.

### 1. PHM2010 data exploration

Open [phm2010-01-data-exploration.ipynb](phm2010-01-data-exploration.ipynb).

1. Attach the PHM2010 recordings and wear labels for C1, C4, and C6.
2. Check that the dataset paths match your environment.
3. Run the cells in order.
4. Preserve the generated `/kaggle/working/phm_features.csv`.

### 2. PHM2010 deep learning

Open [phm2010-02-deep-learning.ipynb](phm2010-02-deep-learning.ipynb).

1. Attach the PHM2010 recordings and wear labels.
2. Enable a GPU for neural training.
3. Make the feature CSV from the exploration notebook available at `/kaggle/working/phm_features.csv`.
4. Run dependent analysis cells in order.
5. Preserve generated predictions, checkpoints, and analysis outputs.

Separate Kaggle notebooks do not automatically share their working directories. Transfer the feature CSV explicitly when using separate sessions.

This notebook retains multiple analysis iterations and depends on earlier variables and generated files. It has not been validated as a clean “Run All” pipeline.

### 3. NASA Milling validation

Open [nasa-milling-independent-dataset-validation.ipynb](nasa-milling-independent-dataset-validation.ipynb).

1. Attach the expected NASA Milling Parquet dataset.
2. Update `DATA_PATH` to match the attached file.
3. Run the analysis cell.
4. Collect the generated reports from `/kaggle/working`.

### 4. Nine-tool independent validation

Open [wear-reliability-nine-tool-independent-validatio.ipynb](wear-reliability-nine-tool-independent-validatio.ipynb).

1. Attach the milling dataset containing `filelist.csv` and the referenced HDF5 recordings.
2. Update every dataset-path definition if necessary.
3. Run the cells in order.
4. Collect generated files from `/kaggle/working/nine_tool_validation`.

The notebook contains assertions for a specific dataset snapshot. A different dataset version may fail these checks and should be investigated before continuing.

### Local execution

To run outside Kaggle, replace all `/kaggle/input` and `/kaggle/working` paths with suitable local directories.

A GPU is recommended for PHM2010 neural training. Runtime and memory requirements have not been benchmarked for this repository.

## Outputs

Depending on the notebook and completed cells, generated artifacts include:

- Engineered feature tables.
- Processed signal windows.
- Dataset splits and protocol specifications.
- Trained model files.
- Per-recording predictions.
- Prediction-accuracy summaries.
- Empirical interval coverage and width summaries.
- Explanation diagnostics.
- Selective-prediction results.
- Warning-timing summaries.
- Figures, tables, and ZIP archives.

The repository currently contains the four notebooks and this README. Raw datasets and separate generated artifacts are not included. Saved outputs embedded in the notebooks are retained.

## Reproducibility status

This repository preserves a research-code snapshot with saved execution outputs. It has not been independently rerun from a clean environment.

The notebooks provide partial code coverage of the associated revised manuscript, *Calibration-Source Sensitivity and Warning Timing in Cutter-Held-Out Milling Tool-Wear Prediction*.

Implementations and output artifacts for the manuscript’s newer paired calibration-source experiment, chronological reference, and four-window Random Forest control have not been located in these notebooks.

The NASA and nine-tool analyses are additional independent-dataset experiments and should be distinguished from the results reported in the supplied PHM2010-focused manuscript.

The manuscript PDF is not currently included in this repository.

## Interpretation and limitations

- Three PHM2010 cutter trajectories do not establish population-level generalization.
- Recordings within a tool are dependent observations.
- Nominal 90% conformal coverage is not guaranteed under cross-tool distribution shift and serial dependence.
- The associated manuscript describes the blocked split as selected after test inspection; those findings are exploratory.
- Ensemble disagreement is not a complete estimate of predictive uncertainty.
- Selective-prediction error must be interpreted alongside the fraction of predictions retained.
- Warning thresholds and relative penalties are retrospective scenarios, not a validated tool-replacement policy or measured cost savings.
- Additional-dataset analyses train new models rather than transferring PHM2010-trained weights.
- Saved outputs document previous execution but do not establish clean end-to-end reproduction.

## Author

**Kazi Ahsan Ahmed Akaid**

GitHub: [@aakaid010](https://github.com/aakaid010)

## Citation

If referring to this repository, acknowledge the author and link to:

https://github.com/aakaid010/phm2010-tool-wear-reliability

A formal manuscript citation can be added when a public manuscript or archive record is available. No publication venue or DOI is claimed here.

## License and data rights

No software license has been selected for this research snapshot.

Dataset rights remain with their respective providers. Obtain source datasets separately and follow their access, attribution, and redistribution terms.

# Milling Tool-Wear Prediction and Reliability Diagnostics

Research notebooks accompanying the draft manuscript **Calibration-Source Sensitivity and Warning Timing in Cutter-Held-Out Milling Tool-Wear Prediction**, by **Kazi Ahsan Ahmed Akaid**.

This project studies a practical question: when a model predicts milling tool wear, how much confidence should we place in its predictions on another cutter?

The analysis examines prediction accuracy, empirical prediction-interval coverage, ensemble disagreement, explanation stability, and retrospective warning timing.

## Project status

This repository is a research snapshot containing four notebooks with saved execution outputs and a manuscript draft.

The repository has not been independently rerun from a clean environment. Some newer experiments reported in the revised manuscript are not included in the supplied notebooks, so the current release provides partial manuscript coverage.

The included PDF contains author-information and declaration placeholders that should be completed before a final public release.

## Repository contents

| File | Description |
| --- | --- |
| [01_phm2010_data_exploration.ipynb](notebooks/01_phm2010_data_exploration.ipynb) | PHM2010 data inspection, wear-label exploration, statistical and spectral feature extraction, and a Random Forest baseline. |
| [02_phm2010_deep_learning.ipynb](notebooks/02_phm2010_deep_learning.ipynb) | Signal-window preparation, CNN–BiLSTM–attention models, ensemble evaluation, empirical prediction intervals, explanation diagnostics, and warning-timing analysis. |
| [03_nasa_milling_validation.ipynb](notebooks/03_nasa_milling_validation.ipynb) | Separate NASA Milling case-held-out evaluation using sensor-based and run/condition Random Forest baselines. |
| [04_nine_tool_independent_validation.ipynb](notebooks/04_nine_tool_independent_validation.ipynb) | Independent nine-tool milling dataset analysis, including feature extraction, Random Forest ensembles, empirical intervals, and selective prediction. |
| [manuscript.pdf](paper/manuscript.pdf) | Revised manuscript draft. |
| [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Execution order, dependencies, dataset paths, and known reproducibility gaps. |

## Research objectives

- Evaluate tool-wear predictions on held-out cutters.
- Compare feature-based Random Forest models with neural signal models.
- Examine whether prediction intervals provide useful empirical coverage.
- Investigate relationships between ensemble disagreement, prediction error, and explanation stability.
- Assess retrospective warning timing under illustrative wear thresholds.
- Explore related evaluation protocols on additional milling datasets.

## Datasets and evaluation

### PHM2010

The main notebooks use labeled cutters **C1, C4, and C6** from the PHM Society 2010 challenge dataset.

The recordings include force, vibration, and acoustic-emission signals.

The analysis includes:

- Statistical and spectral feature extraction.
- Random Forest regression.
- CNN–BiLSTM–attention regression.
- A five-seed neural ensemble.
- Cutter-held-out evaluation.
- Empirical prediction intervals.
- Explanation-stability diagnostics.
- Retrospective warning analysis.

Earlier exploratory cells are retained in the notebooks.

The revised manuscript additionally examines how changing the source of calibration residuals affects interval coverage and width while holding fitted predictions fixed. It also reports chronological and input-coverage controls. Implementations and output artifacts for these newer experiments have not been located in the supplied notebooks.

Source data:
[PHM Society 2010 Data Challenge](https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/)

### NASA Milling

The NASA notebook performs a separate case-held-out evaluation using newly trained Random Forest models.

It compares sensor-based prediction with a run/condition baseline and evaluates empirical prediction intervals.

Its saved execution output reports results for **16 held-out cases**.

NASA case IDs are evaluation groups and should not be interpreted as verified individual cutter identities. PHM2010-trained model weights are not transferred to this dataset.

This analysis is supplementary to the supplied PHM2010-focused manuscript.

### Nine-tool milling dataset

The fourth notebook analyzes a separate milling dataset containing **6,418 recordings** from nine machine/tool groups across three machines.

The split contains:

| Partition | Tools | Recordings |
| --- | --- | ---: |
| Source | M1T1, M1T2, M2T4, M2T5, M3T7, M3T8 | 4,243 |
| Held-out test | M1T3, M2T6, M3T9 | 2,175 |

The notebook includes:

- Dataset and HDF5 structure inspection.
- Machine/tool grouping and split-file generation.
- Statistical and spectral force-signal features.
- Random Forest models and ensembles.
- Standard and adaptive empirical intervals.
- Selective prediction using ensemble disagreement.
- Figure and table generation from saved held-out predictions.

New models are trained on this dataset. PHM2010-trained neural weights are not transferred.

These results are supplementary and are not numerical results reported in the supplied PHM2010 manuscript.

## Software

The notebooks use Python with:

- NumPy
- pandas
- SciPy
- Matplotlib
- scikit-learn
- PyTorch
- h5py
- joblib
- tqdm

The NASA notebook also requires a pandas-compatible Parquet engine, such as PyArrow. Jupyter or a compatible notebook environment is needed for local execution.

A complete, pinned environment has not yet been validated. See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the available environment information.

## Running the notebooks

The notebooks were developed for **Kaggle** and use:

- `/kaggle/input` for source datasets.
- `/kaggle/working` for generated files.

### PHM2010 workflow

1. Obtain the PHM2010 recordings and wear labels for C1, C4, and C6.
2. Upload `01_phm2010_data_exploration.ipynb` to Kaggle.
3. Attach the dataset and run the cells in order.
4. Preserve the generated `phm_features.csv`.
5. Upload `02_phm2010_deep_learning.ipynb` and attach the same recordings.
6. Enable a GPU for neural training.
7. Make the feature CSV available at `/kaggle/working/phm_features.csv`.
8. Run dependent analysis cells in order.

Separate Kaggle notebooks do not automatically share their working directories. The deep-learning notebook retains multiple analysis iterations and has not been validated as a clean “Run All” pipeline.

### NASA workflow

1. Upload `03_nasa_milling_validation.ipynb` to Kaggle.
2. Attach the expected NASA Milling Parquet dataset.
3. Update `DATA_PATH` to match the attached file.
4. Run the analysis cell.

### Nine-tool workflow

1. Upload `04_nine_tool_independent_validation.ipynb` to Kaggle.
2. Attach the milling dataset containing `filelist.csv` and the referenced HDF5 recordings.
3. Update every dataset-path definition if necessary.
4. Run the cells in order.
5. Collect generated files from `/kaggle/working/nine_tool_validation`.

The notebook contains assertions for a specific dataset snapshot. Different dataset versions may require investigation before the analysis can run successfully.

For local execution, adapt all Kaggle-specific paths consistently.

## Generated artifacts

Depending on the notebook and completed analysis cells, generated artifacts include:

- Feature tables and processed signal windows.
- Split files and protocol specifications.
- Model checkpoints.
- Per-recording predictions.
- Accuracy and interval-evaluation tables.
- Explanation diagnostics.
- Warning-timing summaries.
- Figures and result archives.

Raw datasets, generated feature and sequence files, model checkpoints, and result archives are not included in this repository snapshot.

## Interpretation and limitations

- Three PHM2010 cutter trajectories do not establish population-level generalization.
- Recordings from the same tool are dependent observations.
- Nominal 90% conformal coverage is not guaranteed under cross-cutter distribution shift and serial dependence.
- The manuscript states that the blocked split was selected after test inspection; those findings are exploratory.
- Ensemble disagreement is not a complete estimate of predictive uncertainty.
- Selective-prediction error must be interpreted alongside the fraction of predictions retained.
- Warning thresholds and relative penalties are retrospective scenarios, not a validated tool-replacement policy or measured cost savings.
- The additional datasets involve separate model training rather than direct transfer of PHM2010-trained weights.
- Saved notebook outputs document previous execution but do not establish clean end-to-end reproduction of the current manuscript.

## Manuscript and attribution

**Title:** Calibration-Source Sensitivity and Warning Timing in Cutter-Held-Out Milling Tool-Wear Prediction

**Author:** Kazi Ahsan Ahmed Akaid

**Manuscript:** [Read the draft PDF](paper/manuscript.pdf)

No publication venue or DOI is claimed in this repository. A formal citation can be added when a public manuscript or archive record is available.

## License and data rights

No software license has been selected for this research snapshot.

Dataset rights remain with their respective providers. Obtain the source datasets separately and follow their access, attribution, and redistribution terms.

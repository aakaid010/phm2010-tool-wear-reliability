# Calibration Sources for Tool-Wear Prediction Intervals Under Tool and Machine Shift in Milling

Code, result files and the literature table behind the paper

> K. A. A. Akaid and F. Fariha, *Calibration Sources for Tool-Wear Prediction Intervals Under Tool and Machine Shift in Milling*. Submitted to *The International Journal of Advanced Manufacturing Technology*.

Archived on Zenodo (concept DOI, resolves to the latest version):
[10.5281/zenodo.24378510](https://doi.org/10.5281/zenodo.24378510). License: MIT.

## Main results

Split-conformal prediction intervals for tool wear are only as reliable as the data used to calibrate
them. On the nine-tool dataset (9 tools on 3 machines, random forest, nominal 90% coverage, mean over
tools; from `results/t3_summary.txt`):

| Deployment | Calibration source | Coverage (%) | Mean width (µm) |
| --- | --- | --- | --- |
| New tool, seen machine | Residuals of the training tools (interleaved cuts) | 53.7 | 11.0 |
| New tool, seen machine | Late-life block of the training tools | 97.4 | 90.4 |
| New tool, seen machine | Leave-one-tool-out | 88.8 | 29.4 |
| New machine | Residuals of the training tools (interleaved cuts) | 29.1 | 10.9 |
| New machine | Leave-one-tool-out | 65.8 | 30.2 |
| New machine | Leave-one-machine-out | 94.5 | 57.3 |

Calibration data must reproduce the shift the model will face in deployment: held-out tools for a new
tool, held-out machines for a new machine. In-sample residuals give intervals far below nominal coverage,
and a late-life block reaches coverage only with intervals about three times wider. Gradient boosting
gives the same pattern (`results/t3_summary_hgb.txt`; leave-one-tool-out 90.5%, leave-one-machine-out
96.6%). With 9 tools and 3 machines, these are descriptive results, not general guarantees.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py                 # results/*.csv -> every table, figure and in-text number of the paper
make all                          # same, where make is available
```

`run_all.py` writes LaTeX tables and number macros to `<paper>/generated/` and figure PDFs to
`<paper>/figures/`, where `<paper>` is `../paper` or `../manuscript` (override with `--paper`); in a fresh
clone without a paper directory they go to `./output/` and the manuscript check is skipped. It then
runs `scripts/check_manuscript.py`, which fails if the paper uses a number macro that is not generated,
cites a key missing from `references.bib`, or exceeds the abstract word limit.

From raw data (features → experiments → paper):

```bash
python run_all.py --phm-data <PHM2010 dir> --ninetool-data <nine-tool dir>
python run_all.py --experiments   # rerun experiments from data/phm_features.csv, data/nine_tool_features.csv
```

The nine-tool experiments fit several hundred random forests (≈1 h on 4 CPU cores); the PHM2010 ones take a few minutes.

## Data

| Dataset | Source | What the scripts need |
| --- | --- | --- |
| PHM2010 milling (cutters C1, C4, C6) | PHM Society 2010 Data Challenge, <https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/>; also IEEE DataPort, doi:[10.21227/jdxd-yy51](https://doi.org/10.21227/jdxd-yy51) | `c_<k>_<nnn>.csv` sensor files and `c<k>_wear.csv` for k = 1, 4, 6, anywhere below `--phm-data` |
| Nine-tool milling dataset (Denkena, Klemme & Stiehl, 2023, doi:[10.1016/j.dib.2023.109574](https://doi.org/10.1016/j.dib.2023.109574)) | Mendeley Data, version 3, doi:[10.17632/zpxs87bjt8.3](https://doi.org/10.17632/zpxs87bjt8.3) (6,418 HDF5 files, ≈10 GB) | `filelist.csv` and the HDF5 files in `--ninetool-data` |

Raw data are not redistributed here. The feature tables (`data/phm_features.csv`,
`data/nine_tool_features.csv`) are produced by the feature scripts and are git-ignored.

## Map from the paper to the code

Table and figure numbers refer to the current manuscript.

| Paper item | Produced by | From |
| --- | --- | --- |
| Figure 1 (study overview), Figure 2 (calibration sources) | `scripts/fig_schematics.py` (design only, no data) | — |
| Table 1 (literature; `tab:literature`) and the study counts in Sections 1–2 | `scripts/make_literature_table.py` | `data/literature.csv` (one row per study, with the evidence sentence and whether the full text or only the abstract was read) |
| Tables 2–3 (nine-tool, RF and HGB; `tab:ninetool`, `tab:ninetool_hgb`) | `scripts/make_tables.py` | `results/t3_deploy*.csv`, `t3_machine_out*.csv`, `t3_matched*.csv` (`ninetool_calibration.py`, RF and `--model hgb`) |
| Wilcoxon results quoted in Sections 4.1–4.2 | `scripts/stats_wilcoxon.py` → `make_tables.py` | `results/t3_wilcoxon_tests*.csv`, `t3_wilcoxon_pertool*.csv` |
| Figure 3 (per-tool coverage at 90%), Figure 4 (coverage vs nominal level) | `scripts/fig_results.py` | `results/t3_deploy.csv`, `results/t3_machine_out.csv` |
| Table 4 (interval width in µm and runs of tool life; `tab:deploy_guidance`) | `scripts/make_tables.py` (`deployment_guidance`) | `results/t3_deploy.csv`, `t3_machine_out.csv`, `wear_rate_ninetool.csv` |
| Table 5 (first warnings; `tab:warning`) | `scripts/make_tables.py` | `results/t2_warning_timing.csv` (`phm2010_rf_chrono.py`) |
| Table 6 (warning decision cost; `tab:warning_cost`) | `scripts/warning_cost.py` → `make_tables.py` | `results/t2_warning_timing.csv` → `results/t2_warning_cost.csv` |
| Table 7 (PHM2010 matched calibration sources; `tab:phm_matched`) | `scripts/make_tables.py` | `results/t1_matched_rf.csv`, `t1_matched_hgb.csv` (`phm2010_rf_chrono.py`, RF and `--model hgb --matched-only`), `t1_label_range.csv` (`phm2010_label_range.py`) |
| Wear-rate mismatch (Section 4.4) | `scripts/fig_wear_mismatch.py` → `make_tables.py` | `results/t5_spearman.csv`, `t5_wear_mismatch.csv` |
| Table 8 (point accuracy, per tool; `tab:point_accuracy`) | `scripts/make_tables.py` | `results/t2_point_accuracy.csv` (`phm2010_rf_chrono.py`) |
| Every other number in the text | `scripts/make_tables.py` → `generated/numbers.tex` | the result files above |

Shared code: `scripts/calib.py` (conformal quantile with equal tool weights, interval score,
life-stage split). Features: `scripts/features_phm2010.py` and `scripts/features_ninetool.py`, ported
unchanged from the feature-extraction cells of the two notebooks in `notebooks/`.
`scripts/verify_references.py` checks every DOI in the paper's `references.bib` against Crossref/DataCite, and
`scripts/make_reference_audit.py` writes the audit table. `scripts/check_manuscript.py`, `check_numbers.py` and
`list_key_numbers.py` check the manuscript source; `scripts/flatten.py` builds the single-file LaTeX
submission (no `\input`, figures renamed Fig1–FigN).

## Environment

The experiments were originally run in Kaggle notebooks (see `notebooks/`); the tables and figures are
regenerated locally with the pinned versions in `requirements.txt`. Setup:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions with which the post-processing stage (`run_all.py` without
data) was rerun and checked on 21 Nov 2026 (Python 3.13.9). The experiment runs that produced
`results/` were executed on Kaggle before version recording was added to the scripts; their
configuration files (`results/*config*.json`) record seeds, tree counts and options. From this release
on, every experiment run also writes the Python, NumPy, pandas, SciPy and scikit-learn versions into
its configuration file. Random-forest results can change slightly across scikit-learn versions.

## Reproducibility status (21 Nov 2026)

* Verified: `python run_all.py` regenerates every generated table and number macro of the paper
  byte-for-byte from `results/`.
* Not rerun here: raw data → features → `results/`. The PHM2010 raw data require a download that
  was not available in this environment.
* Pending rerun: `phm2010_rf_chrono.py` now also writes per-cut false positives/negatives of each warning
  rule (`fp_cuts`, `fn_cuts`); `results/t2_warning_timing.csv` predates this change, so Table 5 shows
  them only after a rerun.

## Repository layout

```
run_all.py            single entry point (see above)
scripts/              every script that produces a number, table or figure in the paper
results/              experiment outputs used by the paper (CSV/JSON/TXT)
data/literature.csv   the literature table, with evidence per study
notebooks/            the original Kaggle notebooks the feature scripts were ported from
archive/              exploratory work not used in the paper (see archive/README.md)
```

`results/t2_point_accuracy_legacy.csv` is kept for the record and is not read by any script.

## Citation

If you use this code, please cite the Zenodo archive (concept DOI, always resolves to the latest version):

> K. A. A. Akaid and F. Fariha. *Code for: Calibration Sources for Tool-Wear Prediction Intervals Under Tool and Machine Shift in Milling*. Zenodo. doi:[10.5281/zenodo.24378510](https://doi.org/10.5281/zenodo.24378510)

```bibtex
@software{akaid_fariha_code,
  author    = {Akaid, Kazi Ahsan Ahmed and Fariha, Faozia},
  title     = {Code for: Calibration Sources for Tool-Wear Prediction Intervals Under Tool and Machine Shift in Milling},
  publisher = {Zenodo},
  version   = {v1.3},
  doi       = {10.5281/zenodo.24378510},
  url       = {https://doi.org/10.5281/zenodo.24378510}
}
```

## Authors

Kazi Ahsan Ahmed Akaid (corresponding, akaid0001@gmail.com) and Faozia Fariha, Chittagong University
of Engineering & Technology (CUET), Bangladesh.

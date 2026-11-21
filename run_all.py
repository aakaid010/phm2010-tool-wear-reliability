"""Regenerate every table, figure and in-text number of the paper.

  python run_all.py                                   # from the result CSVs in results/
  python run_all.py --phm-data D1 --ninetool-data D2  # from raw data: features -> experiments -> paper
  python run_all.py --experiments                     # rerun experiments from data/*.csv features

Stages
  1 features    (only with --phm-data / --ninetool-data)  -> data/phm_features.csv, data/nine_tool_features.csv
  2 experiments (with --experiments, or after stage 1)   -> results/*.csv   (CPU; the nine-tool runs take ~1 h)
  3 paper       (always)  results/*.csv -> <paper>/generated/*.tex and <paper>/figures/*.pdf
The paper directory defaults to ../paper or ../manuscript (whichever exists); set it with --paper.
If neither exists (e.g. a fresh clone), outputs go to ./output/ and the manuscript check is skipped.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
S = ROOT / "scripts"


def run(*args):
    cmd = [sys.executable, *map(str, args)]
    print("\n>>", " ".join(str(a) for a in args), flush=True)
    subprocess.run(cmd, cwd=S, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", default=None, help="paper directory holding main.tex")
    ap.add_argument("--phm-data", default=None, help="raw PHM2010 directory (c_k_nnn.csv, ck_wear.csv)")
    ap.add_argument("--ninetool-data", default=None, help="unpacked Mendeley record zpxs87bjt8 v3")
    ap.add_argument("--experiments", action="store_true", help="rerun all experiments from data/*.csv")
    a = ap.parse_args()

    paper = Path(a.paper) if a.paper else next(
        (p for p in (ROOT.parent / "paper", ROOT.parent / "manuscript") if p.exists()), None)
    if paper is None:
        paper = ROOT / "output"
        print(f"no paper directory found (../paper or ../manuscript); writing outputs to {paper}")
    data, res = ROOT / "data", ROOT / "results"
    phm_f, nine_f = data / "phm_features.csv", data / "nine_tool_features.csv"

    # 1 features
    if a.phm_data:
        run("features_phm2010.py", "--data", a.phm_data, "--out", phm_f)
    if a.ninetool_data:
        run("features_ninetool.py", "--data", a.ninetool_data, "--out", nine_f)

    # 2 experiments
    if a.experiments or a.phm_data or a.ninetool_data:
        if phm_f.exists():
            run("phm2010_rf_chrono.py", "--features", phm_f, "--out", res)
            run("phm2010_rf_chrono.py", "--features", phm_f, "--out", res, "--model", "hgb", "--matched-only")
            run("phm2010_label_range.py", "--features", phm_f, "--out", res)
        else:
            print("skip PHM2010 experiments: data/phm_features.csv missing")
        if nine_f.exists():
            run("ninetool_calibration.py", "--features", nine_f, "--out", res)
            run("ninetool_calibration.py", "--features", nine_f, "--out", res, "--model", "hgb")
            run("stats_wilcoxon.py", "--results", res)
            run("stats_wilcoxon.py", "--results", res, "--suffix", "_hgb")
        else:
            print("skip nine-tool experiments: data/nine_tool_features.csv missing")

    # 3 paper outputs
    gen, figs = paper / "generated", paper / "figures"
    run("fig_wear_mismatch.py", "--results", res, "--out", res / "extra")   # t5_*.csv (+ unused figure)
    run("warning_cost.py", "--results", res)
    run("make_tables.py", "--results", res, "--out", gen)
    run("make_literature_table.py", "--csv", data / "literature.csv", "--out", gen)
    run("fig_results.py", "--results", res, "--out", figs)
    if (paper / "main.tex").exists():
        run("check_manuscript.py", "--tex", paper / "main.tex")
    else:
        print(f"skip check_manuscript.py: {paper / 'main.tex'} not found")

    sys.path.insert(0, str(S))
    from calib import env_versions
    (res / "postprocess_versions.json").write_text(json.dumps(env_versions(), indent=2) + "\n")
    print("\nDone: tables/macros ->", gen, "| figures ->", figs)


if __name__ == "__main__":
    main()

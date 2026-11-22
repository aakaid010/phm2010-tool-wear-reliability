"""Check that every coverage / width / interval-score macro used in main.tex has a value that
also appears in the generated tables (Tables 2-8), so the text and the tables agree.
Macros for quantities that the tables do not show (e.g. worst-tool coverage in the final third)
are listed as 'not tabulated'.

  python check_text_vs_tables.py --tex ../manuscript/main.tex
"""
import argparse
import re
from pathlib import Path

PREFIX = ("cov", "covmin", "wid", "is", "covLast", "covminLast", "covRange", "widRange",
          "isRange", "covLastRange", "halfWid", "pctAbove", "pctAboveWorst", "runsHalf", "lifeHalf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", required=True)
    a = ap.parse_args()
    tex = Path(a.tex).read_text(encoding="utf-8")
    gen = Path(a.tex).parent / "generated"
    nums = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (gen / "numbers.tex").read_text(encoding="utf-8")))
    cells = set()
    for f in gen.glob("tab_*.tex"):
        if f.name == "tab_literature.tex":
            continue
        cells |= set(re.findall(r"-?\d+(?:\.\d+)?", f.read_text(encoding="utf-8")))
    used = sorted({m for m in re.findall(r"\\([A-Za-z]+)", tex) if m in nums
                   and re.match(r"(%s)[A-Z]" % "|".join(sorted(PREFIX, key=len, reverse=True)), m)})
    ok, missing, untab = 0, [], []
    for m in used:
        v = nums[m]
        if v in cells:
            ok += 1
        elif m.startswith("covminLast") or m.startswith("covLastRange"):
            untab.append(f"{m}={v}")
        else:
            missing.append(f"{m}={v}")
    print(f"macros checked: {len(used)}; found in Tables 2-8: {ok}")
    print("not tabulated (worst tool in final third / label-band final third):", untab or "none")
    print("MISMATCH (value not found in any table):", missing or "none")
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()

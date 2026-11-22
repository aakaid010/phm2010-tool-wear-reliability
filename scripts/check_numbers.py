"""List every number typed literally (not via a generated macro) in the abstract, Results,
Discussion and Conclusion of main.tex, and say where it comes from.

A literal is accepted if it is (a) a design constant of the study (nominal levels, threshold,
cut ranges, counts of tools/cutters/machines, window lengths), listed below with its source,
or (b) found among the values written by make_tables.py / make_literature_table.py
(numbers.tex, lit_numbers.tex, tab_*.tex). Anything else is printed as UNSOURCED.
Spelled-out ratios such as "twice" or "three times" are flagged too.

  python check_numbers.py --tex <paper>/main.tex
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

DESIGN = {  # literal -> where the value is fixed in the study design
    "50": "lowest nominal level (calib.LEVELS) / 50-cut window", "95": "highest nominal level",
    "90": "reported nominal level", "150": "wear threshold (TAU)", "220": "chronological fit cuts 1-220",
    "315": "cuts per PHM2010 cutter", "3": "cutters / machines / tools per machine",
    "9": "tools in the nine-tool data", "1": "cutter / machine identifiers", "4": "cutter C4",
    "6": "cutter C6", "2": "machine 2", "5": "false-negative weight in tuning and cost",
    "12": "physical tools in the study", "20": "sensitivity window", "30": "main warning window",
    "2010": "dataset name", "0.05": "conventional significance level (p < 0.05)",
}
SECTIONS = [("abstract", r"\\abstract\{", r"\\keywords\{"),
            ("results+discussion", r"\\section\{Results\}", r"\\subsection\{Limitations\}"),
            ("limitations", r"\\subsection\{Limitations\}", r"\\section\{Conclusions\}"),
            ("conclusion", r"\\section\{Conclusions\}", r"\\bmhead\{Acknowledgements\}")]
WORDS = r"\b(twice|thrice|half(?!-width)|double|triple|(?:two|three|four|five|ten) times)\b"


def strip(s):
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = re.sub(r"\\(?:ref|label|cite[pt]?|input|includegraphics|textsuperscript)(\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\d*\.?\d+\\linewidth", " ", s)  # layout lengths, e.g. 0.92\linewidth
    s = re.sub(r"\\[A-Za-z]+", " ", s)          # macros (their values are generated)
    s = re.sub(r"\b[CMT]\d\b|\bM\dT\d\b|\bS\d\b|\bH\d\b|\bR\d\b", " ", s)   # identifiers C1, M2, S4, H3...
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", required=True)
    a = ap.parse_args()
    tex = Path(a.tex).read_text(encoding="utf-8")
    gen = Path(a.tex).parent / "generated"
    pool = set()
    for f in gen.glob("*.tex"):
        for v in re.findall(r"-?\d+(?:\.\d+)?", f.read_text(encoding="utf-8")):
            pool.add(v)
            pool.add(v.lstrip("-"))
    bad = 0
    for name, start, end in SECTIONS:
        i = re.search(start, tex).start()
        j = re.search(end, tex[i:]).start() + i
        body = strip(tex[i:j])
        for m in re.finditer(WORDS, body, re.I):
            print(f"[{name}] WORD RATIO  '{m.group(0)}' - replace with a generated ratio macro")
            bad += 1
        for v in sorted(set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", body)), key=float):
            if v in DESIGN:
                continue
            src = "generated output" if v in pool else None
            if src is None:
                print(f"[{name}] UNSOURCED   {v}")
                bad += 1
    print("number check:", "PASS" if not bad else f"{bad} item(s) to review")


if __name__ == "__main__":
    main()

"""List every generated number used in the abstract and the conclusions, with its value from
generated/numbers.tex, and flag any literal digit typed there that is not a design constant.

  python list_key_numbers.py --tex ../manuscript/main.tex
"""
import argparse
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", required=True)
    a = ap.parse_args()
    tex = Path(a.tex).read_text(encoding="utf-8")
    gen = Path(a.tex).parent / "generated"
    nums = {}
    for f in ("numbers.tex", "lit_numbers.tex"):
        nums.update(dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (gen / f).read_text(encoding="utf-8"))))
    parts = {"Abstract": tex[tex.index("\\abstract{"):tex.index("\\keywords{")],
             "Conclusions": tex[tex.index("\\section{Conclusions}"):tex.index("\\bmhead{Acknowledgements}")]}
    for name, s in parts.items():
        used = [m for m in re.findall(r"\\([A-Za-z]+)", s) if m in nums]
        print(f"## {name}")
        for m in dict.fromkeys(used):
            print(f"- \\{m} = {nums[m]}")
        stripped = re.sub(r"\\[A-Za-z]+", " ", s)
        lits = sorted(set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", stripped)))
        print("  literal digits typed:", lits)


if __name__ == "__main__":
    main()

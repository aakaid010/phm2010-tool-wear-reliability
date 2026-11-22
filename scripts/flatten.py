"""Build the flat LaTeX submission folder that IJAMT / Editorial Manager requires.

The journal asks for one .tex file and no subfolders. This script copies the working source
(manuscript/) to <out>/, inlines every \\input{generated/...} file, renames the figures to
Fig1.pdf ... FigN.pdf in order of appearance, and copies references.bib, the .bbl (if built),
sn-jnl.cls and sn-mathphys-num.bst. Numbers are not touched: the inlined text is exactly the
generated files.

  python flatten.py --src ../manuscript --out ../submission/latex
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../manuscript")
    ap.add_argument("--out", default="../submission/latex")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tex = (src / "main.tex").read_text(encoding="utf-8")

    def inline(m):
        name = m.group(1)
        p = src / (name if name.endswith(".tex") else name + ".tex")
        body = p.read_text(encoding="utf-8").rstrip("\n")
        return f"% ---- begin {name} (generated; do not edit) ----\n{body}\n% ---- end {name} ----"

    tex = re.sub(r"\\input\{(generated/[^}]+)\}", inline, tex)
    assert "\\input{" not in tex, "unexpected remaining \\input"

    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    mapping = {}
    for f in figs:
        if f in mapping:
            continue
        n = len(mapping) + 1
        ext = Path(f).suffix or ".pdf"
        new = f"Fig{n}{ext}"
        shutil.copyfile(src / f, out / new)
        mapping[f] = new
    for old, new in mapping.items():
        tex = tex.replace("{" + old + "}", "{" + new + "}")
    tex = tex.replace("%  WORKING SOURCE.", "%  FLATTENED SUBMISSION SOURCE (built by experiments/flatten.py from the working source).\n%  WORKING SOURCE NOTE:")
    (out / "main.tex").write_text(tex, encoding="utf-8", newline="\n")
    for f in ("references.bib", "main.bbl", "sn-jnl.cls", "sn-mathphys-num.bst"):
        if (src / f).exists():
            shutil.copyfile(src / f, out / f)
    print("figures:", mapping)
    print("wrote", out / "main.tex")


if __name__ == "__main__":
    main()

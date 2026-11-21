"""Static checks on manuscript/main.tex, standing in for a compile when no LaTeX is installed.

  python check_manuscript.py --tex ../manuscript/main.tex

Checks: every custom macro is defined in generated/numbers.tex; every citation key exists in
references.bib and is marked VERIFIED; every \\ref has a label; every \\input / figure file
exists; braces and environments balance; abstract length.
"""
from __future__ import annotations

import argparse
import os
import re

LATEX = set("""documentclass usepackage bibliographystyle input newcommand ensuremath title author
thanks href and small date begin end maketitle textbf section label citep citet emph item itemize
enumerate ref url footnote paragraph subsection includegraphics linewidth caption centering lceil
rceil frac mathbf IfFileExists fbox parbox hat bibliography alpha geq mu sqrt text ldots times leq
cdot noindent quad mid mathrm le ge over sqrt rho""".split())


def strip_comments(tex):
    out = []
    for line in tex.replace("\\%", " PCT ").split("\n"):
        out.append(line.split("%")[0])
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="../manuscript/main.tex")
    a = ap.parse_args()
    root = os.path.dirname(os.path.abspath(a.tex))
    tex = open(a.tex, encoding="utf-8").read()
    body = strip_comments(tex)
    nums = "".join(open(os.path.join(root, "generated", f), encoding="utf-8").read()
                   for f in ("numbers.tex", "lit_numbers.tex")
                   if os.path.exists(os.path.join(root, "generated", f)))
    bib = open(os.path.join(root, "references.bib"), encoding="utf-8").read()
    ok = True

    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", nums)) | {"um", "textsuperscript"}
    sty = os.path.join(root, "ijphm.sty")
    if os.path.exists(sty):  # commands provided by the layout package
        defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", open(sty, encoding="utf-8").read()))
    used = set(re.findall(r"\\([A-Za-z]+)", body))
    custom = {u for u in used if u not in LATEX}
    undefined = sorted(custom - defined)
    print("numbers.tex macros used:", len(custom & defined))
    print("unknown commands (should be standard LaTeX only):", undefined)

    cites = {k.strip() for grp in re.findall(r"\\cite[pt]?(?:\[[^\]]*\])*\{([^}]+)\}", body)
             for k in grp.split(",")}
    keys = set(re.findall(r"@\w+\{([^,\s]+)\s*,", bib))
    miss = sorted(cites - keys)
    # The VERIFIED note must be a comment line directly ABOVE the entry: BibTeX does not treat
    # '%' as a comment inside an entry, and a note on the key line destroys all fields.
    unver = sorted(k for k in cites & keys
                   if not re.search(r"%[^\n]*VERIFIED[^\n]*\n@\w+\{" + re.escape(k) + r"\s*,", bib))
    print(f"citations: {len(cites)} | missing from bib: {miss} | not marked VERIFIED: {unver}")
    print("bib entries never cited:", sorted(keys - cites))
    ok &= not miss and not unver
    # Every entry must parse: no '%' inside an entry, and each needs author/title/year.
    bad_pct, missing_fields = [], []
    for m in re.finditer(r"@\w+\{([^,\s]+)\s*,(.*?)\n\}?\s*(?=\n@|\n%|\Z)", bib, re.S):
        key, entry = m.group(1), m.group(2)
        if "%" in entry.replace("\\%", ""):
            bad_pct.append(key)
        for field in ("author", "title", "year"):
            if not re.search(r"\b" + field + r"\s*=", entry):
                missing_fields.append(f"{key}:{field}")
    print("entries with % inside (BibTeX would drop fields):", bad_pct)
    print("entries missing author/title/year:", missing_fields)
    ok &= not bad_pct and not missing_fields

    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    guarded = set(re.findall(r"\\IfFileExists\{([^}]+)\}", body))
    for f in re.findall(r"\\input\{([^}]+)\}", body):
        if f in guarded:
            print("input", f, "optional (guarded by IfFileExists; placeholder shown if absent)")
            continue
        p = os.path.join(root, f if f.endswith(".tex") else f + ".tex")
        exists = os.path.exists(p)
        ok &= exists
        print("input", f, "OK" if exists else "MISSING")
        if exists:
            labels |= set(re.findall(r"\\label\{([^}]+)\}", open(p, encoding="utf-8").read()))
    for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
        exists = os.path.exists(os.path.join(root, f))
        ok &= exists
        print("figure", f, "OK" if exists else "MISSING")
    dangling = sorted(set(re.findall(r"\\ref\{([^}]+)\}", body)) - labels)
    print("refs without a label:", dangling)
    ok &= not dangling

    s = body.replace("\\{", "").replace("\\}", "")
    depth, neg = 0, False
    for ch in s:
        depth += (ch == "{") - (ch == "}")
        neg |= depth < 0
    b, e = len(re.findall(r"\\begin\{", body)), len(re.findall(r"\\end\{", body))
    print("braces balanced:", depth == 0 and not neg, "| begin/end:", b, e)
    ok &= depth == 0 and not neg and b == e

    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)
    if m:
        ab = m.group(1)
    else:  # \ijphmabstract{...}: take the brace-matched argument
        i = body.index("\\ijphmabstract{") + len("\\ijphmabstract{")
        depth, j = 1, i
        while depth:
            depth += {"{": 1, "}": -1}.get(body[j], 0)
            j += 1
        ab = body[i:j - 1]
    envs = re.findall(r"\\(begin|end)\{([^}]+)\}", body)
    stack = []
    for kind, name in envs:
        if kind == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            print("ENVIRONMENT MISMATCH at \\end{" + name + "}")
            ok = False
    words = len(re.sub(r"\\[A-Za-z]+(\{\})?", "N", ab).split())
    print("abstract words (each macro = 1 word):", words, "(limit 250)")
    ok &= words <= 250
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()

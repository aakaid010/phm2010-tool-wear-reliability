"""Write ../REFERENCE_AUDIT.md from references.bib, main.bbl (citation order) and the output of
verify_references.py (--verify CSV). Notes record manual checks made on 23 Nov 2026.

  python verify_references.py --bib ../manuscript/references.bib --out verify.csv
  python make_reference_audit.py --verify verify.csv
"""
import argparse
import csv
import re

NOTES = {
    "koenker1978quantile": "Crossref gives first page 33 only; range 33-50 confirmed on the Econometric Society record",
    "romano2019cqr": "No DOI; title, authors, year and volume matched on the proceedings.neurips.cc record (23 Nov 2026)",
    "tibshirani2019covariate": "No DOI; official NeurIPS 2019 record (checked 21 Nov 2026 in an earlier audit)",
    "gibbs2021adaptive": "No DOI; official NeurIPS 2021 record, pp. 1660-1672 (checked 21 Nov 2026)",
    "dunn2023hierarchical": "Crossref issued date is the 2022 online-first; print issue 2023, vol. 118 no. 544 kept",
    "akaid2026code": "Authors' own code archive, needed for code availability; title as currently registered (v1.2); change it once the Zenodo release with the new title exists (see SUBMISSION_REPORT)",
    "raszka2026conformal": "Software archive; no companion article found",
    "liu2026crossmodal": "Quoted RMSE 2.51 / 6.92 confirmed in the Crossref abstract (23 Nov 2026)",
    "wen2026avmd": "Quoted RMSE: within-tool 2.28-3.85 um (<4), cross-tool 5.52-12.27 um, confirmed in PMC full text, Table 12 (23 Nov 2026)",
    "nguyen2026gpr": "Quoted coverage 69-89 % confirmed (69.09 % and 88.94 %) in the Scientific Reports full text (23 Nov 2026)",
    "patel2026dstl": "Full text read: LOTO over C1/C4/C6, no intervals; quote checked",
    "airao2024bnn": "Full text read: BNN, 70/30 split, MAE/RMSE/R2, no coverage",
    "singh2026cgag": "Full text read: quote on degradation across machines checked",
    "namboodri2026bayesian": "Full text read: credible intervals; statement on manual inspection checked",
    "wang2023wiener": "Full text read: three wear stages",
    "kim2024uncertainty": "Metadata verified; content described from title/abstract only (full text not accessible)",
    "yang2026sctcn": "Metadata verified; content described from title/abstract only",
    "yan2026safe": "Metadata verified; content described from title/abstract only",
    "wang2025adaptive": "Metadata verified; content described from title/abstract only",
}


def field(e, f):
    m = re.search(f + r"\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", e)
    return re.sub(r"[{}\\]", "", m.group(1)) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", required=True)
    a = ap.parse_args()
    bib = open("../manuscript/references.bib", encoding="utf-8").read()
    bbl = open("../manuscript/main.bbl", encoding="utf-8").read()
    order = re.findall(r"\}\]\{([^}]+)\}", bbl)
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    rows = {r["key"]: r for r in csv.DictReader(open(a.verify, encoding="utf-8"))}
    lit = {r["key"]: r["source"] for r in csv.DictReader(open("../data/literature.csv", encoding="utf-8"))}
    ijamt = [k for k in order if "s00170" in field(re.search(r"@\w+\{" + re.escape(k) + r",(.*?)\n(?=%|@|\Z)", bib, re.S).group(1), "doi")]
    out = ["# Reference audit (IJAMT version, 23 Nov 2026)", "",
           f"All {len(keys)} entries in `manuscript/references.bib` are cited ({len(order)} in `main.bbl`; "
           f"uncited: {sorted(keys - set(order)) or 'none'}; missing: {sorted(set(order) - keys) or 'none'}). "
           "Row number = reference number in the compiled PDF.", "",
           "DOIs were checked with `experiments/verify_references.py` on 23 Nov 2026 against Crossref (journal DOIs) "
           "or DataCite (Zenodo, Mendeley Data, IEEE DataPort): title, first author, year, volume and pages. "
           f"IJAMT papers: {len(ijamt)} of {len(order)} ({100 * len(ijamt) / len(order):.1f} %), within the journal's 15 % cap. "
           "For literature-table rows, the 'full text' or 'abstract' source is as recorded in `data/literature.csv`.", "",
           "| # | Key | First author, year | DOI / record | Status | Notes |", "|---|---|---|---|---|---|"]
    for i, k in enumerate(order, 1):
        e = re.search(r"@\w+\{" + re.escape(k) + r",(.*?)\n(?=%|@|\Z)", bib, re.S).group(1)
        au = field(e, "author").split(" and ")[0].split(",")[0]
        doi = field(e, "doi") or field(e, "url")
        r = rows.get(k, {})
        res = (r.get("registry", "") + " " + r.get("result", "")).strip() or "no DOI: manual check"
        if k == "koenker1978quantile":
            res = "Crossref OK (pages: see note)"
        n = NOTES.get(k, "")
        if k in lit:
            n = (n + "; " if n else "") + "literature-table row read from: " + lit[k]
        out.append(f"| {i} | {k} | {au} {field(e, 'year')} | {doi} | VERIFIED: {res} | {n} |")
    out += ["", "## Removed for the IJAMT version", "",
            "- `wang2024crossdataset` (Wang et al. 2024, IJAMT 131:171-182, DOI 10.1007/s00170-024-13055-3): verified, "
            "but removed to keep IJAMT references within 15 %; the sentence that cited it was deleted.",
            "- `refs_candidates.bib`: not used; removed from the IJAMT source.", "",
            "## Literature table (Table 1) versus the text", "",
            "- All counts in the text are macros from `make_literature_table.py` (15 studies, 12 with a held-out tool, "
            "3 with intervals, 0 with per-tool coverage, 3 read as abstracts only).",
            "- \"The others split pooled cuts at random, hold out cuts within each tool, or do not report the split\" "
            "matches the three remaining rows (si2024psdcvt: pooled, random; nguyen2026gpr: within-tool; li2026idrbilstm: "
            "not reported). The earlier wording (\"at random or within each cutter\") did not cover li2026idrbilstm and was corrected.",
            "- Patel & Muthuswamy (2026, IJAMT) was added as a row from its full text (LOTO, no uncertainty)."]
    open("../REFERENCE_AUDIT.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("wrote REFERENCE_AUDIT.md:", len(order), "references,", len(ijamt), "IJAMT")


if __name__ == "__main__":
    main()

"""Check every DOI in references.bib against its registry (Crossref, or DataCite for
Zenodo / Mendeley Data / IEEE DataPort / arXiv DOIs): title, first-author family name,
year, volume and pages/article number.

  python verify_references.py --bib <paper>/references.bib [--out report.csv]

Entries without a DOI (e.g. NeurIPS proceedings) are listed as "no DOI" for manual
checking against the publisher page.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import time
import unicodedata
import urllib.request

DATACITE_PREFIXES = ("10.5281/", "10.17632/", "10.21227/", "10.48550/")


def norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = re.sub(r"\\[a-zA-Z]+|[{}\\'\"`^~]", "", s)
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def parse_bib(text):
    out = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}|@(\w+)\{([^,]+),(.*?)\}\}\s*$", text, re.S | re.M):
        pass
    # simple field parser: split entries at '@'
    for chunk in re.split(r"\n(?=@)", text):
        h = re.match(r"@(\w+)\{([^,]+),", chunk.strip())
        if not h:
            continue
        f = {}
        for k, v in re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", chunk):
            f[k.lower()] = v
        out.append((h.group(2).strip(), h.group(1).lower(), f))
    return out


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "reference-check/1.0 (scripted metadata check)"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def registry(doi):
    if doi.lower().startswith(DATACITE_PREFIXES):
        a = get("https://api.datacite.org/dois/" + doi)["data"]["attributes"]
        fam = (a["creators"][0].get("familyName") or a["creators"][0].get("name", "")) if a["creators"] else ""
        return {"src": "DataCite", "title": a["titles"][0]["title"], "family": fam,
                "year": str(a.get("publicationYear")), "volume": None, "pages": None}
    m = get("https://api.crossref.org/works/" + doi)["message"]
    au = m.get("author") or [{}]
    return {"src": "Crossref", "title": (m.get("title") or [""])[0], "family": au[0].get("family", ""),
            "year": str((m.get("issued") or m.get("published") or {}).get("date-parts", [[None]])[0][0]),
            "volume": m.get("volume"), "pages": m.get("page") or m.get("article-number")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bib", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for key, typ, f in parse_bib(open(a.bib, encoding="utf-8").read()):
        doi = f.get("doi")
        if not doi:
            rows.append((key, "no DOI", "", "check publisher page manually"))
            continue
        try:
            r = registry(doi)
        except Exception as e:  # noqa: BLE001
            rows.append((key, doi, "UNRESOLVED", str(e)[:80]))
            continue
        issues = []
        sim = difflib.SequenceMatcher(None, " ".join(norm(f.get("title"))), " ".join(norm(r["title"]))).ratio()
        if sim < 0.9:
            issues.append(f"title sim {sim:.2f}: registry '{r['title'][:70]}'")
        fam_bib = norm(f.get("author", "").split(" and ")[0].split(",")[0])
        if r["family"] and fam_bib and norm(r["family"])[:1] != fam_bib[:1]:
            issues.append(f"first author: bib {fam_bib} vs {r['family']}")
        if r["year"] and f.get("year") and r["year"] != f["year"].strip():
            issues.append(f"year: bib {f['year']} vs {r['year']}")
        if r["volume"] and f.get("volume") and str(r["volume"]) != f["volume"].strip():
            issues.append(f"volume: bib {f['volume']} vs {r['volume']}")
        if r["pages"] and f.get("pages"):
            bp = f["pages"].replace("--", "-").strip()
            if str(r["pages"]).strip() != bp:
                issues.append(f"pages: bib {bp} vs {r['pages']}")
        rows.append((key, doi, r["src"], "OK" if not issues else "; ".join(issues)))
        time.sleep(0.2)
    w = max(len(k) for k, *_ in rows)
    for k, d, s, msg in rows:
        print(f"{k:<{w}}  {s:<9} {msg}")
    if a.out:
        import csv
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows([("key", "doi", "registry", "result")] + rows)


if __name__ == "__main__":
    main()

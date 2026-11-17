"""Resolve every candidate reference at its registry (Crossref for DOIs, arXiv
API for preprints) and write literature/refs_verified.json with the metadata
the reference list is built from. Nothing in the bibliography is typed by hand."""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "literature" / "refs_verified.json"
UA = {"User-Agent": "reference-check/1.0"}

REFS = [  # key, identifier
    ("pacheco2020data", "doi:10.1016/j.dib.2020.106221"),
    ("pacheco2020impact", "doi:10.1016/j.compbiomed.2019.103545"),
    ("esteva2017", "doi:10.1038/nature21056"),
    ("tschandl2018", "doi:10.1038/sdata.2018.161"),
    ("daneshjou2022", "doi:10.1126/sciadv.abq6147"),
    ("chuchu2018", "doi:10.1002/14651858.CD013193"),
    ("jakhar2020", "doi:10.1111/ced.14227"),
    ("koh2022", "doi:10.2196/35254"),
    ("biswas2026", "doi:10.1117/12.3095077"),
    ("cetinel2026", "doi:10.3390/diagnostics16162543"),
    ("shakir2026", "doi:10.3390/diagnostics16203368"),
    ("hu2024", "arxiv:2405.11289"),
    ("hendrycks2019", "arxiv:1903.12261"),
    ("dodge2016", "doi:10.1109/QoMEX.2016.7498955"),
    ("itu_t81", "manual:itu_t81"),
    ("paszke2019", "arxiv:1912.01703"),
    ("pedregosa2011", "arxiv:1201.0490"),
    ("he2016", "doi:10.1109/CVPR.2016.90"),
    ("tan2019", "arxiv:1905.11946"),
    ("oquab2023", "arxiv:2304.07193"),
    ("radford2021", "arxiv:2103.00020"),
    ("cherti2023", "doi:10.1109/CVPR52729.2023.00276"),
    ("guo2017", "arxiv:1706.04599"),
]


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def crossref(doi):
    m = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    issued = (m.get("issued", {}).get("date-parts") or [[None]])[0]
    return dict(type=m.get("type"), title=(m.get("title") or [""])[0], container=(m.get("container-title") or [""])[0],
                short_container=(m.get("short-container-title") or [""])[0],
                authors=[(a.get("family", ""), a.get("given", "")) for a in m.get("author", [])],
                year=issued[0], volume=m.get("volume"), issue=m.get("issue"), page=m.get("page") or m.get("article-number"),
                doi=m.get("DOI"), publisher=m.get("publisher"))


def arxiv(aid):
    x = get("http://export.arxiv.org/api/query?id_list=" + aid).decode("utf-8")
    ent = x.split("<entry>")[1]
    title = re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", ent, re.S).group(1)).strip()
    names = [re.sub(r"\s+", " ", n).strip() for n in re.findall(r"<name>(.*?)</name>", ent, re.S)]
    year = int(re.search(r"<published>(\d{4})", ent).group(1))
    authors = [(n.split()[-1], " ".join(n.split()[:-1])) for n in names]
    return dict(type="preprint", title=title, container="arXiv", authors=authors, year=year, arxiv=aid,
                doi="10.48550/arXiv." + aid)


# Published venue of an arXiv-identified work, each one read inside the
# downloaded PDF itself (literature/sources/<key>.md), never assumed.
VENUE = {
    "hendrycks2019": dict(type="proceedings-article", container="International Conference on Learning Representations (ICLR)", year=2019),
    "tan2019": dict(type="proceedings-article", container="Proceedings of the 36th International Conference on Machine Learning (ICML)", year=2019),
    "oquab2023": dict(type="journal-article", container="Transactions on Machine Learning Research", year=2024),
    "guo2017": dict(type="proceedings-article", container="Proceedings of the 34th International Conference on Machine Learning (ICML), PMLR 70", year=2017),
    "paszke2019": dict(type="proceedings-article", container="Advances in Neural Information Processing Systems 32 (NeurIPS 2019)", year=2019),
    "pedregosa2011": dict(type="journal-article", container="Journal of Machine Learning Research", volume="12", page="2825-2830", year=2011),
}
MANUAL = {  # no DOI; read from the downloaded document's own title page
    "itu_t81": dict(type="standard", title="Information technology - Digital compression and coding of continuous-tone "
                    "still images - Requirements and guidelines", authors=[("CCITT", "")],
                    container="Recommendation T.81", publisher="International Telecommunication Union", year=1992,
                    url="https://www.w3.org/Graphics/JPEG/itu-t81.pdf"),
}

out, bad = {}, []
for key, ident in REFS:
    kind, val = ident.split(":", 1)
    try:
        out[key] = dict(MANUAL[val]) if kind == "manual" else (crossref if kind == "doi" else arxiv)(val)
        out[key].update(VENUE.get(key, {}))
        out[key]["id"] = ident
        a = out[key]["authors"]
        print(f"OK  {key:16s} {out[key]['year']} {a[0][0] if a else '?'}: {out[key]['title'][:70]}")
    except Exception as e:
        bad.append((key, ident, str(e)))
        print(f"BAD {key:16s} {ident}: {e}")
    time.sleep(0.4)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(len(out), "verified,", len(bad), "failed")
sys.exit(1 if bad else 0)

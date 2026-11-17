"""Download the open-access PDF of each cited work into literature/pdfs/,
check it is a PDF, and record SHA-256 + source URL in literature/manifest.json.
Works without a free copy are listed as MISSING for a manual download; they are
not cited until a copy exists."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

LIT = Path(__file__).resolve().parents[1] / "literature"
PDF = LIT / "pdfs"
PDF.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (literature-fetch for a research paper)"}

SOURCES = {
    "hendrycks2019": "https://arxiv.org/pdf/1903.12261",
    "tan2019": "https://arxiv.org/pdf/1905.11946",
    "oquab2023": "https://arxiv.org/pdf/2304.07193",
    "radford2021": "https://arxiv.org/pdf/2103.00020",
    "guo2017": "https://arxiv.org/pdf/1706.04599",
    "hu2024": "https://arxiv.org/pdf/2405.11289",
    "dodge2016": "https://arxiv.org/pdf/1604.04004",
    "pacheco2020impact": "https://arxiv.org/pdf/1909.12912",
    "he2016": "https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf",
    "cherti2023": "https://openaccess.thecvf.com/content/CVPR2023/papers/Cherti_Reproducible_Scaling_Laws_for_Contrastive_Language-Image_Learning_CVPR_2023_paper.pdf",
    "tschandl2018": "https://www.nature.com/articles/sdata2018161.pdf",
    "yu2026": "https://www.nature.com/articles/s41598-026-64721-3.pdf",
    "daneshjou2022": "https://www.science.org/doi/pdf/10.1126/sciadv.abq6147",
    "koh2022": "https://derma.jmir.org/2022/2/e35254/PDF",
    "cetinel2026": "https://www.mdpi.com/2075-4418/16/16/2543/pdf",
    "shakir2026": "https://www.mdpi.com/2075-4418/16/20/3368/pdf",
    "back2021": "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9335571",
    "pacheco2020data": "https://www.sciencedirect.com/science/article/pii/S235234092030115X/pdfft",
    "paszke2019": "https://arxiv.org/pdf/1912.01703",
    "pedregosa2011": "https://arxiv.org/pdf/1201.0490",
    "itu_t81": "https://www.w3.org/Graphics/JPEG/itu-t81.pdf",
}

man_p = LIT / "manifest.json"
man = json.loads(man_p.read_text()) if man_p.exists() else {}
for key, url in SOURCES.items():
    out = PDF / f"{key}.pdf"
    if out.exists() and key in man:
        continue
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
            data = r.read()
        if not data.startswith(b"%PDF"):
            raise ValueError(f"not a PDF ({data[:15]!r})")
        out.write_bytes(data)
        man[key] = dict(file=out.name, url=url, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                        fetched="2026-11-12")
        print(f"OK      {key} {len(data) // 1024} KB")
    except Exception as e:
        print(f"MISSING {key}: {e}")
    time.sleep(1.5)
man_p.write_text(json.dumps(man, indent=1))
print(len(man), "PDFs on disk")

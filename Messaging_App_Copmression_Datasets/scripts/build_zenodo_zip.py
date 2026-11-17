"""Build the Zenodo deposit for this paper: code, split assignments, every
result table, per-image predictions and the 120 WhatsApp round-trip files.
No original PAD-UFES-20 images are included (they are cited by DOI); the
received WhatsApp files are derivatives released under the dataset's CC BY 4.0
terms with attribution in the README.

Output: deposit/zenodo_deposit.zip and deposit/zenodo_metadata.md (title,
description, keywords, licence) for whoever uploads it.
"""
import hashlib
import zipfile
from pathlib import Path

P = Path(__file__).resolve().parents[1]
OUT = P / "deposit"
OUT.mkdir(exist_ok=True)
TITLE = ("Code, predictions and WhatsApp round-trip files for: Does Messaging-App Compression Degrade Skin Lesion "
         "Classification from Smartphone Photographs?")
README = f"""# {TITLE}

Contents
- code/      experiment pipeline (common.py, precompute.py, train_finetune.py, frozen.py, analyze.py,
             eval_whatsapp.py, make_figures.py, make_tables.py) and tests
- scripts/   one-off tools (sample selection for the WhatsApp round trip, matching, duplicate audit)
- results/   every table and statistic reported in the paper; results/splits.csv holds the
             patient-grouped fold of every image; results/preds/*.npz hold the softmax outputs of every
             model, regime, seed and fold for all 17 compression conditions
- whatsapp/received/  the 120 photographs as saved back from WhatsApp Desktop (Windows, standard quality),
             renamed to their PAD-UFES-20 image id; results/whatsapp_roundtrip.csv describes them

Data source: PAD-UFES-20 (Pacheco et al., Mendeley Data, https://doi.org/10.17632/zr7vgbcyr2.1, CC BY 4.0).
The original images are not redistributed here; download them from the DOI above into data/ to re-run.
The files in whatsapp/received are re-encoded copies of 120 PAD-UFES-20 images, shared under the same
CC BY 4.0 licence with attribution to the dataset authors.

Reproduce: pip install -r code/requirements.txt; python code/precompute.py; python code/run_all.py;
python code/eval_whatsapp.py; python code/make_figures.py; python code/make_tables.py
Hardware used: one NVIDIA RTX 4050 Laptop GPU (6 GB).
"""


def main():
    files = []
    for sub, pats in [("code", ["*.py", "requirements.txt"]), ("scripts", ["*.py"]),
                      ("results", ["*.csv", "*.json", "*.txt"]), ("results/preds", ["*.npz"])]:
        for pat in pats:
            files += [(f, f"{sub}/{f.name}") for f in sorted((P / sub).glob(pat))]
    files += [(f, f"whatsapp/received/{f.name}") for f in sorted((P / "data" / "whatsapp" / "matched").glob("*.jpeg"))]
    zp = OUT / "zenodo_deposit.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.md", README)
        for f, arc in files:
            z.write(f, arc)
    h = hashlib.sha256(zp.read_bytes()).hexdigest()
    (OUT / "zenodo_metadata.md").write_text(
        f"# Zenodo upload sheet\n\n- **Upload type:** Dataset (software + data)\n- **Title:** {TITLE}\n"
        "- **Authors:** same order as the manuscript byline\n- **Licence:** CC BY 4.0\n"
        "- **Keywords:** teledermatology; image compression; JPEG; WhatsApp; skin lesion classification; PAD-UFES-20\n"
        "- **Related identifier:** is derived from https://doi.org/10.17632/zr7vgbcyr2.1\n"
        "- **Description:** paste README.md from the zip\n"
        f"- **File:** zenodo_deposit.zip ({zp.stat().st_size // 1024} KB, SHA-256 {h})\n\n"
        "After publishing, put the DOI into draft/07-declarations.md (two highlighted places) and rebuild.\n",
        encoding="utf-8")
    print(f"{zp} {zp.stat().st_size // 1024} KB, {len(files)} files")


if __name__ == "__main__":
    main()

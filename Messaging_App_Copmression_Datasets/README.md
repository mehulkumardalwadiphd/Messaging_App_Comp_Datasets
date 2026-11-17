# Code, predictions and WhatsApp round-trip files for: Does Messaging-App Compression Degrade Skin Lesion Classification from Smartphone Photographs?

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

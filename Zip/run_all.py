"""Run every stage in one High-priority process: fine-tuned CNNs, frozen
encoders, then the analysis. Each stage resumes from its own per-job files."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import psutil
    psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
except Exception as e:  # priority is a convenience, never a reason to stop
    print("could not raise priority:", e)

import analyze, frozen, train_finetune  # noqa: E402

t0 = time.time()
sys.argv = [sys.argv[0]]
for name, mod in [("finetune", train_finetune), ("frozen", frozen), ("analyze", analyze)]:
    train_finetune.log(f"[run_all] stage {name} start")
    mod.main()
train_finetune.log(f"[run_all] DONE in {(time.time() - t0) / 60:.0f} min")

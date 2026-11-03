# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

from __future__ import annotations
import json, os, time
from pathlib import Path
REQUIRED = {"config.json","environment.json","support_manifest.json","support_manifest.sha256",
"validation_manifest.json","validation_manifest.sha256","full_reference_manifest.json",
"full_reference_manifest.sha256","diagnostic_manifest_b16.json","diagnostic_manifest_b16.sha256",
"training_order_hashes.json","epoch_metrics.csv","update_metrics.csv.gz","diagnostic_global.csv",
"diagnostic_encoder.csv","diagnostic_block.csv","diagnostic_histograms.npz","summary.json","stdout.log","DONE.json"}

def validate_done(run_dir: Path):
    missing = sorted(x for x in REQUIRED if not (run_dir / x).is_file())
    if missing: return False, missing
    try:
        done = json.loads((run_dir / "DONE.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False, ["malformed DONE.json"]
    if done.get("final_epoch") != 30 or done.get("expected_diagnostic_checkpoints") != done.get("observed_diagnostic_checkpoints"):
        return False, ["invalid DONE.json"]
    return True, []

def claim(path: Path, stale_seconds=86400):
    path.parent.mkdir(parents=True, exist_ok=True)
    try: fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if time.time() - path.stat().st_mtime <= stale_seconds: return False
        os.replace(path, path.with_suffix(f".stale.{int(time.time())}"))
        return claim(path, stale_seconds)
    with os.fdopen(fd, "w") as stream: stream.write(f"{os.getpid()}\n")
    return True

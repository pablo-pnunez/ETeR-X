"""Free disk space by deleting model weights that are not needed to reproduce results.

Dry-run by default; pass --apply to actually delete. What is removed:

  A. dev/weights.*            Dev-phase (validation) weights, in every run. Only
                              dev/log.csv etc. are read afterwards (CollectResults.py);
                              final evaluation loads the root-level `weights`.
  B. _pre_reviewid_fix_backup Obsolete backups made before the reviewId fix.
  C. weights.* (root)         Final weights of runs that are NOT a "keep" run. A run is
                              kept when its md5 is in models/best_models.csv or is
                              referenced by any file in results/, scripts/ or root *.py.

Logs, cfg.json, emissions and *eval*.csv files are never touched.
"""
import argparse
import glob
import os
import re
import shutil

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
MD5 = re.compile(r"\b[0-9a-f]{32}\b")


def size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)


def keep_md5s():
    keep = set(pd.read_csv(os.path.join(MODELS, "best_models.csv")).model_md5)
    files = glob.glob(os.path.join(ROOT, "results", "**", "*"), recursive=True)
    files += glob.glob(os.path.join(ROOT, "scripts", "**", "*.sh"), recursive=True)
    files += glob.glob(os.path.join(ROOT, "*.py"))
    for f in files:
        if os.path.isfile(f):
            with open(f, errors="ignore") as fh:
                keep |= set(MD5.findall(fh.read()))
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="delete (default: dry-run)")
    args = ap.parse_args()

    keep = keep_md5s()
    targets = {"A dev weights": [], "B reviewid backups": [], "C unused final weights": []}
    for cfg in glob.glob(os.path.join(MODELS, "*", "*", "*", "*", "cfg.json")):
        run = os.path.dirname(cfg)
        targets["A dev weights"] += glob.glob(os.path.join(run, "dev", "weights*"))
        bak = os.path.join(run, "_pre_reviewid_fix_backup")
        if os.path.isdir(bak):
            targets["B reviewid backups"].append(bak)
        if os.path.basename(run) not in keep:
            targets["C unused final weights"] += glob.glob(os.path.join(run, "weights*"))

    total = 0
    for name, paths in targets.items():
        s = sum(size(p) for p in paths)
        total += s
        print(f"{name:26s} {len(paths):5d} items  {s / 1e9:8.1f} GB")
    print(f"{'TOTAL':26s} {'':5s}        {total / 1e9:8.1f} GB  (kept md5s: {len(keep)})")

    if not args.apply:
        print("Dry-run only. Re-run with --apply to delete.")
        return
    for paths in targets.values():
        for p in paths:
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    print("Done.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Retrains USEM2ITM/BERT2ITM on the 10 TripAdvisor datasets with the post-fix dataset cache (same
train/test split as ETeR-X), one process per dev/final phase, and repeats a phase in float32 if the
mixed-precision run diverges (NaN loss, seen on USEM/Restaurants-Gijon).

Usage: python scripts/retrain_correct_cache.py BERT2ITM [GPU]
"""
import csv, os, shutil, subprocess, sys, time

model = sys.argv[1]
gpu = sys.argv[2] if len(sys.argv) > 2 else "0"
PY = "/media/nas/pperez/conda/ns3/envs/ETeR-X/bin/python"
OUT = f"scripts/out/retrain_{model[:4].lower()}"
os.makedirs(OUT, exist_ok=True)
ORDER = [("pois", "paris"), ("pois", "newyorkcity"), ("pois", "barcelona"), ("pois", "london"), ("pois", "madrid"),
         ("restaurants", "gijon"), ("restaurants", "barcelona"), ("restaurants", "madrid"),
         ("restaurants", "paris"), ("restaurants", "newyorkcity")]
best = list(csv.DictReader(open("models/best_models.csv")))
md5 = lambda m, d, s: [r for r in best if r["dataset"] == d and r["subset"] == s and r["model"] == m][0]["model_md5"]


def log(msg):
    with open(f"{OUT}/driver.log", "a") as f:
        f.write(f"{msg} {time.strftime('%a %d %b %H:%M:%S')}\n")


def has_nan(path):
    if not os.path.exists(path):
        return True
    return any("nan" in l.lower() for l in open(path).read().splitlines()[1:])


def run(d, s, phase, fp32):
    old, bow = md5(model, d, s), md5("BOW2ITM", d, s)
    tag = f"{d}_{s}_{phase}" + ("_fp32" if fp32 else "")
    cmd = [PY, "-u", "legacy/run_one_retrain_tripadvisor_correct_cache.py", "--model", model, "--dataset", d, "--subset", s,
           "--old-md5", old, "--bow-md5", bow, "--phase", phase, "--gpu", gpu] + (["--fp32"] if fp32 else [])
    with open(f"{OUT}/{tag}.txt", "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    # locate the new model folder from the log
    mp = None
    for l in open(f"{OUT}/{tag}.txt"):
        if l.startswith("MODEL_PATH:"):
            mp = l.split(":", 1)[1].strip()
    return rc, mp


for d, s in ORDER:
    log(f"### {d} {s}")
    for phase in ("dev", "final"):
        for fp32 in (False, True):
            rc, mp = run(d, s, phase, fp32)
            logcsv = (mp + ("dev/" if phase == "dev" else "") + "log.csv") if mp else None
            bad = rc != 0 or (logcsv is not None and has_nan(logcsv))
            log(f"   {phase} fp32={fp32} exit={rc} nan_or_failed={bad}")
            if not bad:
                break
            if fp32:
                log(f"   !! {phase} failed also in fp32; skipping this dataset")
                break
            # move the diverged outputs aside before retrying in fp32
            bk = mp + f"_nan_fp16_{phase}_backup/"
            os.makedirs(bk, exist_ok=True)
            src = mp + ("dev/" if phase == "dev" else "")
            for fn in ("checkpoint", "log.csv", "emissions_train.csv", "history.jpg", "weights.index", "weights.data-00000-of-00001"):
                if os.path.exists(src + fn):
                    shutil.move(src + fn, bk + fn)
        else:
            continue
        if bad:
            break
log("ALLDONE")

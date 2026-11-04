#!/bin/bash
# Retrains USEM2ITM on the 10 TripAdvisor datasets with the post-fix dataset cache (same split as ETeR-X).
# Each dev/final phase runs in its own process. Usage: legacy/RetrainUSEMCorrectCache.sh [GPU]
GPU=${1:-0}
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
MAP=${MAPPING:-models/best_models.csv}
ORDER="pois,paris pois,newyorkcity pois,barcelona pois,london pois,madrid restaurants,gijon restaurants,barcelona restaurants,madrid restaurants,paris restaurants,newyorkcity"
for ds in $ORDER; do
  d=${ds%,*}; s=${ds#*,}
  old=$(grep "^[0-9]*,$d,$s,USEM2ITM," models/best_models.csv | head -1 | cut -d, -f11)
  bow=$(grep "^[0-9]*,$d,$s,BOW2ITM," models/best_models.csv | head -1 | cut -d, -f11)
  echo "### $d $s old=$old bow=$bow $(date)"
  for phase in dev final; do
    $PY -u legacy/run_one_retrain_tripadvisor_correct_cache.py --model USEM2ITM --dataset $d --subset $s --old-md5 $old --bow-md5 $bow --phase $phase --gpu $GPU > scripts/out/retrain_usem/${d}_${s}_${phase}.txt 2>&1
    echo "   $phase exit=$? $(date)"
  done
done
echo ALLDONE

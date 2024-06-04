#!/bin/bash

MAXTSTS=1
GPU=0

declare -A DATASETS
declare -A MODELS
# DATASETS["restaurants"]="gijon barcelona madrid newyorkcity paris"
# DATASETS["pois"]="barcelona madrid newyorkcity paris london"
# DATASETS["amazon"]="digital_music fashion"

DATASETS["restaurants"]="gijon barcelona madrid newyorkcity paris"
DATASETS["pois"]="barcelona madrid newyorkcity paris london"
DATASETS["amazon"]="digital_music fashion"

MODELS["BOW2ITM"]="" 
# MODELS["ATT2ITM"]="" 
# MODELS["USEM2ITM"]="" 
# MODELS["BERT2ITM"]="" 
# MODELS["ATT2ITM_2"]="" 

for DATASET_NAME in ${!DATASETS[@]}; do 
  for SUBSET_NAME in ${DATASETS[$DATASET_NAME]}; do
    echo "[$DATASET_NAME] -> $SUBSET_NAME"
    for MODEL_NAME in ${!MODELS[@]}; do 
      echo "  ╚═ $MODEL_NAME"

      TXT_PATH="scripts/out/"$DATASET_NAME"/$SUBSET_NAME/final/"
      mkdir -p $TXT_PATH

      nohup /media/nas/pperez/conda/ns3/envs/ETeR-X/bin/python -u FinalTrain.py -model $MODEL_NAME -dataset $DATASET_NAME -subset $SUBSET_NAME > "$TXT_PATH"$MODEL_NAME".txt" &

      echo "═══════ [$!] $MODEL_NAME ══════"

      # Si se alcanza el máximo de procesos simultaneos, esperar
      while [ $(jobs -r | wc -l) -eq $MAXTSTS ];
      do
        sleep 5
      done

      #Esperar X segundos entre pruebas para que le de tiempo a ocupar memoria en GPU
      sleep 10

    done  
  done
done


# Esperar por los últimos
while [ $(jobs -r | wc -l) -gt 0 ];
do
  sleep 5
done


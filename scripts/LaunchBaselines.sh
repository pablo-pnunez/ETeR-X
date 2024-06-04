#!/bin/bash

MAXTSTS=2
GPU=0

declare -A DATASETS

DATASETS["restaurants"]="gijon barcelona madrid newyorkcity paris"
DATASETS["pois"]="barcelona madrid newyorkcity paris london"
DATASETS["amazon"]="digital_music fashion" 
# DATASETS["restaurants"]="paris"


for DATASET_NAME in ${!DATASETS[@]}; do 
  for SUBSET_NAME in ${DATASETS[$DATASET_NAME]}; do
    echo "[$DATASET_NAME] -> $SUBSET_NAME"   
            
    # source /media/nas/pperez/miniconda3/etc/profile.d/conda.sh
    # conda activate TAV_text

    nohup /media/nas/pperez/conda/ns3/envs/ETeR-X/bin/python -u Baselines.py -gpu $GPU -dst $DATASET_NAME -sst $SUBSET_NAME &
  
    GPU=$(($(($GPU+1%2))%2))

    # Si se alcanza el máximo de procesos simultaneos, esperar
    while [ $(jobs -r | wc -l) -eq $MAXTSTS ];
    do
      sleep 5
    done

    #Esperar X segundos entre pruebas para que le de tiempo a ocupar memoria en GPU
    sleep 25

  done
done

# Esperar por los últimos
while [ $(jobs -r | wc -l) -gt 0 ];
do
  sleep 5
done


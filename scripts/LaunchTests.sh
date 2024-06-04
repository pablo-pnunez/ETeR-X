#!/bin/bash

MAXTSTS=2
STAGE=0 # GRIDSEARCH o TRAIN
GPU=1
ESP=10 # Early stop patience (50 se utilizó en modelos 0 y 1)

declare -A DATASETS

DATASETS["restaurants"]="gijon barcelona madrid newyorkcity paris"
DATASETS["pois"]="barcelona madrid newyorkcity paris london"
DATASETS["amazon"]="digital_music fashion"

# DATASETS["restaurants"]="madrid newyorkcity paris"
# DATASETS["amazon"]="digital_music fashion"
# DATASETS["pois"]="barcelona madrid newyorkcity paris london"

declare -A MODELS
declare -A BATCHES
declare -A LRATES

MODELS["MOSTPOP2ITM"]="0" # DA IGUAL PARA ESTE MODELO
BATCHES["MOSTPOP2ITM"]="256" # DA IGUAL PARA ESTE MODELO
LRATES["MOSTPOP2ITM"]="0.0" # DA IGUAL PARA ESTE MODELO

# MODELS["ATT2ITM_2"]="0" 
# BATCHES["ATT2ITM_2"]="512 1024"
# LRATES["ATT2ITM_2"]="1e-5 5e-5 1e-4 5e-4"

# MODELS["BOW2ITM"]="0" 
# BATCHES["BOW2ITM"]="256 512 1024 2048 4096"
# LRATES["BOW2ITM"]="1e-5 5e-5 1e-4 5e-4 1e-3" 

# MODELS["ATT2ITM"]="0 1" 
# BATCHES["ATT2ITM"]="512 1024 2048"
# LRATES["ATT2ITM"]="1e-5 5e-5 1e-4 5e-4" 

# MODELS["ATT2ITM"]="0" 
# BATCHES["ATT2ITM"]="256 512 1024 2048 4096"
# LRATES["ATT2ITM"]="5e-6 1e-5 5e-5 1e-4 5e-4" 

# MODELS["ATT2ITM"]="2" 
# BATCHES["ATT2ITM"]="64 128 256"
# LRATES["ATT2ITM"]="5e-5 1e-4 5e-4" 

# MODELS["USEM2ITM"]="0" 
# BATCHES["USEM2ITM"]="512 1024 2048"
# LRATES["USEM2ITM"]="1e-5 5e-5 1e-4 5e-4 1e-3 5e-3" 

# MODELS["BERT2ITM"]="0" 
# BATCHES["BERT2ITM"]="512 1024"
# LRATES["BERT2ITM"]="5e-5 1e-4 5e-4 1e-3" 

for DATASET_NAME in ${!DATASETS[@]}; do 
  for SUBSET_NAME in ${DATASETS[$DATASET_NAME]}; do
    echo "[$DATASET_NAME] -> $SUBSET_NAME"
    for MODEL_NAME in ${!MODELS[@]}; do 
      echo "  ╚═ $MODEL_NAME"
      for MODEL_VERSION in ${MODELS[$MODEL_NAME]}; do
        echo "   ╚═ $MODEL_VERSION"
        for BATCH in ${BATCHES[$MODEL_NAME]}; do
          echo "    ╚═ $BATCH"
          for LRATE in ${LRATES[$MODEL_NAME]}; do
            echo "     ╚═ $LRATE"
            
            TXT_PATH="scripts/out/"$DATASET_NAME"/$SUBSET_NAME/"
            mkdir -p $TXT_PATH

            # source /media/nas/pperez/miniconda3/etc/profile.d/conda.sh
            # conda activate TAV_text

            nohup /media/nas/pperez/conda/ns3/envs/ETeR-X/bin/python -u Main.py -stg $STAGE -mn $MODEL_NAME -dst $DATASET_NAME -sst $SUBSET_NAME -mv $MODEL_VERSION -esp $ESP -bs $BATCH -lr $LRATE >> "$TXT_PATH"$MODEL_NAME"_["$MODEL_VERSION"]_"$BOWNWORDS"_("$BATCH"_"$LRATE").txt" &
          
            # GPU=$(($(($GPU+1%2))%2))

            echo "═══════ [$!] $MODEL_NAME ══════"

            # Si se alcanza el máximo de procesos simultaneos, esperar
            while [ $(jobs -r | wc -l) -eq $MAXTSTS ];
            do
              sleep 5
            done

            #Esperar X segundos entre pruebas para que le de tiempo a ocupar memoria en GPU
            sleep 25

          done
        done
      done
    done  
  done
done

# Esperar por los últimos
while [ $(jobs -r | wc -l) -gt 0 ];
do
  sleep 5
done


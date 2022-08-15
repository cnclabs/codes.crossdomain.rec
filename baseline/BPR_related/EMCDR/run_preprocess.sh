#!/bin/bash
set -xe

update_times=200 
num_checkpoint=1
split=$(($update_times/$num_checkpoint))

declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=10)

for d in "${datasets[@]}"; do
    mkdir $d

	IFS='_'
	read -a domains <<< "$d"
	IFS=' '
	src=${domains[0]}
	tar=${domains[1]}
	ncore=${ncores[$d]}

    for checkpoint in $(seq 1 $num_checkpoint); do
        current_split=$(($checkpoint*$split))
        
        python3 preprocess.py \
        --dataset_name $d \
        --current_epoch $current_split \
        --ncore $ncore

        python3 preprocess_cold.py \
        --current_epoch $current_split \
        --dataset_name $d \
        --ncore $ncore
    done
done

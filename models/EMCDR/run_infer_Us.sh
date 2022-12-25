#!/bin/bash

mom_save_dir=$1
update_times=1000 
num_checkpoint=1
split=$((200/$num_checkpoint))

declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=5)

for d in "${datasets[@]}"; do
    IFS='_'
	read -a domains <<< "$d"
	IFS=' '
	src=${domains[0]}
	tar=${domains[1]}
	ncore=${ncores[$d]}

    for checkpoint in $(seq 1 $num_checkpoint); do
        current_split=$(($checkpoint*$split))

        python3 infer_Us.py \
	--mom_save_dir ${mom_save_dir} \
        --meta_path ${d}/model_${current_split}/mlp_epoch_${update_times}/mlp_epoch_${update_times}.meta \
        --ckpt_path ${d}/model_${current_split}/mlp_epoch_${update_times} \
        --dataset_name ${d} \
        --current_epoch $current_split \
        --ncore $ncore

        # cold
        python3 infer_Us_cold.py \
	--mom_save_dir ${mom_save_dir} \
        --meta_path ${d}/model_cold_${current_split}/mlp_epoch_${update_times}/mlp_epoch_${update_times}.meta \
        --ckpt_path ${d}/model_cold_${current_split}/mlp_epoch_${update_times} \
        --dataset_name ${d} \
        --current_epoch $current_split \
        --ncore $ncore
    done
done

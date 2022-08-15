#!/bin/bash

set -xe
update_times=200 
num_checkpoint=1
split=$(($update_times/$num_checkpoint))

declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=10)

for d in "${datasets[@]}"; do
	IFS='_'
	read -a domains <<< "$d"
	IFS=' '
	src=${domains[0]}
	tar=${domains[1]}
	ncore=${ncores[$d]}

    for checkpoint in $(seq 1 $num_checkpoint); do
        current_split=$(($checkpoint*$split))
        # ======== KK ========
        python3 MLP.py \
        --epoch_log ./${d}/model_log/MLP_lightfm_tv_vod.txt \
        --model_save_dir ./${d}/model_${current_split} \
        --Us ./${d}/lightfm_bpr_Us_${current_split}.pickle \
        --Ut ./${d}/lightfm_bpr_Ut_${current_split}.pickle

        # cold
        python3 MLP.py \
        --epoch_log ./${d}/model_log_cold/MLP_lightfm_tv_vod_cold.txt \
        --model_save_dir ./${d}/model_cold_${current_split} \
        --Us ./${d}/lightfm_bpr_Us_cold_${current_split}.pickle \
        --Ut ./${d}/lightfm_bpr_Ut_cold_${current_split}.pickle
    done
done

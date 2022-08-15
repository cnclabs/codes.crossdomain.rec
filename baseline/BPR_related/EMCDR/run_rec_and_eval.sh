#!/bin/bash
set -xe

update_times=200 
method="EMCDR"

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

    python3 rec_and_eval_ncore_EMCDR.py \
    --test_users target \
    --output_file ./result/${d}_target_result_${method}_${update_times}.txt \
    --current_epoch $update_times \
    --workers 8 \
    --dataset_name $d \
    --ncore $ncore

    python3 rec_and_eval_ncore_EMCDR.py \
    --test_users shared \
    --output_file ./result/${d}_shared_result_${method}_${update_times}.txt \
    --current_epoch $update_times \
    --workers 8 \
    --dataset_name $d \
    --ncore $ncore

    python3 rec_and_eval_ncore_cold_EMCDR.py \
    --output_file ./result/${d}_cold_result_${method}_${update_times}.txt \
    --current_epoch $update_times \
    --workers 8 \
    --dataset_name $d \
    --ncore $ncore
done
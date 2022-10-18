#!/bin/bash
set -xe

mom_save_dir=/TOP/tmp2/cpr/fix_ncore
update_times=200 
method="EMCDR"

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

    python3 rec_and_eval_ncore_EMCDR.py \
    --mom_save_dir ${mom_save_dir} \
    --test_users target \
    --output_file ./result/${d}_target_result_${method}_${update_times}.txt \
    --src ${src} \
    --tar ${tar} \
    --current_epoch $update_times \
    --workers 8 \
    --dataset_name $d \
    --ncore $ncore

    python3 rec_and_eval_ncore_EMCDR.py \
    --mom_save_dir ${mom_save_dir} \
    --test_users shared \
    --output_file ./result/${d}_shared_result_${method}_${update_times}.txt \
    --src ${src} \
    --tar ${tar} \
    --current_epoch $update_times \
    --workers 8 \
    --dataset_name $d \
    --ncore $ncore

    python3 rec_and_eval_ncore_cold_EMCDR.py \
    --mom_save_dir ${mom_save_dir} \
    --output_file ./result/${d}_cold_result_${method}_${update_times}.txt \
    --src ${src} \
    --tar ${tar} \
    --current_epoch $update_times \
    --workers 8 \
    --dataset_name $d \
    --ncore $ncore
done

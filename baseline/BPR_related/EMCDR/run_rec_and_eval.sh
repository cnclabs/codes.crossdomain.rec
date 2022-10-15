#!/bin/bash
set -xe

data_dir=$1
exp_record_dir=$2
model_name=emcdr
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

    python3 ../../../rec_and_eval_ncore.py \
    --data_dir ${data_dir} \
    --test_mode target \
    --save_dir ${exp_record_dir} \
    --save_name M_${model_name}_D_${src}_${tar}_T_target \
    --output_file ./result/${d}_target_result_${method}_${update_times}.txt \
    --src ${src} \
    --tar ${tar} \
    --n_worker 8 \
    --ncore $ncore \
    --model_name ${model_name} \
    --item_emb_path /TOP/home/ythuang/CODE/tmp/refactor_eval/codes.crossdomain.rec/baseline/BPR_related/lfm_bpr_graphs/${tar}_lightfm_bpr_${update_times}_10e-5.txt \
    --user_emb_path_shared /TOP/home/ythuang/CODE/tmp/refactor_eval/codes.crossdomain.rec/baseline/BPR_related/EMCDR/${src}_${tar}/shared_users_mapped_emb_dict_${update_times}.pickle\
    --user_emb_path_target /TOP/home/ythuang/CODE/tmp/refactor_eval/codes.crossdomain.rec/baseline/BPR_related/lfm_bpr_graphs/${tar}_lightfm_bpr_${update_times}_10e-5.txt

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

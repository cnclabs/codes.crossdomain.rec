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

echo pwd

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
    --src ${src} \
    --tar ${tar} \
    --n_worker 8 \
    --ncore $ncore \
    --model_name ${model_name} \
    --item_emb_path $(pwd)/../lfm_bpr_graphs/${tar}_lightfm_bpr_${update_times}_10e-5.txt \
    --user_emb_path_shared $(pwd)/${src}_${tar}/shared_users_mapped_emb_dict_${update_times}.pickle\
    --user_emb_path_target $(pwd)/../lfm_bpr_graphs/${tar}_lightfm_bpr_${update_times}_10e-5.txt


    python3 ../../../rec_and_eval_ncore.py \
    --data_dir ${data_dir} \
    --test_mode shared \
    --save_dir ${exp_record_dir} \
    --save_name M_${model_name}_D_${src}_${tar}_T_shared \
    --src ${src} \
    --tar ${tar} \
    --n_worker 8 \
    --ncore $ncore\
    --model_name ${model_name} \
    --item_emb_path $(pwd)/../lfm_bpr_graphs/${tar}_lightfm_bpr_${update_times}_10e-5.txt \
    --user_emb_path_shared $(pwd)/${src}_${tar}/shared_users_mapped_emb_dict_${update_times}.pickle\
    --user_emb_path_target $(pwd)/../lfm_bpr_graphs/${tar}_lightfm_bpr_${update_times}_10e-5.txt


    python3 ../../../rec_and_eval_ncore.py \
    --data_dir ${data_dir} \
    --test_mode cold \
    --save_dir ${exp_record_dir} \
    --save_name M_${model_name}_D_${src}_${tar}_T_cold \
    --src ${src} \
    --tar ${tar} \
    --n_worker 8 \
    --ncore $ncore \
    --model_name emcdr\
    --item_emb_path $(pwd)/../lfm_bpr_graphs/cold_${tar}_lightfm_bpr_${update_times}_10e-5.txt \
    --user_emb_path_cold $(pwd)/${src}_${tar}/cold_users_mapped_emb_dict_${update_times}.pickle
done

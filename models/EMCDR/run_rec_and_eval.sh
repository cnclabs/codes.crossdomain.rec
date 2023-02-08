#!/bin/bash
ncore_data_dir=$1
exp_record_dir=$2
emcdr_emb_dir=$3
bpr_emb_dir=$4
model_name=emcdr

declare -a datasets=("hk_csjj" "spo_csj" "mt_b")

for d in "${datasets[@]}"; do
    IFS='_'
    read -a domains <<< "$d"
    IFS=' '
    src=${domains[0]}
    tar=${domains[1]}

    python3 ../../rec_and_eval_ncore.py \
    --ncore_data_dir ${ncore_data_dir} \
    --test_mode target \
    --save_dir ${exp_record_dir} \
    --save_name M_${model_name}_D_${src}_${tar}_T_target \
    --user_emb_path_shared ${emcdr_emb_dir}/${src}_${tar}_src_tar_shared.pickle\
    --user_emb_path_target ${bpr_emb_dir}/${tar}_tar.txt \
    --item_emb_path ${bpr_emb_dir}/${tar}_tar.txt \
    --src ${src} \
    --tar ${tar} \
    --model_name ${model_name} || exit 1

    python3 ../../rec_and_eval_ncore.py \
    --ncore_data_dir ${ncore_data_dir} \
    --test_mode shared \
    --save_dir ${exp_record_dir} \
    --save_name M_${model_name}_D_${src}_${tar}_T_shared \
    --user_emb_path_shared ${emcdr_emb_dir}/${src}_${tar}_src_tar_shared.pickle\
    --user_emb_path_target ${bpr_emb_dir}/${tar}_tar.txt\
    --item_emb_path ${bpr_emb_dir}/${tar}_tar.txt \
    --src ${src} \
    --tar ${tar} \
    --model_name ${model_name} || exit 1

    python3 ../../rec_and_eval_ncore.py \
    --ncore_data_dir ${ncore_data_dir} \
    --test_mode cold \
    --save_dir ${exp_record_dir} \
    --save_name M_${model_name}_D_${src}_${tar}_T_cold \
    --user_emb_path_cold ${emcdr_emb_dir}/${src}_${tar}_src_ctar_cold.pickle\
    --item_emb_path ${bpr_emb_dir}/${tar}_ctar.txt \
    --src ${src} \
    --tar ${tar} \
    --model_name ${model_name} || exit 1
done

#!/bin/bash
ncore_data_dir=$1
bpr_emb_dir=$2
emcdr_input_dir=$3

declare -a datasets=("hk_csjj" "spo_csj" "mt_b")

for d in "${datasets[@]}"; 
do
    IFS='_'
    read -a domains <<< "$d"
    IFS=' '
    src=${domains[0]}
    tar=${domains[1]}

    # normal 
    python3 preprocess.py \
    --share_user_path ${ncore_data_dir}/${src}_${tar}_src_tar_all_shared_users.pickle \
    --pretrained_source_emb_path ${bpr_emb_dir}/${src}_src.txt\
    --pretrained_target_emb_path ${bpr_emb_dir}/${tar}_tar.txt\
    --Us_save_path ${emcdr_input_dir}/${src}_${tar}_src_tar_Us.pickle\
    --Us_id_map_save_path ${emcdr_input_dir}/${src}_${tar}_src_tar_Us_id_map.pickle\
    --Ut_save_path ${emcdr_input_dir}/${src}_${tar}_src_tar_Ut.pickle\

    # cold
    python3 preprocess.py \
    --share_user_path ${ncore_data_dir}/${src}_${tar}_src_tar_all_shared_users.pickle \
    --cold_user_path ${ncore_data_dir}/${src}_${tar}_src_tar_sample_testing_cold_users.pickle \
    --pretrained_source_emb_path ${bpr_emb_dir}/${src}_src.txt\
    --pretrained_target_emb_path ${bpr_emb_dir}/${tar}_ctar.txt\
    --Us_save_path ${emcdr_input_dir}/${src}_${tar}_src_ctar_Us.pickle\
    --Us_id_map_save_path ${emcdr_input_dir}/${src}_${tar}_src_ctar_Us_id_map.pickle\
    --Ut_save_path ${emcdr_input_dir}/${src}_${tar}_src_ctar_Ut.pickle\
    --cold_Us_save_path ${emcdr_input_dir}/${src}_${tar}_src_ctar_cold_Us.pickle\
    --cold_Us_id_map_save_path ${emcdr_input_dir}/${src}_${tar}_src_ctar_cold_Us_id_map.pickle || exit 1;

done

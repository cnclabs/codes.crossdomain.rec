#!/bin/bash
emcdr_input_dir=$1
emcdr_emb_dir=$2
ncore_data_dir=$3

declare -a datasets=("hk_csjj" "spo_csj" "mt_b")

for d in "${datasets[@]}"; do
    IFS='_'
    read -a domains <<< "$d"
    IFS=' '
    src=${domains[0]}
    tar=${domains[1]}
    
    # normal
    python3 MLP.py \
    --model_save_dir ${emcdr_emb_dir}/${src}_${tar}_src_tar \
    --Us ${emcdr_input_dir}/${src}_${tar}_src_tar_Us.pickle \
    --Ut ${emcdr_input_dir}/${src}_${tar}_src_tar_Ut.pickle || exit 1;
    
    # cold
    python3 MLP.py \
    --model_save_dir ${emcdr_emb_dir}/${src}_${tar}_src_ctar \
    --Us ${emcdr_input_dir}/${src}_${tar}_src_ctar_Us.pickle \
    --Ut ${emcdr_input_dir}/${src}_${tar}_src_ctar_Ut.pickle || exit 1;
done

for d in "${datasets[@]}"; do
    IFS='_'
	read -a domains <<< "$d"
	IFS=' '
	src=${domains[0]}
	tar=${domains[1]}

   python3 infer_Us.py \
       --user_to_infer_path ${ncore_data_dir}/${src}_${tar}_src_tar_sample_testing_shared_users.pickle \
       --Us_path ${emcdr_input_dir}/${src}_${tar}_src_tar_Us.pickle \
       --meta_path ${emcdr_emb_dir}/${src}_${tar}_src_tar/mlp.meta \
       --ckpt_path ${emcdr_emb_dir}/${src}_${tar}_src_tar/ \
       --emb_save_path ${emcdr_emb_dir}/${src}_${tar}_src_tar_shared.pickle || exit 1;

   # cold
   python3 infer_Us.py \
       --user_to_infer_path ${ncore_data_dir}/${src}_${tar}_src_tar_sample_testing_cold_users.pickle \
       --Us_path ${emcdr_input_dir}/${src}_${tar}_src_ctar_cold_Us.pickle \
       --meta_path ${emcdr_emb_dir}/${src}_${tar}_src_ctar/mlp.meta \
       --ckpt_path ${emcdr_emb_dir}/${src}_${tar}_src_ctar/ \
       --emb_save_path ${emcdr_emb_dir}/${src}_${tar}_src_ctar_cold.pickle || exit 1;
   
done

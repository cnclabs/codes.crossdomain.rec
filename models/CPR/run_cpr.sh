#!/bin/bash
ncore_data_dir=$1
cpr_input_dir=$2
emb_save_dir=$3
exp_record_dir=$4
mode=$5
src=$6
tar=$7
model_name=cpr
epoch=200
dim=100
worker=20
lr=0.025
user_reg=0.01
item_reg=0.06

if [[ ! -d ${emb_save_dir} ]]
then
	mkdir -p ${emb_save_dir}
fi

if [[ "$mode" == "train" || "$mode" == "traineval" ]]
then 
    ./cpr \
    -train_ut ${cpr_input_dir}/${tar}_tar_train_input.txt \
    -train_us ${cpr_input_dir}/${src}_src_train_input.txt \
    -save ${emb_save_dir}/${src}_${tar}_normal.txt \
    -dimension ${dim} -update_times ${epoch} -worker ${worker} -init_alpha ${lr} -user_reg ${user_reg} -item_reg ${item_reg} || exit 1
    
    # cold
    ./cpr \
    -train_ut ${cpr_input_dir}/${tar}_ctar_train_input.txt \
    -train_us ${cpr_input_dir}/${src}_src_train_input.txt \
    -save ${emb_save_dir}/${src}_${tar}_cold.txt \
    -dimension ${dim} -update_times ${epoch} -worker ${worker} -init_alpha ${lr} -user_reg ${user_reg} -item_reg ${item_reg} || exit 1
fi

if [[ "$mode" == "eval" || "$mode" == "traineval" ]]
then
    python3 ../../rec_and_eval_ncore.py \
    --ncore_data_dir ${ncore_data_dir} \
    --test_mode target \
    --save_dir ${exp_record_dir} \
    --save_name M_cpr_D_${src}_${tar}_T_target \
    --user_emb_path ${emb_save_dir}/${src}_${tar}_normal.txt \
    --item_emb_path ${emb_save_dir}/${src}_${tar}_normal.txt \
    --src ${src} \
    --tar ${tar} \
    --model_name ${model_name} || exit 1
    
    python3 ../../rec_and_eval_ncore.py \
    --ncore_data_dir ${ncore_data_dir} \
    --test_mode shared \
    --save_dir ${exp_record_dir} \
    --save_name M_cpr_D_${src}_${tar}_T_shared \
    --user_emb_path ${emb_save_dir}/${src}_${tar}_normal.txt \
    --item_emb_path ${emb_save_dir}/${src}_${tar}_normal.txt \
    --src ${src} \
    --tar ${tar} \
    --model_name ${model_name} || exit 1
    
    python3 ../../rec_and_eval_ncore.py \
    --ncore_data_dir ${ncore_data_dir} \
    --test_mode cold \
    --save_dir ${exp_record_dir} \
    --save_name M_cpr_D_${src}_${tar}_T_cold \
    --user_emb_path ${emb_save_dir}/${src}_${tar}_cold.txt \
    --item_emb_path ${emb_save_dir}/${src}_${tar}_cold.txt \
    --src ${src} \
    --tar ${tar} \
    --model_name ${model_name} || exit 1
fi

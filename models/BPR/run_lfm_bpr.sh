#!/bin/bash
ncore_data_dir=$1
cpr_input_dir=$2
bpr_input_dir=$3
emb_save_dir=$4
exp_record_dir=$5
mode=$6
src=$7
tar=$8
model_name=bpr
epoch=200
dim=100
worker=20
l2_reg=0.00001

if [[ ! -d ${emb_save_dir} ]]
then
        mkdir -p ${emb_save_dir}
fi

if [[ "$mode" == "train" || "$mode" == "traineval" ]]
then 
    # src tar
    python3 ./lfm_bpr.py \
    --train ${bpr_input_dir}/${src}_${tar}_src_tar_train_input.txt \
    --save ${emb_save_dir}/${src}_${tar}_src_tar.txt \
    --dim ${dim} \
    --iter ${epoch} \
    --worker ${worker} \
    --item_alpha ${l2_reg} \
    --user_alpha ${l2_reg} || exit 1
    
    # src ctar
    python3 ./lfm_bpr.py \
    --train ${bpr_input_dir}/${src}_${tar}_src_ctar_train_input.txt \
    --save ${emb_save_dir}/${src}_${tar}_src_ctar.txt \
    --dim ${dim} \
    --iter ${epoch} \
    --worker ${worker} \
    --item_alpha ${l2_reg} \
    --user_alpha ${l2_reg} || exit 1
    
    # tar
    python3 ./lfm_bpr.py \
    --train ${cpr_input_dir}/${tar}_tar_train_input.txt \
    --save ${emb_save_dir}/${tar}_tar.txt \
    --dim ${dim} \
    --iter ${epoch} \
    --worker ${worker} \
    --item_alpha ${l2_reg} \
    --user_alpha ${l2_reg} || exit 1
    
    # src
    python3 ./lfm_bpr.py \
    --train ${cpr_input_dir}/${src}_src_train_input.txt \
    --save ${emb_save_dir}/${src}_tar.txt \
    --dim ${dim} \
    --iter ${epoch} \
    --worker ${worker} \
    --item_alpha ${l2_reg} \
    --user_alpha ${l2_reg}  || exit 1
    
    # ctar
    python3 ./lfm_bpr.py \
    --train ${cpr_input_dir}/${tar}_ctar_train_input.txt \
    --save ${emb_save_dir}/${tar}_ctar.txt \
    --dim ${dim} \
    --iter ${epoch} \
    --worker ${worker} \
    --item_alpha ${l2_reg} \
    --user_alpha ${l2_reg} || exit 1
fi

if [[ "$mode" == "eval" || "$mode" == "traineval" ]]
then
# bpr target
python3 ../../rec_and_eval_ncore.py \
    --ncore_data_dir ${ncore_data_dir} \
    --test_mode target \
    --save_dir ${exp_record_dir} \
    --save_name M_${model_name}_D_${src}_${tar}_T_target \
    --user_emb_path ${emb_save_dir}/${tar}_tar.txt \
    --item_emb_path ${emb_save_dir}/${tar}_tar.txt \
    --src ${src} \
    --tar ${tar} \
    --model_name ${model_name} || exit 1

# bpr shared
python3 ../../rec_and_eval_ncore.py \
    --ncore_data_dir ${ncore_data_dir} \
    --test_mode shared \
    --save_dir ${exp_record_dir} \
    --save_name M_${model_name}_D_${src}_${tar}_T_shared \
    --user_emb_path ${emb_save_dir}/${tar}_tar.txt \
    --item_emb_path ${emb_save_dir}/${tar}_tar.txt \
    --src ${src} \
    --tar ${tar} \
    --model_name ${model_name} || exit 1

# bpr* target 
python3 ../../rec_and_eval_ncore.py \
    --ncore_data_dir ${ncore_data_dir} \
    --test_mode target \
    --save_dir ${exp_record_dir} \
    --save_name M_${model_name}_s_D_${src}_${tar}_T_target \
    --user_emb_path ${emb_save_dir}/${src}_${tar}_src_tar.txt \
    --item_emb_path ${emb_save_dir}/${src}_${tar}_src_tar.txt \
    --src ${src} \
    --tar ${tar} \
    --model_name ${model_name}_s || exit 1

# bpr* shared
python3 ../../rec_and_eval_ncore.py \
    --ncore_data_dir ${ncore_data_dir} \
    --test_mode shared \
    --save_dir ${exp_record_dir} \
    --save_name M_${model_name}_s_D_${src}_${tar}_T_shared \
    --user_emb_path ${emb_save_dir}/${src}_${tar}_src_tar.txt \
    --item_emb_path ${emb_save_dir}/${src}_${tar}_src_tar.txt \
    --src ${src} \
    --tar ${tar} \
    --model_name ${model_name}_s || exit 1

# bpr* cold
python3 ../../rec_and_eval_ncore.py \
    --ncore_data_dir ${ncore_data_dir} \
    --test_mode cold \
    --save_dir ${exp_record_dir} \
    --save_name M_${model_name}_s_D_${src}_${tar}_T_cold \
    --user_emb_path ${emb_save_dir}/${src}_${tar}_src_ctar.txt \
    --item_emb_path ${emb_save_dir}/${src}_${tar}_src_ctar.txt \
    --src ${src} \
    --tar ${tar} \
    --model_name ${model_name}_s || exit 1
fi

#!/bin/bash
ncore_data_dir=$1
lgn_input_dir=$2
emb_save_dir=$3
exp_record_dir=$4
mode=$5
src=$6
tar=$7
gpu_id=$8
single_normal_cold=$9
model_name=lgn
epoch=200
embed_size=100

# lgn lil
if [[ $single_normal_cold == "single" ]]
then
    if [[ "$mode" == "train" || "$mode" == "traineval" ]]
    then 
	CUDA_VISIBLE_DEVICES=${gpu_id} \
        python lightgcn/code/main.py \
       	    --decay=1e-4 \
	    --lr=0.001 \
	    --layer=3 \
	    --seed=2020 \
	    --recdim=${embed_size} \
	    --epochs ${epoch} \
	    --user_unique_id_map_path=${lgn_input_dir}/${tar}_tar_user_id_map.txt \
	    --item_unique_id_map_path=${lgn_input_dir}/${tar}_tar_item_id_map.txt \
	    --train_file_path=${lgn_input_dir}/${tar}_tar_train_input.txt \
	    --s_pre_adj_mat_path=${lgn_input_dir}/${tar}_tar_train_adj_mat \
	    --emb_save_path=${emb_save_dir}/${tar}_tar.txt  || exit 1
    fi

    if [[ "$mode" == "eval" || "$mode" == "traineval" ]]
    then
        python3 ../rec_and_eval_ncore.py \
	    --ncore_data_dir ${ncore_data_dir} \
	    --test_mode target \
            --save_dir ${exp_record_dir} \
	    --save_name M_${model_name}_lil_D_${src}_${tar}_T_target \
            --item_emb_path ${emb_save_dir}/${tar}_tar.txt \
            --user_emb_path ${emb_save_dir}/${tar}_tar.txt \
            --src ${src} \
            --tar ${tar} \
            --model_name ${model_name}_lil || exit 1

        python3 ../rec_and_eval_ncore.py \
	    --ncore_data_dir ${ncore_data_dir} \
	    --test_mode shared \
            --save_dir ${exp_record_dir} \
	    --save_name M_${model_name}_lil_D_${src}_${tar}_T_shared \
            --item_emb_path ${emb_save_dir}/${tar}_tar.txt \
            --user_emb_path ${emb_save_dir}/${tar}_tar.txt \
            --src ${src} \
            --tar ${tar} \
            --model_name ${model_name}_lil || exit 1
    fi
fi 

# lgn big
if [[ $single_normal_cold == "normal" ]]
then
    if [[ "$mode" == "train" || "$mode" == "traineval" ]]
    then 
	# lgn big normal 
	CUDA_VISIBLE_DEVICES=${gpu_id} \
        python lightgcn/code/main.py \
       	    --decay=1e-4 \
	    --lr=0.001 \
	    --layer=3 \
	    --seed=2020 \
	    --recdim=${embed_size} \
	    --epochs ${epoch} \
	    --user_unique_id_map_path=${lgn_input_dir}/${src}_${tar}_src_tar_user_id_map.txt \
	    --item_unique_id_map_path=${lgn_input_dir}/${src}_${tar}_src_tar_item_id_map.txt \
	    --train_file_path=${lgn_input_dir}/${src}_${tar}_src_tar_train_input.txt \
	    --s_pre_adj_mat_path=${lgn_input_dir}/${src}_${tar}_src_tar_train_adj_mat \
	    --emb_save_path=${emb_save_dir}/${src}_${tar}_src_tar.txt  || exit 1
    fi

    if [[ "$mode" == "eval" || "$mode" == "traineval" ]]
    then
        python3 ../rec_and_eval_ncore.py \
	    --ncore_data_dir ${ncore_data_dir} \
	    --test_mode target \
            --save_dir ${exp_record_dir} \
	    --save_name M_${model_name}_big_D_${src}_${tar}_T_target \
            --item_emb_path ${emb_save_dir}/${src}_${tar}_src_tar.txt \
            --user_emb_path ${emb_save_dir}/${src}_${tar}_src_tar.txt \
            --src ${src} \
            --tar ${tar} \
            --model_name ${model_name}_big || exit 1

        python3 ../rec_and_eval_ncore.py \
	    --ncore_data_dir ${ncore_data_dir} \
	    --test_mode shared \
            --save_dir ${exp_record_dir} \
	    --save_name M_${model_name}_big_D_${src}_${tar}_T_shared \
            --item_emb_path ${emb_save_dir}/${src}_${tar}_src_tar.txt \
            --user_emb_path ${emb_save_dir}/${src}_${tar}_src_tar.txt \
            --src ${src} \
            --tar ${tar} \
            --model_name ${model_name}_big || exit 1
    fi
fi 

if [[ $single_normal_cold == "cold" ]]
then
    if [[ "$mode" == "train" || "$mode" == "traineval" ]]
    then 
	# lgn big normal 
	CUDA_VISIBLE_DEVICES=${gpu_id} \
        python lightgcn/code/main.py \
       	    --decay=1e-4 \
	    --lr=0.001 \
	    --layer=3 \
	    --seed=2020 \
	    --recdim=${embed_size} \
	    --epochs ${epoch} \
	    --user_unique_id_map_path=${lgn_input_dir}/${src}_${tar}_src_ctar_user_id_map.txt \
	    --item_unique_id_map_path=${lgn_input_dir}/${src}_${tar}_src_ctar_item_id_map.txt \
	    --train_file_path=${lgn_input_dir}/${src}_${tar}_src_ctar_train_input.txt \
	    --s_pre_adj_mat_path=${lgn_input_dir}/${src}_${tar}_src_ctar_train_adj_mat \
	    --emb_save_path=${emb_save_dir}/${src}_${tar}_src_ctar.txt  || exit 1
    fi

    if [[ "$mode" == "eval" || "$mode" == "traineval" ]]
    then
        python3 ../rec_and_eval_ncore.py \
	    --ncore_data_dir ${ncore_data_dir} \
	    --test_mode cold \
            --save_dir ${exp_record_dir} \
	    --save_name M_${model_name}_big_D_${src}_${tar}_T_cold \
            --item_emb_path ${emb_save_dir}/${src}_${tar}_src_ctar.txt \
            --user_emb_path ${emb_save_dir}/${src}_${tar}_src_ctar.txt \
            --src ${src} \
            --tar ${tar} \
            --model_name ${model_name}_big || exit 1
    fi
fi 

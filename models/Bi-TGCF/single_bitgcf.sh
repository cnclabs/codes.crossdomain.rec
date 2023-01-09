#!/bin/bash
ncore_data_dir=$1
cpr_input_dir=$2
emb_save_dir=$3
exp_record_dir=$4
mode=$5
src=$6
tar=$7
gpu_id=$8
normal_cold=$9
model_name=bitgcf
epoch=200 # current code will save only {20, 40,....}
embed_size=25
batch_size=65536

if [[ $normal_cold == "normal" ]]
then
    if [[ "$mode" == "train" || "$mode" == "traineval" ]]
    then 
	CUDA_VISIBLE_DEVICES=${gpu_id} \
        python3 ./BiTGCF/main.py \
            --data_path ${cpr_input_dir} \
            --source_dataset ${src}_src \
            --target_dataset ${tar}_tar \
            --epoch ${epoch} \
            --batch_size ${batch_size} \
	    --emb_save_part_path ${emb_save_dir}/${src}_${tar}_src_tar \
            --embed_size ${embed_size} || exit 1
    fi

    if [[ "$mode" == "eval" || "$mode" == "traineval" ]]
    then
        python3 ../../rec_and_eval_ncore.py \
	    --ncore_data_dir ${ncore_data_dir} \
	    --test_mode target \
            --save_dir ${exp_record_dir} \
	    --save_name M_${model_name}_D_${src}_${tar}_T_target \
            --item_emb_path ${emb_save_dir}/${src}_${tar}_src_tar_${epoch}.txt \
            --user_emb_path ${emb_save_dir}/${src}_${tar}_src_tar_${epoch}.txt \
            --src ${src} \
            --tar ${tar} \
            --model_name ${model_name} || exit 1

        python3 ../../rec_and_eval_ncore.py \
	    --ncore_data_dir ${ncore_data_dir} \
	    --test_mode shared \
            --save_dir ${exp_record_dir} \
	    --save_name M_${model_name}_D_${src}_${tar}_T_shared \
            --item_emb_path ${emb_save_dir}/${src}_${tar}_src_tar_${epoch}.txt \
            --user_emb_path ${emb_save_dir}/${src}_${tar}_src_tar_${epoch}.txt \
            --src ${src} \
            --tar ${tar} \
            --model_name ${model_name} || exit 1
    fi
fi 

if [[ $normal_cold == "cold" ]]
then
    if [[ "$mode" == "train" || "$mode" == "traineval" ]]
    then 
	CUDA_VISIBLE_DEVICES=${gpu_id} \
        python3 ./BiTGCF/main.py\
            --data_path ${cpr_input_dir} \
            --source_dataset ${src}_src \
            --target_dataset ${tar}_ctar \
            --epoch ${epoch} \
            --batch_size ${batch_size} \
	    --emb_save_part_path ${emb_save_dir}/${src}_${tar}_src_ctar \
            --embed_size ${embed_size} || exit 1
    fi

    if [[ "$mode" == "eval" || "$mode" == "traineval" ]]
    then
        python3 ../../rec_and_eval_ncore.py \
	    --ncore_data_dir ${ncore_data_dir} \
	    --test_mode cold \
            --save_dir ${exp_record_dir} \
	    --save_name M_${model_name}_D_${src}_${tar}_T_cold \
            --item_emb_path ${emb_save_dir}/${src}_${tar}_src_ctar_${epoch}.txt \
            --user_emb_path ${emb_save_dir}/${src}_${tar}_src_ctar_${epoch}.txt \
            --src ${src} \
            --tar ${tar} \
            --model_name ${model_name} || exit 1
    fi
fi

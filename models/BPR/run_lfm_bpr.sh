#!/bin/bash
data_dir=$1
model_save_dir=$2
exp_record_dir=$3
mode=$4
model_name=bpr
update_times=200
dim=100
declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=5)
sample=3500

if [[ ! -d ${model_save_dir} ]]
then
        mkdir -p ${model_save_dir}
fi

for d in "${datasets[@]}"; do
    IFS='_'
    read -a domains <<< "$d"
    IFS=' '
    src=${domains[0]}
    tar=${domains[1]}
    ncore=${ncores[$d]}

    if [[ "$mode" == "train" || "$mode" == "traineval" ]]
    then 
    ### tar
    python3 ./lfm_bpr.py \
    --train ${data_dir}/input_${ncore}core/${tar}_train_input.txt \
    --save ${model_save_dir}/${tar}.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001

    ### src
    python3 ./lfm_bpr.py \
    --train ${data_dir}/input_${ncore}core/all_${src}_train_input.txt \
    --save ${model_save_dir}/${src}.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001

    ### tar cold
    python3 ./lfm_bpr.py \
    --train ${data_dir}/input_${ncore}core/cold_${tar}_train_input.txt \
    --save ${model_save_dir}/cold_${tar}.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001


    # all input (target)
    python3 ./lfm_bpr.py \
    --train ${data_dir}/input_${ncore}core/all_cpr_train_u_${src}+${tar}.txt \
    --save ${model_save_dir}/${src}+${tar}.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001

    # all input (cold)

    python3 ./lfm_bpr.py \
    --train ${data_dir}/input_${ncore}core/cold_cpr_train_u_${src}+${tar}.txt \
    --save ${model_save_dir}/cold_${src}+${tar}.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001

    fi

    if [[ "$mode" == "eval" || "$mode" == "traineval" ]]
    then
    ### eval @tar @shared
    python3 ../../rec_and_eval_ncore.py \
	--data_dir ${data_dir} \
	--test_mode target \
        --save_dir ${exp_record_dir} \
	--save_name M_${model_name}_D_${src}_${tar}_T_target \
	--user_emb_path ${model_save_dir}/${tar}.txt \
	--item_emb_path ${model_save_dir}/${tar}.txt \
	--src ${src} \
	--tar ${tar} \
        --model_name ${model_name} \
	--ncore ${ncore}

    python3 ../../rec_and_eval_ncore.py \
	--data_dir ${data_dir} \
	--test_mode shared \
        --save_dir ${exp_record_dir} \
	--save_name M_${model_name}_D_${src}_${tar}_T_shared \
	--user_emb_path ${model_save_dir}/${tar}.txt \
	--item_emb_path ${model_save_dir}/${tar}.txt \
	--src ${src} \
	--tar ${tar} \
        --model_name ${model_name} \
	--ncore ${ncore} 
    fi

    ### eval @tar @shared
    python3 ../../rec_and_eval_ncore.py \
	--data_dir ${data_dir} \
	--test_mode target \
        --save_dir ${exp_record_dir} \
	--save_name M_${model_name}_s_D_${src}_${tar}_T_target \
	--user_emb_path ${model_save_dir}/${src}+${tar}.txt \
	--item_emb_path ${model_save_dir}/${src}+${tar}.txt \
	--src ${src} \
	--tar ${tar} \
        --model_name ${model_name}_s \
	--ncore ${ncore} 

    python3 ../../rec_and_eval_ncore.py \
	--data_dir ${data_dir} \
	--test_mode shared \
        --save_dir ${exp_record_dir} \
	--save_name M_${model_name}_s_D_${src}_${tar}_T_shared \
	--user_emb_path ${model_save_dir}/${src}+${tar}.txt \
	--item_emb_path ${model_save_dir}/${src}+${tar}.txt \
	--src ${src} \
	--tar ${tar} \
        --model_name ${model_name}_s\
	--ncore ${ncore} 

    ### eval @cold
    python3 ../../rec_and_eval_ncore.py \
	--data_dir ${data_dir} \
	--test_mode cold \
        --save_dir ${exp_record_dir} \
	--save_name M_${model_name}_s_D_${src}_${tar}_T_cold \
	--user_emb_path ${model_save_dir}/cold_${src}+${tar}.txt \
	--item_emb_path ${model_save_dir}/cold_${src}+${tar}.txt \
	--src ${src} \
	--tar ${tar} \
        --model_name ${model_name}_s\
	--ncore ${ncore} 
done

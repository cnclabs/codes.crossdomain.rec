#!/bin/bash
ncore_data_dir=$1
cpr_input_dir=$2
bpr_input_dir=$3
model_save_dir=$4
exp_record_dir=$5
mode=$6
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
    # all input (target)
    python3 ./lfm_bpr.py \
    --train ${bpr_input_dir}/${src}_${tar}_src_tar_train_input.txt \
    --save ${model_save_dir}/${src}_${tar}_src_tar.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001 || exit 1

    # all input (cold)

    python3 ./lfm_bpr.py \
    --train ${bpr_input_dir}/${src}_${tar}_src_ctar_train_input.txt \
    --save ${model_save_dir}/${src}_${tar}_src_ctar.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001 || exit 1

    ### tar
    python3 ./lfm_bpr.py \
    --train ${cpr_input_dir}/${tar}_tar_train_input.txt \
    --save ${model_save_dir}/${tar}_tar.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001 || exit 1

    ### src
    python3 ./lfm_bpr.py \
    --train ${cpr_input_dir}/${src}_src_train_input.txt \
    --save ${model_save_dir}/${src}_tar.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001  || exit 1

    ### tar cold
    python3 ./lfm_bpr.py \
    --train ${cpr_input_dir}/${tar}_ctar_train_input.txt \
    --save ${model_save_dir}/${tar}_ctar.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001 || exit 1



    fi

    if [[ "$mode" == "eval" || "$mode" == "traineval" ]]
    then
    ### eval @tar @shared
    python3 ../../rec_and_eval_ncore.py \
	--ncore_data_dir ${ncore_data_dir} \
	--test_mode target \
        --save_dir ${exp_record_dir} \
	--save_name M_${model_name}_D_${src}_${tar}_T_target \
	--user_emb_path ${model_save_dir}/${tar}_tar.txt \
	--item_emb_path ${model_save_dir}/${tar}_tar.txt \
	--src ${src} \
	--tar ${tar} \
        --model_name ${model_name} \
	--ncore ${ncore} || exit 1

    python3 ../../rec_and_eval_ncore.py \
	--ncore_data_dir ${ncore_data_dir} \
	--test_mode shared \
        --save_dir ${exp_record_dir} \
	--save_name M_${model_name}_D_${src}_${tar}_T_shared \
	--user_emb_path ${model_save_dir}/${tar}_tar.txt \
	--item_emb_path ${model_save_dir}/${tar}_tar.txt \
	--src ${src} \
	--tar ${tar} \
        --model_name ${model_name} \
	--ncore ${ncore} || exit 1

    ### eval @tar @shared
    python3 ../../rec_and_eval_ncore.py \
	--ncore_data_dir ${ncore_data_dir} \
	--test_mode target \
        --save_dir ${exp_record_dir} \
	--save_name M_${model_name}_s_D_${src}_${tar}_T_target \
	--user_emb_path ${model_save_dir}/${src}_${tar}_src_tar.txt \
	--item_emb_path ${model_save_dir}/${src}_${tar}_src_tar.txt \
	--src ${src} \
	--tar ${tar} \
        --model_name ${model_name}_s \
	--ncore ${ncore} || exit 1

    python3 ../../rec_and_eval_ncore.py \
	--ncore_data_dir ${ncore_data_dir} \
	--test_mode shared \
        --save_dir ${exp_record_dir} \
	--save_name M_${model_name}_s_D_${src}_${tar}_T_shared \
	--user_emb_path ${model_save_dir}/${src}_${tar}_src_tar.txt \
	--item_emb_path ${model_save_dir}/${src}_${tar}_src_tar.txt \
	--src ${src} \
	--tar ${tar} \
        --model_name ${model_name}_s\
	--ncore ${ncore} || exit 1

    ### eval @cold
    python3 ../../rec_and_eval_ncore.py \
	--ncore_data_dir ${ncore_data_dir} \
	--test_mode cold \
        --save_dir ${exp_record_dir} \
	--save_name M_${model_name}_s_D_${src}_${tar}_T_cold \
	--user_emb_path ${model_save_dir}/${src}_${tar}_src_ctar.txt \
	--item_emb_path ${model_save_dir}/${src}_${tar}_src_ctar.txt \
	--src ${src} \
	--tar ${tar} \
        --model_name ${model_name}_s\
	--ncore ${ncore} || exit 1
    fi
done

#!/bin/bash
data_dir=$1
exp_record_dir=$2
mode=$3
model_name=bpr
update_times=200
dim=100
declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=5)
sample=3500

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
    python3 ./BPR/lfm-bpr.py \
    --train ${data_dir}/input_${ncore}core/${tar}_train_input.txt \
    --save ./lfm_bpr_graphs/${tar}_lightfm_bpr_${update_times}_10e-5.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001

    ### src
    python3 ./BPR/lfm-bpr.py \
    --train ${data_dir}/input_${ncore}core/all_${src}_train_input.txt \
    --save ./lfm_bpr_graphs/${src}_lightfm_bpr_${update_times}_10e-5.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001

    ### tar cold
    python3 ./BPR/lfm-bpr.py \
    --train ${data_dir}/input_${ncore}core/cold_${tar}_train_input.txt \
    --save ./lfm_bpr_graphs/cold_${tar}_lightfm_bpr_${update_times}_10e-5.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001


    # all input (target)
    python3 ./BPR/lfm-bpr.py \
    --train ${data_dir}/input_${ncore}core/all_cpr_train_u_${src}+${tar}.txt \
    --save ./lfm_bpr_graphs/${src}+${tar}_lightfm_bpr_${update_times}_10e-5.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001

    # all input (cold)

    python3 ./BPR/lfm-bpr.py \
    --train ${data_dir}/input_${ncore}core/cold_cpr_train_u_${src}+${tar}.txt \
    --save ./lfm_bpr_graphs/cold_${src}+${tar}_lightfm_bpr_${update_times}_10e-5.txt \
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
	--user_emb_path $(pwd)/lfm_bpr_graphs/${tar}_lightfm_bpr_${update_times}_10e-5.txt \
	--item_emb_path $(pwd)/lfm_bpr_graphs/${tar}_lightfm_bpr_${update_times}_10e-5.txt \
	--src ${src} \
	--tar ${tar} \
        --model_name ${model_name} \
	--ncore ${ncore}

    python3 ../../rec_and_eval_ncore.py \
	--data_dir ${data_dir} \
	--test_mode shared \
        --save_dir ${exp_record_dir} \
	--save_name M_${model_name}_D_${src}_${tar}_T_shared \
	--user_emb_path $(pwd)/lfm_bpr_graphs/${tar}_lightfm_bpr_${update_times}_10e-5.txt \
	--item_emb_path $(pwd)/lfm_bpr_graphs/${tar}_lightfm_bpr_${update_times}_10e-5.txt \
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
	--user_emb_path $(pwd)/lfm_bpr_graphs/${src}+${tar}_lightfm_bpr_${update_times}_10e-5.txt \
	--item_emb_path $(pwd)/lfm_bpr_graphs/${src}+${tar}_lightfm_bpr_${update_times}_10e-5.txt \
	--src ${src} \
	--tar ${tar} \
        --model_name ${model_name}_s \
	--ncore ${ncore} 

    python3 ../../rec_and_eval_ncore.py \
	--data_dir ${data_dir} \
	--test_mode shared \
        --save_dir ${exp_record_dir} \
	--save_name M_${model_name}_s_D_${src}_${tar}_T_shared \
	--user_emb_path $(pwd)/lfm_bpr_graphs/${src}+${tar}_lightfm_bpr_${update_times}_10e-5.txt \
	--item_emb_path $(pwd)/lfm_bpr_graphs/${src}+${tar}_lightfm_bpr_${update_times}_10e-5.txt \
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
	--user_emb_path $(pwd)/lfm_bpr_graphs/cold_${src}+${tar}_lightfm_bpr_${update_times}_10e-5.txt \
	--item_emb_path $(pwd)/lfm_bpr_graphs/cold_${src}+${tar}_lightfm_bpr_${update_times}_10e-5.txt \
	--src ${src} \
	--tar ${tar} \
        --model_name ${model_name}_s\
	--ncore ${ncore} 
done

#!/bin/bash
data_dir=$1
exp_record_dir=$2
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
    ### tar
    python3 ./BPR/lfm-bpr.py \
    --train ${data_dir}/input_${ncore}core/${tar}_train_input.txt \
    --save ./lfm_bpr_graphs/${tar}_lightfm_bpr_${update_times}_10e-5.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001

    ### eval @tar @shared
    python3 ../../rec_and_eval_ncore.py \
	--data_dir ${data_dir} \
	--test_mode target \
        --save_dir ${exp_record_dir} \
	--save_name M_${model_name}_D_${src}_${tar}_T_target \
	--output_file $(pwd)/lfm_bpr_results/${src}_${tar}_lightfm_bpr_result_${update_times}_target.txt \
	--graph_file $(pwd)/lfm_bpr_graphs/${tar}_lightfm_bpr_${update_times}_10e-5.txt \
	--src ${src} \
	--tar ${tar} \
        --model_name ${model_name} \
	--ncore ${ncore}

    python3 ../../rec_and_eval_ncore.py \
	--data_dir ${data_dir} \
	--test_mode shared \
        --save_dir ${exp_record_dir} \
	--save_name M_${model_name}_D_${src}_${tar}_T_shared \
	--output_file $(pwd)/lfm_bpr_results/${src}_${tar}_lightfm_bpr_result_${update_times}_shared.txt \
	--graph_file $(pwd)/lfm_bpr_graphs/${tar}_lightfm_bpr_${update_times}_10e-5.txt \
	--src ${src} \
	--tar ${tar} \
        --model_name ${model_name} \
	--ncore ${ncore} 

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

    ### eval @tar @shared
    python3 ../../rec_and_eval_ncore.py \
	--data_dir ${data_dir} \
	--test_mode target \
        --save_dir ${exp_record_dir} \
	--save_name M_${model_name}_s_D_${src}_${tar}_T_target \
	--output_file $(pwd)/lfm_bpr_results/${src}_${tar}_lightfm_bpr+_result_${update_times}_target.txt \
	--graph_file $(pwd)/lfm_bpr_graphs/${src}+${tar}_lightfm_bpr_${update_times}_10e-5.txt \
	--src ${src} \
	--tar ${tar} \
        --model_name ${model_name}_s \
	--ncore ${ncore} 

    python3 ../../rec_and_eval_ncore.py \
	--data_dir ${data_dir} \
	--test_mode shared \
        --save_dir ${exp_record_dir} \
	--save_name M_${model_name}_s_D_${src}_${tar}_T_shared \
	--output_file $(pwd)/lfm_bpr_results/${src}_${tar}_lightfm_bpr+_result_${update_times}_shared.txt \
	--graph_file $(pwd)/lfm_bpr_graphs/${src}+${tar}_lightfm_bpr_${update_times}_10e-5.txt \
	--src ${src} \
	--tar ${tar} \
        --model_name ${model_name}_s\
	--ncore ${ncore} 

    # all input (cold)

    python3 ./BPR/lfm-bpr.py \
    --train ${data_dir}/input_${ncore}core/cold_cpr_train_u_${src}+${tar}.txt \
    --save ./lfm_bpr_graphs/cold_${src}+${tar}_lightfm_bpr_${update_times}_10e-5.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001

    ### eval @cold
    python3 ../../rec_and_eval_ncore.py \
	--data_dir ${data_dir} \
	--test_mode cold \
        --save_dir ${exp_record_dir} \
	--save_name M_${model_name}_s_D_${src}_${tar}_T_cold \
	--output_file $(pwd)/lfm_bpr_results/${src}_${tar}_lightfm_bpr+_result_${update_times}_cold.txt \
	--graph_file $(pwd)/lfm_bpr_graphs/cold_${src}+${tar}_lightfm_bpr_${update_times}_10e-5.txt \
	--src ${src} \
	--tar ${tar} \
        --model_name ${model_name}_s\
	--ncore ${ncore} 
done

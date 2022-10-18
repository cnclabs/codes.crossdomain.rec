#!/bin/bash
src=$1
tar=$2
gpu_id=$3
normal_cold=$4
data_dir=$5
exp_record_dir=$6
mode=$7
n_epoch=200 # current code will save only {20, 40,....}
model_name=bitgcf
sample=3500

declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=5)
ncore=${ncores[${src}_${tar}]}

if [[ ! -d ./graph ]]
then
	mkdir -p ./graph
fi

if [[ ! -d ./result ]]
then
	mkdir -p ./result
fi

if [[ $normal_cold == "normal" ]]
then
    if [[ "$mode" == "train" || "$mode" == "traineval" ]]
    then 
	CUDA_VISIBLE_DEVICES=${gpu_id} \
        python3 ./BiTGCF/main.py \
            --data_path ${data_dir}/input_${ncore}core \
            --source_dataset all_${src} \
            --target_dataset ${tar} \
            --epoch ${n_epoch} \
            --batch_size 65536 \
            --embed_size 25
    fi

    if [[ "$mode" == "eval" || "$mode" == "traineval" ]]
    then
        python3 ../../rec_and_eval_ncore.py \
	    --data_dir ${data_dir} \
	    --test_mode target \
            --save_dir ${exp_record_dir} \
	    --save_name M_${model_name}_D_${src}_${tar}_T_target \
            --item_emb_path $(pwd)/graph/all_${src}_${tar}_${n_epoch}.graph \
            --user_emb_path $(pwd)/graph/all_${src}_${tar}_${n_epoch}.graph \
            --src ${src} \
            --tar ${tar} \
            --model_name ${model_name} \
            --ncore ${ncore}&&
        python3 ../../rec_and_eval_ncore.py \
	    --data_dir ${data_dir} \
	    --test_mode shared \
            --save_dir ${exp_record_dir} \
	    --save_name M_${model_name}_D_${src}_${tar}_T_shared \
            --user_emb_path $(pwd)/graph/all_${src}_${tar}_${n_epoch}.graph \
            --item_emb_path $(pwd)/graph/all_${src}_${tar}_${n_epoch}.graph \
            --src ${src} \
            --tar ${tar} \
            --model_name ${model_name} \
            --ncore ${ncore}
    fi
fi 

if [[ $normal_cold == "cold" ]]
then
    if [[ "$mode" == "train" || "$mode" == "traineval" ]]
    then 
	CUDA_VISIBLE_DEVICES=${gpu_id} \
        python3 ./BiTGCF/main.py\
            --data_path ${data_dir}/input_${ncore}core \
            --source_dataset all_${src} \
            --target_dataset cold_${tar} \
            --epoch ${n_epoch} \
            --batch_size 65536 \
            --embed_size 25
    fi

    if [[ "$mode" == "eval" || "$mode" == "traineval" ]]
    then
        python3 ../../rec_and_eval_ncore.py \
	    --data_dir ${data_dir} \
	    --test_mode cold \
            --save_dir ${exp_record_dir} \
	    --save_name M_${model_name}_D_${src}_${tar}_T_cold \
            --user_emb_path $(pwd)/graph/all_${src}_cold_${tar}_${n_epoch}.graph \
            --item_emb_path $(pwd)/graph/all_${src}_cold_${tar}_${n_epoch}.graph \
            --src ${src} \
            --tar ${tar} \
            --model_name ${model_name}\
            --ncore ${ncore}
    fi
fi

#!/bin/bash
src=$1 #(hk spo mt) 
tar=$2 #(csjj csj b)
graph=$3 #(big lil)
test_mode=$4 #(target shared cold)
mode=$5 #train, eval, traineval
gpu_id=$6
data_dir=$7
exp_record_dir=$8
epoch=200
declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=5)
sample=3500
model_name=lgn_${graph}


dataset=${src}_${tar}
ncore=${ncores[$dataset]}
fullname=${dataset}_${graph}_${test_mode}

if [[ ! -d ./graph ]]
then
	mkdir -p ./graph
fi

if [[ ! -d ./result ]]
then
	mkdir -p ./result
fi

if [[ "$mode" == "train" || "$mode" == "traineval" ]]
then 
	echo Start training $fullname ...
        python3 edit_properties.py --dataset ${fullname} --gpu ${gpu_id}
        python3 main.py
        echo Done training $fullname
fi

if [[ "$mode" == "eval" || "$mode" == "traineval" ]]
then
	echo Start evaluating $fullname ...
        python3 ../../rec_and_eval_ncore.py \
            --data_dir ${data_dir} \
            --test_mode ${test_mode} \
            --save_dir ${exp_record_dir} \
            --save_name M_cpr_D_${src}_${tar}_T_target \
            --user_emb_path $(pwd)/graph/${fullname}_${epoch}epoch.graph \
            --item_emb_path $(pwd)/graph/${fullname}_${epoch}epoch.graph \
            --src ${src} \
            --tar ${tar} \
            --model_name ${model_name}\
            --ncore ${ncore}
        echo Done evaluating $fullname
fi

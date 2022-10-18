#!/bin/bash
mom_save_dir='/TOP/tmp2/cpr/fix_ncore'
src=$1 #(hk spo mt) 
tar=$2 #(csjj csj b)
graph=$3 #(big lil)
te=$4 #(target shared cold)
mode=$5 #train, eval, traineval
gpu_id=$6
epoch=200
declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=5)
sample=3500


dataset=${src}_${tar}
ncore=${ncores[$dataset]}
echo "Make sure you've run build_cython & preprocess for LGN."
fullname=${dataset}_${graph}_${te}

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
            --test_users ${te} \
            --mom_save_dir ${mom_save_dir} \
            --output_file $(pwd)/result/${fullname}_${epoch}_lightgcn_result.txt \
            --graph_file $(pwd)/graph/${fullname}_${epoch}epoch.graph \
            --src ${src} \
            --tar ${tar} \
            --ncore ${ncore} \
            --sample ${sample}
        echo Done evaluating $fullname
fi

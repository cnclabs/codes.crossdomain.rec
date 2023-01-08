#!/bin/bash
data_dir=$1
model_save_dir=$2
sample=3500
declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=5)
epoch=200
worker=16

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
    
    ./cpr \
    -train_ut ${data_dir}/input_${ncore}core/${tar}_train_input.txt \
    -train_us ${data_dir}/input_${ncore}core/all_${src}_train_input.txt \
    -train_ust ${data_dir}/input_${ncore}core/all_cpr_train_u_${src}+${tar}.txt \
    -save ${model_save_dir}/${src}_${tar}_normal.txt \
    -dimension 100 -update_times $((epoch)) -worker ${worker} -init_alpha 0.025 -user_reg 0.01 -item_reg 0.06 
    
    # cold
    ./cpr \
    -train_ut ${data_dir}/input_${ncore}core/cold_${tar}_train_input.txt \
    -train_us ${data_dir}/input_${ncore}core/all_${src}_train_input.txt \
    -train_ust ${data_dir}/input_${ncore}core/cold_cpr_train_u_${src}+${tar}.txt \
    -save ${model_save_dir}/${src}_${tar}_cold.txt \
    -dimension 100 -update_times $((epoch)) -worker ${worker} -init_alpha 0.025 -user_reg 0.01 -item_reg 0.06 
    
done

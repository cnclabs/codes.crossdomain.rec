#!/bin/bash
ncore_data_dir=$1
cpr_input_dir=$2
bpr_input_dir=$3
emb_save_dir=$4
exp_record_dir=$5
mode=$6
n_round=$7

#declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -a datasets=("hk_csjj")

for round in `seq 1 $n_round` ;
do
    for d in "${datasets[@]}";
    do
        IFS='_'
        read -a domains <<< "$d"
        IFS=' '
        src=${domains[0]}
        tar=${domains[1]}
        bash run_lfm_bpr.sh\
	       	${ncore_data_dir}\
		${cpr_input_dir}\
		${bpr_input_dir}\
		${emb_save_dir}/${round}\
		${exp_record_dir}\
		${mode}\
		${src}\
		${tar}
    done 
done

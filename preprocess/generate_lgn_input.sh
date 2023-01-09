#!/bin/bash
cpr_input_dir=$1
bpr_input_dir=$2
lgn_input_dir=$3

declare -a datasets=("hk_csjj" "spo_csj" "mt_b")

for d in "${datasets[@]}";
do
    IFS="_"
    read -a domains <<< "$d"
    IFS=" "
    src=${domains[0]}
    tar=${domains[1]}

    python3 tools/generate_lgn_input.py \
        --input_data_path ${cpr_input_dir}/${tar}_tar_train_input.txt \
        --lgn_input_dir ${lgn_input_dir} \
        --save_file_name ${tar}_tar || exit 1;
    sed -i 's/\"//g' ${lgn_input_dir}/${tar}_tar_train_input.txt || exit 1;

    python3 tools/generate_lgn_input.py \
        --input_data_path ${bpr_input_dir}/${src}_${tar}_src_tar_train_input.txt \
        --lgn_input_dir ${lgn_input_dir} \
        --save_file_name ${src}_${tar}_src_tar || exit 1;
    sed -i 's/\"//g' ${lgn_input_dir}/${src}_${tar}_src_tar_train_input.txt || exit 1;

    python3 tools/generate_lgn_input.py \
        --input_data_path ${bpr_input_dir}/${src}_${tar}_src_ctar_train_input.txt \
        --lgn_input_dir ${lgn_input_dir} \
        --save_file_name ${src}_${tar}_src_ctar || exit 1;
    sed -i 's/\"//g' ${lgn_input_dir}/${src}_${tar}_src_ctar_train_input.txt || exit 1;
done

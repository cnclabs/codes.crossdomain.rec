#!/bin/bash
loo_data_dir=$1
ncore_data_dir=$2
n_testing_user=3500
n_worker=8
declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
ncores=(["hk_csjj"]=5 ["spo_csj"]=5 ["mt_b"]=5)

declare -a test_modes=("target" "shared" "cold")

for d in "${datasets[@]}";
do
    IFS="_"
    read -a domains <<< "$d"
    IFS=" "
    src=${domains[0]}
    tar=${domains[1]}
    ncore=${ncores[$d]}

    python3 tools/convert_to_ncore.py \
    --loo_data_dir ${loo_data_dir} \
    --ncore_data_dir ${ncore_data_dir} \
    --ncore ${ncore} \
    --src ${src} \
    --tar ${tar} \
    --n_testing_user ${n_testing_user}&&

    for test_mode in "${test_modes[@]}";
    do        
        python tools/pre_sample_testing_neg99_pos1.py \
            --ncore_data_dir ${ncore_data_dir}\
            --test_mode ${test_mode}\
            --n_worker ${n_worker}\
            --src ${src}\
            --tar ${tar} || exit 1
    done

done

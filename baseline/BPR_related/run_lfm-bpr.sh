#!/bin/bash
update_times=200
dim=100
declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=10)

for d in "${datasets[@]}"; do
    IFS='_'
	read -a domains <<< "$d"
	IFS=' '
    src=${domains[0]}
	tar=${domains[1]}
	ncore=${ncores[$d]}
    ### tar
    python3 ./BPR/lfm-bpr.py \
    --train ../../input_${ncore}/${tar}_train_input.txt \
    --save ./lfm_bpr_graphs/${tar}_lightfm_bpr_${update_times}_10e-5.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001

    ### eval @tar @shared

    ### src
    python3 ./BPR/lfm-bpr.py \
    --train ../../input_${ncore}/all_${src}_train_input.txt \
    --save ./lfm_bpr_graphs/${src}_lightfm_bpr_${update_times}_10e-5.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001

    ### tar cold
    python3 ./BPR/lfm-bpr.py \
    --train ../../input_${ncore}/cold_${tar}_train_input.txt \
    --save ./lfm_bpr_graphs/cold_${tar}_lightfm_bpr_${update_times}_10e-5.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001


    # all input (target)
    python3 ./BPR/lfm-bpr.py \
    --train ../../input_${ncore}/all_cpr_train_u_${src}+${tar}.txt \
    --save ./lfm_bpr_graphs/${src}+${tar}_lightfm_bpr_${update_times}_10e-5.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001

    ### eval @tar @shared

    # all input (cold)

    python3 ./BPR/lfm-bpr.py \
    --train ../../input_${ncore}/cold_cpr_train_u_${src}+${tar}.txt \
    --save ./lfm_bpr_graphs/cold_${src}+${tar}_lightfm_bpr_${update_times}_10e-5.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001

    ### eval @cold
done
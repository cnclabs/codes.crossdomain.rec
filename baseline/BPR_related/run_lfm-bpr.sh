mom_save_dir=/TOP/tmp2/cpr/from_yzliu
#!/bin/bash
update_times=200
dim=100
declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=10)
sample=4000

for d in "${datasets[@]}"; do
    IFS='_'
	read -a domains <<< "$d"
	IFS=' '
    src=${domains[0]}
	tar=${domains[1]}
	ncore=${ncores[$d]}
    ### tar
    python3 ./BPR/lfm-bpr.py \
    --train ${mom_save_dir}/input_${ncore}core/${tar}_train_input.txt \
    --save ./lfm_bpr_graphs/${tar}_lightfm_bpr_${update_times}_10e-5.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001

    ### eval @tar @shared
    python3 ../../rec_and_eval_ncore.py \
	--test_users target \
        --mom_save_dir ${mom_save_dir} \
	--output_file $(pwd)/lfm_bpr_results/${src}_${tar}_lightfm_bpr_result_${update_times}_target.txt \
	--graph_file $(pwd)/lfm_bpr_graphs/${tar}_lightfm_bpr_${update_times}_10e-5.txt \
	--src ${src} \
	--tar ${tar} \
	--ncore ${ncore} \
	--sample ${sample}

    python3 ../../rec_and_eval_ncore.py \
	--test_users shared \
        --mom_save_dir ${mom_save_dir} \
	--output_file $(pwd)/lfm_bpr_results/${src}_${tar}_lightfm_bpr_result_${update_times}_shared.txt \
	--graph_file $(pwd)/lfm_bpr_graphs/${tar}_lightfm_bpr_${update_times}_10e-5.txt \
	--src ${src} \
	--tar ${tar} \
	--ncore ${ncore} \
	--sample ${sample}

    ### src
    python3 ./BPR/lfm-bpr.py \
    --train ${mom_save_dir}/input_${ncore}core/all_${src}_train_input.txt \
    --save ./lfm_bpr_graphs/${src}_lightfm_bpr_${update_times}_10e-5.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001

    ### tar cold
    python3 ./BPR/lfm-bpr.py \
    --train ${mom_save_dir}/input_${ncore}core/cold_${tar}_train_input.txt \
    --save ./lfm_bpr_graphs/cold_${tar}_lightfm_bpr_${update_times}_10e-5.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001


    # all input (target)
    python3 ./BPR/lfm-bpr.py \
    --train ${mom_save_dir}/input_${ncore}core/all_cpr_train_u_${src}+${tar}.txt \
    --save ./lfm_bpr_graphs/${src}+${tar}_lightfm_bpr_${update_times}_10e-5.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001

    ### eval @tar @shared
    python3 ../../rec_and_eval_ncore.py \
	--test_users target \
        --mom_save_dir ${mom_save_dir} \
	--output_file $(pwd)/lfm_bpr_results/${src}_${tar}_lightfm_bpr+_result_${update_times}_target.txt \
	--graph_file $(pwd)/lfm_bpr_graphs/${src}+${tar}_lightfm_bpr_${update_times}_10e-5.txt \
	--src ${src} \
	--tar ${tar} \
	--ncore ${ncore} \
	--sample ${sample}

    python3 ../../rec_and_eval_ncore.py \
	--test_users shared \
        --mom_save_dir ${mom_save_dir} \
	--output_file $(pwd)/lfm_bpr_results/${src}_${tar}_lightfm_bpr+_result_${update_times}_shared.txt \
	--graph_file $(pwd)/lfm_bpr_graphs/${src}+${tar}_lightfm_bpr_${update_times}_10e-5.txt \
	--src ${src} \
	--tar ${tar} \
	--ncore ${ncore} \
	--sample ${sample}

    # all input (cold)

    python3 ./BPR/lfm-bpr.py \
    --train ${mom_save_dir}/input_${ncore}core/cold_cpr_train_u_${src}+${tar}.txt \
    --save ./lfm_bpr_graphs/cold_${src}+${tar}_lightfm_bpr_${update_times}_10e-5.txt \
    --dim 100 \
    --iter $update_times \
    --worker 20 \
    --item_alpha 0.00001 \
    --user_alpha 0.00001

    ### eval @cold
    python3 ../../rec_and_eval_ncore.py \
	--test_users cold \
        --mom_save_dir ${mom_save_dir} \
	--output_file $(pwd)/lfm_bpr_results/${src}_${tar}_lightfm_bpr+_result_${update_times}_cold.txt \
	--graph_file $(pwd)/lfm_bpr_graphs/cold_${src}+${tar}_lightfm_bpr_${update_times}_10e-5.txt \
	--src ${src} \
	--tar ${tar} \
	--ncore ${ncore} \
	--sample ${sample}
done

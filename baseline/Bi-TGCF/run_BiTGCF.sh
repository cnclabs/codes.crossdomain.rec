#!/bin/bash
mom_save_dir=/TOP/tmp2/cpr/from_yzliu
sample=4000
declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
declare -A gpus
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=10)
gpus=(['hk_csjj']=2 ["spo_csj"]=2 ["mt_b"]=2)

for d in "${datasets[@]}"; do
    IFS='_'
    read -a domains <<< "$d"
    IFS=' '
    src=${domains[0]}
    tar=${domains[1]}
    ncore=${ncores[$d]}
    CUDA_VISIBLE_DEVICES=${gpus[$d]}\
        python3 ./BiTGCF/main.py \
            --data_path ${mom_save_dir}/input_${ncore}core \
            --source_dataset all_${src} \
            --target_dataset ${tar} \
            --epoch 200 \
            --batch_size 65536 \
            --embed_size 25; 
        python3 ../../rec_and_eval_ncore.py \
    	    --mom_save_dir ${mom_save_dir} \
            --test_users target \
            --output_file $(pwd)/result/${src}_${tar}_bitgcf_result_${update_times}_target.txt \
            --graph_file $(pwd)/graph/all_${src}_${tar}_200.graph \
            --src ${src} \
            --tar ${tar} \
            --ncore ${ncore} \
            --sample ${sample};
        python3 ../../rec_and_eval_ncore.py \
    	    --mom_save_dir ${mom_save_dir} \
            --test_users shared \
            --output_file $(pwd)/result/${src}_${tar}_bitgcf_result_${update_times}_shared.txt \
            --graph_file $(pwd)/graph/all_${src}_${tar}_200.graph \
            --src ${src} \
            --tar ${tar} \
            --ncore ${ncore} \
            --sample ${sample};
        python3 ./BiTGCF/main.py\
            --data_path ${mom_save_dir}/input_${ncore}core \
            --source_dataset all_${src} \
            --target_dataset cold_${tar} \
            --epoch 200 \
            --batch_size 65536 \
            --embed_size 25; 
        python3 ../../rec_and_eval_ncore.py \
    	    --mom_save_dir ${mom_save_dir} \
            --test_users cold \
            --output_file $(pwd)/result/${src}_${tar}_bitgcf_result_${update_times}_cold.txt \
            --graph_file $(pwd)/graph/all_${src}_cold_${tar}_200.graph \
            --src ${src} \
            --tar ${tar} \
            --ncore ${ncore} \
            --sample ${sample};
done

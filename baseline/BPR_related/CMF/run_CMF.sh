mom_save_dir='/TOP/tmp2/cpr/fix_ncore/'
#!/bin/bash
set -xe

# update_times=500 
#num_checkpoint=5
declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=5)
workers=8
l2_reg=0.0025
init_alpha=0.025
dim=100
current_split=200

for d in "${datasets[@]}";
do
    IFS='_'
	read -a domains <<< "$d"
	IFS=' '
    src=${domains[0]}
	tar=${domains[1]}
	ncore=${ncores[$d]}

    python3 CMF_run.py \
    --current_epoch $current_split \
    --mom_save_dir ${mom_save_dir} \
    --output_file ./result/new_CMF_${src}+${tar}_${current_split} \
    --k 50 \
    --workers $workers \
    --dataset $d \
    --ncore ${ncore}

    python3 CMF_run_cold.py \
    --current_epoch $current_split \
    --mom_save_dir ${mom_save_dir} \
    --output_file ./result/new_CMF_${src}+${tar}_${current_split}_cold_result.txt \
    --k 50 \
    --workers $workers \
    --dataset $d \
    --ncore ${ncore}
done





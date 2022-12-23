#!/bin/bash
data_dir=$1
model_save_dir=$2
sample=3500
declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=5)

if [[ ! -d ${model_save_dir}/graph ]]
then
	mkdir -p ${model_save_dir}/graph
fi

for d in "${datasets[@]}"; do
	IFS='_'
	read -a domains <<< "$d"
	IFS=' '
	src=${domains[0]}
	tar=${domains[1]}
	ncore=${ncores[$d]}

	for i in $(seq 2 2);
	do
	epoch=$((i*100))

	## ================================= Amazon dataset (SRC-TAR) =================================
	if [ $i == 1 ]; then
		pretrain=""
	else
		pretrain="-pre-train ${model_save_dir}/graph/all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch-100))epoch.txt"
		coldpretrain="-pre-train ${model_save_dir}/graph/cold_all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch-100))epoch.txt"
	fi
	./ADS_crossDomainRec/smore-stack/pre-train_changeUpt_cpr \
	-train_ut ${data_dir}/input_${ncore}core/${tar}_train_input.txt \
	-train_us ${data_dir}/input_${ncore}core/all_${src}_train_input.txt \
	-train_ust ${data_dir}/input_${ncore}core/all_cpr_train_u_${src}+${tar}.txt \
	-save ${model_save_dir}/graph/all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch))epoch.txt \
	-dimension 100 -update_times $((epoch)) -worker 16 -init_alpha 0.025 -user_reg 0.01 -item_reg 0.06 
	#$pretrain

	# cold
	./ADS_crossDomainRec/smore-stack/pre-train_changeUpt_cpr \
	-train_ut ${data_dir}/input_${ncore}core/cold_${tar}_train_input.txt \
	-train_us ${data_dir}/input_${ncore}core/all_${src}_train_input.txt \
	-train_ust ${data_dir}/input_${ncore}core/cold_cpr_train_u_${src}+${tar}.txt \
	-save ${model_save_dir}/graph/cold_all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch))epoch.txt \
	-dimension 100 -update_times $((epoch)) -worker 16 -init_alpha 0.025 -user_reg 0.01 -item_reg 0.06 
	#$coldpretrain

	done
done

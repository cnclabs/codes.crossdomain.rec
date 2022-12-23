#!/bin/bash
data_dir=$1
model_save_dir=$2
exp_record_dir=$3
sample=3500
declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
#declare -a datasets=("hk_csjj")
declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=5)

if [[ ! -d ${model_save_dir}/graph ]]
then
	mkdir -p ${model_save_dir}/graph
	mkdir -p ${model_save_dir}/result
fi

for d in "${datasets[@]}"; do
	IFS='_'
	read -a domains <<< "$d"
	IFS=' '
	src=${domains[0]}
	tar=${domains[1]}
	ncore=${ncores[$d]}
        epoch=200

	python3 ../rec_and_eval_ncore.py \
	--data_dir ${data_dir} \
	--test_mode target \
	--save_dir ${exp_record_dir} \
	--save_name M_cpr_D_${src}_${tar}_T_target \
	--user_emb_path ${model_save_dir}/graph/all_${src}_${tar}_cpr_ug_0.01_ig_0.06_${epoch}epoch.txt \
	--item_emb_path ${model_save_dir}/graph/all_${src}_${tar}_cpr_ug_0.01_ig_0.06_${epoch}epoch.txt \
	--src ${src} \
	--tar ${tar} \
	--model_name cpr\
	--ncore ${ncore}

	python3 ../rec_and_eval_ncore.py \
	--data_dir ${data_dir} \
	--test_mode shared \
	--save_dir ${exp_record_dir} \
	--save_name M_cpr_D_${src}_${tar}_T_shared \
	--user_emb_path ${model_save_dir}/graph/all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch))epoch.txt \
	--item_emb_path ${model_save_dir}/graph/all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch))epoch.txt \
	--src ${src} \
	--tar ${tar} \
	--model_name cpr\
	--ncore ${ncore}

	python3 ../rec_and_eval_ncore.py \
	--data_dir ${data_dir} \
	--test_mode cold \
	--save_dir ${exp_record_dir} \
	--save_name M_cpr_D_${src}_${tar}_T_cold \
	--user_emb_path ${model_save_dir}/graph/cold_all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch))epoch.txt \
	--item_emb_path ${model_save_dir}/graph/cold_all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch))epoch.txt \
	--src ${src} \
	--tar ${tar} \
	--model_name cpr\
	--ncore ${ncore}
done

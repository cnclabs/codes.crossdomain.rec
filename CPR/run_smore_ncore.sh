#!/bin/bash
sample=4000
declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=10)

if [[ ! -d ./result ]]
then
	mkdir -p ./graph
	mkdir -p ./result
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
		pretrain="-pre-train ./graph/all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch-100))epoch.txt"
		coldpretrain="-pre-train ./graph/cold_all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch-100))epoch.txt"
	fi
	./ADS_crossDomainRec/smore-stack/pre-train_changeUpt_cpr \
	-train_ut ../input_${ncore}core/${tar}_train_input.txt \
	-train_us ../input_${ncore}core/all_${src}_train_input.txt \
	-train_ust ../input_${ncore}core/all_cpr_train_u_${src}+${tar}.txt \
	-save ./graph/all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch))epoch.txt \
	-dimension 100 -update_times 200 -worker 16 -init_alpha 0.025 -user_reg 0.01 -item_reg 0.06 
	#$pretrain

	python3 ../rec_and_eval_ncore.py \
	--test_users target \
	--output_file $(pwd)/result/all_${src}_${tar}_cpr_target_result_$((epoch))epoch.txt \
	--graph_file $(pwd)/graph/all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch))epoch.txt \
	--src ${src} \
	--tar ${tar} \
	--ncore ${ncore} \
	--sample ${sample}

	python3 ../rec_and_eval_ncore.py \
	--test_users shared \
	--output_file $(pwd)/result/all_${src}_${tar}_cpr_shared_result_$((epoch))epoch.txt \
	--graph_file $(pwd)/graph/all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch))epoch.txt \
	--src ${src} \
	--tar ${tar} \
	--ncore ${ncore} \
	--sample ${sample}

	# cold
	./ADS_crossDomainRec/smore-stack/pre-train_changeUpt_cpr \
	-train_ut ../input_${ncore}core/cold_${tar}_train_input.txt \
	-train_us ../input_${ncore}core/all_${src}_train_input.txt \
	-train_ust ../input_${ncore}core/cold_cpr_train_u_${src}+${tar}.txt \
	-save ./graph/cold_all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch))epoch.txt \
	-dimension 100 -update_times 200 -worker 16 -init_alpha 0.025 -user_reg 0.01 -item_reg 0.06 
	#$coldpretrain

	python3 ../rec_and_eval_ncore.py \
	--test_users cold \
	--output_file $(pwd)/result/all_${src}_${tar}_cpr_cold_result_$((epoch))epoch.txt \
	--graph_file $(pwd)/graph/cold_all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch))epoch.txt \
	--src ${src} \
	--tar ${tar} \
	--ncore ${ncore} \
	--sample ${sample}
	done
done

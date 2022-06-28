#!/bin/bash
BASEDIR=$(dirname $(realpath "$0"))
count=$1
src=$2
tar=$3
ncore=$4
sample=$5

if [[ -n $count && ! -d ${BASEDIR}/result/$count ]]
then
	mkdir -p ${BASEDIR}/graph/$count
	mkdir -p ${BASEDIR}/result/$count
fi

for i in $(seq 2 2);
do
epoch=$((i*100))

## ================================= Amazon dataset (SRC-TAR) =================================
if [ $i == 1 ]; then
	pretrain=""
else
	pretrain="-pre-train ${BASEDIR}/graph/$((count))/all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch-100))epoch.txt"
	coldpretrain="-pre-train ${BASEDIR}/graph/$((count))/cold_all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch-100))epoch.txt"
fi
${BASEDIR}/ADS_crossDomainRec/smore-stack/pre-train_changeUpt_cpr \
-train_ut ${BASEDIR}/input_${ncore}core/${tar}_train_input.txt \
-train_us ${BASEDIR}/input_${ncore}core/all_${src}_train_input.txt \
-train_ust ${BASEDIR}/input_${ncore}core/all_cpr_train_u_${src}+${tar}.txt \
-save ${BASEDIR}/graph/$((count))/all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch))epoch.txt \
-dimension 100 -update_times 200 -worker 16 -init_alpha 0.025 -user_reg 0.01 -item_reg 0.06 
#$pretrain

python3 ${BASEDIR}/rec_and_eval_ncore.py \
--test_users target \
--output_file ./result/$((count))/all_${src}_${tar}_cpr_target_result_$((epoch))epoch.txt \
--graph_file ./graph/$((count))/all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch))epoch.txt \
--src ${src} \
--tar ${tar} \
--ncore ${ncore} \
--sample ${sample}

python3 ${BASEDIR}/rec_and_eval_ncore.py \
--test_users shared \
--output_file ./result/$((count))/all_${src}_${tar}_cpr_shared_result_$((epoch))epoch.txt \
--graph_file ./graph/$((count))/all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch))epoch.txt \
--src ${src} \
--tar ${tar} \
--ncore ${ncore} \
--sample ${sample}

# cold
${BASEDIR}/ADS_crossDomainRec/smore-stack/pre-train_changeUpt_cpr \
-train_ut ${BASEDIR}/input_${ncore}core/cold_${tar}_train_input.txt \
-train_us ${BASEDIR}/input_${ncore}core/all_${src}_train_input.txt \
-train_ust ${BASEDIR}/input_${ncore}core/cold_cpr_train_u_${src}+${tar}.txt \
-save ${BASEDIR}/graph/$((count))/cold_all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch))epoch.txt \
-dimension 100 -update_times 200 -worker 16 -init_alpha 0.025 -user_reg 0.01 -item_reg 0.06 
#$coldpretrain

python3 ${BASEDIR}/rec_and_eval_ncore.py \
--test_users cold \
--output_file ./result/$((count))/all_${src}_${tar}_cpr_cold_result_$((epoch))epoch.txt \
--graph_file ./graph/$((count))/cold_all_${src}_${tar}_cpr_ug_0.01_ig_0.06_$((epoch))epoch.txt \
--src ${src} \
--tar ${tar} \
--ncore ${ncore} \
--sample ${sample}
done

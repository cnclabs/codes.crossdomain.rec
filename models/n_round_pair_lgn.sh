#!/bin/bash
ncore_data_dir=$1
lgn_input_dir=$2
emb_save_dir=$3
exp_record_dir=$4
mode=$5
nth_round=$6

if [[ ! -d ${emb_save_dir}/${nth_round} ]]
then
	mkdir -p ${emb_save_dir}/${nth_round}
fi

bash single_lgn.sh\
       	${ncore_data_dir}\
	${lgn_input_dir}\
	${emb_save_dir}/${nth_round}\
	${exp_record_dir}\
	${mode}\
	hk\
	csjj\
	0\
	single &
bash single_lgn.sh\
       	${ncore_data_dir}\
	${lgn_input_dir}\
	${emb_save_dir}/${nth_round}\
	${exp_record_dir}\
	${mode}\
	hk\
	csjj\
	0\
	normal &
bash single_lgn.sh\
       	${ncore_data_dir}\
	${lgn_input_dir}\
	${emb_save_dir}/${nth_round}\
	${exp_record_dir}\
	${mode}\
	hk\
	csjj\
	0\
	cold &
bash single_lgn.sh\
       	${ncore_data_dir}\
	${lgn_input_dir}\
	${emb_save_dir}/${nth_round}\
	${exp_record_dir}\
	${mode}\
	spo\
	csj\
	1\
	single &
bash single_lgn.sh\
       	${ncore_data_dir}\
	${lgn_input_dir}\
	${emb_save_dir}/${nth_round}\
	${exp_record_dir}\
	${mode}\
	spo\
	csj\
	1\
	normal &
bash single_lgn.sh\
       	${ncore_data_dir}\
	${lgn_input_dir}\
	${emb_save_dir}/${nth_round}\
	${exp_record_dir}\
	${mode}\
	spo\
	csj\
	1\
	cold &
bash single_lgn.sh\
       	${ncore_data_dir}\
	${lgn_input_dir}\
	${emb_save_dir}/${nth_round}\
	${exp_record_dir}\
	${mode}\
	mt\
	b\
	2\
	single &
bash single_lgn.sh\
       	${ncore_data_dir}\
	${lgn_input_dir}\
	${emb_save_dir}/${nth_round}\
	${exp_record_dir}\
	${mode}\
	mt\
	b\
	2\
	normal &
bash single_lgn.sh\
       	${ncore_data_dir}\
	${lgn_input_dir}\
	${emb_save_dir}/${nth_round}\
	${exp_record_dir}\
	${mode}\
	mt\
	b\
	2\
	cold &

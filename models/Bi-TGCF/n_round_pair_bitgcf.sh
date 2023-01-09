ncore_data_dir=$1
cpr_input_dir=$2
emb_save_dir=$3
exp_record_dir=$4
mode=$5
nth_round=$6

if [[ ! -d ./graph ]]
then
	mkdir -p ${emb_save_dir}/${nth_round}
fi

bash single_bitgcf.sh\
       	${ncore_data_dir}\
	${cpr_input_dir}\
	${emb_save_dir}/${nth_round}\
	${exp_record_dir}\
	${mode}\
	hk\
	csjj\
	0\
	normal &
bash single_bitgcf.sh\
       	${ncore_data_dir}\
	${cpr_input_dir}\
	${emb_save_dir}/${nth_round}\
	${exp_record_dir}\
	${mode}\
	hk\
	csjj\
	0\
	cold &
#bash single_bitgcf.sh\
#       	${ncore_data_dir}\
#	${cpr_input_dir}\
#	${emb_save_dir}/${nth_round}\
#	${exp_record_dir}\
#	${mode}\
#	spo\
#	csj\
#	1\
#	normal &
bash single_bitgcf.sh\
       	${ncore_data_dir}\
	${cpr_input_dir}\
	${emb_save_dir}/${nth_round}\
	${exp_record_dir}\
	${mode}\
	spo\
	csj\
	1\
	cold &
#bash single_bitgcf.sh\
#       	${ncore_data_dir}\
#	${cpr_input_dir}\
#	${emb_save_dir}/${nth_round}\
#	${exp_record_dir}\
#	${mode}\
#	mt\
#	b\
#	2\
#	normal &
bash single_bitgcf.sh\
       	${ncore_data_dir}\
	${cpr_input_dir}\
	${emb_save_dir}/${nth_round}\
	${exp_record_dir}\
	${mode}\
	mt\
	b\
	2\
	cold 

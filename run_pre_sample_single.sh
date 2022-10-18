src=$1
tar=$2
test_mode=$3
data_dir=$4
ncore=5
python pre_sample_testing_users_rec_dict.py \
	--data_dir ${data_dir}\
	--test_mode ${test_mode}\
	--ncore ${ncore}\
	--n_worker 8\
	--src ${src}\
	--tar ${tar}

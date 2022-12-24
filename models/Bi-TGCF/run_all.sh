data_dir=$1
exp_record_dir=$2
mode=$3
bash single_bitgcf.sh hk csjj 0 normal ${data_dir} ${exp_record_dir} ${mode}&
bash single_bitgcf.sh hk csjj 0 cold ${data_dir} ${exp_record_dir} ${mode}&
bash single_bitgcf.sh spo csj 1 normal ${data_dir} ${exp_record_dir} ${mode}&
bash single_bitgcf.sh spo csj 1 cold ${data_dir} ${exp_record_dir} ${mode}&
bash single_bitgcf.sh mt b 2 normal ${data_dir} ${exp_record_dir} ${mode}& 
bash single_bitgcf.sh mt b 2 cold ${data_dir} ${exp_record_dir} ${mode}

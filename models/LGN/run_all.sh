data_dir=$1
exp_record_dir=$2
mode=$3
bash run_multi.sh hk csjj 0 ${mode} ${data_dir} ${exp_record_dir}&
sleep 300s && bash run_multi.sh spo csj 1 ${mode} ${data_dir} ${exp_record_dir}&
sleep 600s && bash run_multi.sh mt b 2 ${mode} ${data_dir} ${exp_record_dir}

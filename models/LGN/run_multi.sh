src=$1
tar=$2
gpu_id=$3
mode=$4
data_dir=$5
exp_record_dir=$6

bash single_LGN.sh ${src} ${tar} big target ${mode} ${gpu_id} ${data_dir} ${exp_record_dir}&
sleep 60s && bash single_LGN.sh ${src} ${tar} big shared ${mode} ${gpu_id} ${data_dir} ${exp_record_dir}&
sleep 120s && bash single_LGN.sh ${src} ${tar} big cold ${mode} ${gpu_id} ${data_dir} ${exp_record_dir}&
sleep 180s && bash single_LGN.sh ${src} ${tar} lil target ${mode} ${gpu_id} ${data_dir} ${exp_record_dir}&
sleep 240s && bash single_LGN.sh ${src} ${tar} lil shared ${mode} ${gpu_id} ${data_dir} ${exp_record_dir}

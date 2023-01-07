data_dir=$1
src=hk
tar=csjj
bash run_pre_sample_single.sh ${src} ${tar} target ${data_dir}&&
bash run_pre_sample_single.sh ${src} ${tar} shared ${data_dir}&&
bash run_pre_sample_single.sh ${src} ${tar} cold ${data_dir}&&

src=mt
tar=b
bash run_pre_sample_single.sh ${src} ${tar} target ${data_dir}&&
bash run_pre_sample_single.sh ${src} ${tar} shared ${data_dir}&&
bash run_pre_sample_single.sh ${src} ${tar} cold ${data_dir}&&

src=spo
tar=csj
bash run_pre_sample_single.sh ${src} ${tar} target ${data_dir}&&
bash run_pre_sample_single.sh ${src} ${tar} shared ${data_dir}&&
bash run_pre_sample_single.sh ${src} ${tar} cold ${data_dir}

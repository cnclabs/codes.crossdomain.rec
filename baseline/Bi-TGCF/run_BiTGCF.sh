
declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
ncores=(['hk_csjj']=5 "spo_csj"=5 "mt_b"=10)
declare -A gpus
gpus=(['hk_csjj']=0 "spo_csj"=1 "mt_b"=0)

for d in "${datasets[@]}"; do
IFS='_'
read -a domains <<< "$d"
IFS=' '
screen -dm -S bitgcf_${d} bash -c "CUDA_VISIBLE_DEVICES=${gpus[$d]} python3 ./BiTGCF/main.py \
     --data_path ../../input_${ncores[$d]}core \
     --source_dataset all_${domains[0]} \
     --target_dataset ${domains[1]} \
     --epoch 200 \
     --batch_size 65536 \
     --embed_size 25; exec sh"

screen -dm -S bitgcf_${d}_cold bash -c "CUDA_VISIBLE_DEVICES=${gpus[$d]} python3 ./BiTGCF/main.py \
     --data_path ../../input_${ncores[$d]}core \
     --source_dataset all_${domains[0]} \
     --target_dataset cold_${domains[1]} \
     --epoch 200 \
     --batch_size 65536 \
     --embed_size 25; exec sh"
done
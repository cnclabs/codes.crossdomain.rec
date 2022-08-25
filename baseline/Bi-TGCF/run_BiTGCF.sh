
declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=10)
declare -A gpus
gpus=(['hk_csjj']=0 ["spo_csj"]=1 ["mt_b"]=0)

for d in "${datasets[@]}"; do
IFS='_'
read -a domains <<< "$d"
IFS=' '
src=${domains[0]}
tar=${domains[1]}
ncore=${ncores[$d]}
screen -dm -S bitgcf_${d} bash -c "CUDA_VISIBLE_DEVICES=${gpus[$d]} python3 ./BiTGCF/main.py \
     --data_path ../../input_${ncore}core \
     --source_dataset all_${src} \
     --target_dataset ${tar} \
     --epoch 200 \
     --batch_size 65536 \
     --embed_size 25; 
     python3 ../../rec_and_eval_ncore.py \
	--test_users target \
	--output_file $(pwd)/result/${src}_${tar}_bitgcf_result_${update_times}_target.txt \
	--graph_file $(pwd)/graph/all_${src}_${tar}_200.graph \
	--src ${src} \
	--tar ${tar} \
	--ncore ${ncore} \
	--sample ${sample};
     python3 ../../rec_and_eval_ncore.py \
	--test_users shared \
	--output_file $(pwd)/result/${src}_${tar}_bitgcf_result_${update_times}_shared.txt \
	--graph_file $(pwd)/graph/all_${src}_${tar}_200.graph \
	--src ${src} \
	--tar ${tar} \
	--ncore ${ncore} \
	--sample ${sample};
     exec sh"

screen -dm -S bitgcf_${d}_cold bash -c "CUDA_VISIBLE_DEVICES=${gpus[$d]} python3 ./BiTGCF/main.py \
     --data_path ../../input_${ncore}core \
     --source_dataset all_${src} \
     --target_dataset cold_${tar} \
     --epoch 200 \
     --batch_size 65536 \
     --embed_size 25;
     python3 ../../rec_and_eval_ncore.py \
	--test_users cold \
	--output_file $(pwd)/result/${src}_${tar}_bitgcf_result_${update_times}_cold.txt \
	--graph_file $(pwd)/graph/all_${src}_cold_${tar}_200.graph \
	--src ${src} \
	--tar ${tar} \
	--ncore ${ncore} \
	--sample ${sample};
     exec sh"
done
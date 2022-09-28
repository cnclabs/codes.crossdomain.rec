#!/bin/bash

datasets=(hk_csjj spo_csj mt_b)
graphs=(big lil)
tests=(target shared cold)
epoch=200
declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=10)
sample=4000

nowgpu=0

echo "Make sure you've run build_cython & preprocess for LGN."

for dataset in "${datasets[@]}";
do
	IFS='_'
	read -a domains <<< "$dataset"
	IFS=' '
	src=${domains[0]}
	tar=${domains[1]}
	ncore=${ncores[$dataset]}
	for graph in "${graphs[@]}";
	do
		read -p 'Do you want to change GPU? (y/n)' answer
		if [[ $answer = y || $answer = Y ]]
		then
			pregpu=$nowgpu
			read -p 'Which GPU ID? ' nowgpu
			python3 edit_properties.py --gpu ${nowgpu}
		fi

		for te in "${tests[@]}";
		do
			if [[ $graph != lil || $te != cold ]]
			then
				fullname=${dataset}_${graph}_${te}
				python3 edit_properties.py --dataset ${fullname}
				python3 main.py
				python3 ../../rec_and_eval_ncore.py \
				--test_users ${te} \
				--output_file $(pwd)/result/${src}_${tar}_lightgcn_result_${epoch}_${te}.txt \
				--graph_file $(pwd)/graph/${fullname}_${epoch}epoch.graph \
				--src ${src} \
				--tar ${tar} \
				--ncore ${ncore} \
				--sample ${sample}
				#sleep 1s	
				echo Start training $fullname !
			fi
		
		done
	done
done

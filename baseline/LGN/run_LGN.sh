#!/bin/bash

datasets=(hk_csjj spo_csj mt_b)
graphs=(big lil)
tests=(target shared cold)

nowgpu=0

echo "Make sure you've run build_cython & preprocess for LGN."

for dataset in "${datasets[@]}";
do
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
				screen -dm -S ${fullname} bash -c 'python3 main.py; exec sh'
				#sleep 1s	
				echo Start training $fullname !
			fi
		
		done
	done
done
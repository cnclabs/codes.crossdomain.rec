#mode=train
#mode=eval
#gpu_id=2
#mode=traineval

#src=hk
#tar=csjj

#src=spo
#tar=csj
#
#src=mt
#tar=b

src=$1
tar=$2
gpu_id=$3
mode=$4

bash single_LGN.sh ${src} ${tar} big target ${mode} ${gpu_id}&
bash single_LGN.sh ${src} ${tar} big shared ${mode} ${gpu_id}&
bash single_LGN.sh ${src} ${tar} big cold ${mode} ${gpu_id}&
bash single_LGN.sh ${src} ${tar} lil target ${mode} ${gpu_id}&
bash single_LGN.sh ${src} ${tar} lil shared ${mode} ${gpu_id}

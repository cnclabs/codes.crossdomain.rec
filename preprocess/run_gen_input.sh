mom_save_dir=$1
save_dir=$2
cold_sample=3500
declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=5)

if [[ ! -d ${save_dir} ]]
then
	mkdir -p ${save_dir}
fi


for d in "${datasets[@]}"; do
    IFS='_'
    read -a domains <<< "$d"
    IFS=' '
    src=${domains[0]}
	tar=${domains[1]}
	ncore=${ncores[$d]}

    python3 convert_to_ncore.py \
    --mom_save_dir ${mom_save_dir} \
    --save_dir ${save_dir} \
    --ncore ${ncore} \
    --src ${src} \
    --tar ${tar} \
    --cold_sample ${cold_sample}

    python3  generate_ncore_input.py \
    --save_dir ${save_dir} \
    --ncore ${ncore} \
    --src ${src} \
    --tar ${tar} \
    --n_testing_user ${cold_sample}
done

loo_data_dir=$1
ncore_data_dir=$2
cold_sample=3500
declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=5)

if [[ ! -d ${processed_data_dir} ]]
then
	mkdir -p ${processed_data_dir}
fi


for d in "${datasets[@]}"; do
    IFS='_'
    read -a domains <<< "$d"
    IFS=' '
    src=${domains[0]}
    tar=${domains[1]}
    ncore=${ncores[$d]}

    python3 convert_to_ncore.py \
    --loo_data_dir ${loo_data_dir} \
    --ncore_data_dir ${ncore_data_dir} \
    --ncore ${ncore} \
    --src ${src} \
    --tar ${tar} \
    --cold_sample ${cold_sample}

    python3  generate_ncore_input.py \
    --ncore_data_dir ${ncore_data_dir} \
    --ncore ${ncore} \
    --src ${src} \
    --tar ${tar} \
    --n_testing_user ${cold_sample}
done

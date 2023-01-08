loo_data_dir=$1
sample_dir=$2
ncore_data_dir=${sample_dir}/ncore_data
cpr_input_dir=${sample_dir}/cpr_input
n_testing_user=3500

declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
ncores=(["hk_csjj"]=5 ["spo_csj"]=5 ["mt_b"]=5)

for d in "${datasets[@]}";
do
    IFS="_"
    read -a domains <<< "$d"
    IFS=" "
    src=${domains[0]}
    tar=${domains[1]}
    ncore=${ncores[$d]}

    python3 convert_to_ncore.py \
    --loo_data_dir ${loo_data_dir} \
    --ncore_data_dir ${ncore_data_dir} \
    --ncore ${ncore} \
    --src ${src} \
    --tar ${tar} \
    --n_testing_user ${n_testing_user}&&

    python3 generate_cpr_input.py \
    --ncore_data_dir ${ncore_data_dir} \
    --cpr_input_dir ${cpr_input_dir} \
    --src ${src} \
    --tar ${tar} || exit 1;
done

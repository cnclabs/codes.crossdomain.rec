ncore_data_dir=$1
cpr_input_dir=$2

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

    python3 tools/generate_cpr_input.py \
        --ncore_data_dir ${ncore_data_dir} \
        --cpr_input_dir ${cpr_input_dir} \
        --src ${src} \
        --tar ${tar} || exit 1;
done

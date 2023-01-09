cpr_input_dir=$1
bpr_input_dir=$2

declare -a datasets=("hk_csjj" "spo_csj" "mt_b")

for d in "${datasets[@]}";
do
    IFS="_"
    read -a domains <<< "$d"
    IFS=" "
    src=${domains[0]}
    tar=${domains[1]}

    python3 tools/generate_bpr_input.py \
        --cpr_input_dir ${cpr_input_dir} \
        --bpr_input_dir ${bpr_input_dir} \
        --src ${src} \
        --tar ${tar} || exit 1;
done

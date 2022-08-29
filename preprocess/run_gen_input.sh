cold_sample=4000
declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=10)

for d in "${datasets[@]}"; do
    IFS='_'
    read -a domains <<< "$d"
    IFS=' '
    src=${domains[0]}
	tar=${domains[1]}
	ncore=${ncores[$d]}

    python3 convert_to_ncore.py \
    --ncore ${ncore} \
    --src ${src} \
    --tar ${tar} \
    --cold_sample ${cold_sample}

    python3  generate_ncore_input.py \
    --ncore ${ncore} \
    --src ${src} \
    --tar ${tar}
done
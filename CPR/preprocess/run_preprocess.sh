declare -a datasets=("hk_csjj")
declare -A ncores
ncores=(['hk_csjj']=5)
for d in "${datasets[@]}"; do
    IFS='_'
    read -a domains <<< "$d"
    IFS=' '
    python3  generate_ncore_input.py \
    --ncore ${ncores[$d]} \
    --src ${domains[0]} \
    --tar ${domains[1]}
done
datasets=(hk_csjj)
declare -A ncores
ncores=(['hk_csjj']=5)

for d in "${datasets[@]}"; do
	python3 generate_ncore_input.py --dataset ${d} --ncore ${ncores[$d]}
done

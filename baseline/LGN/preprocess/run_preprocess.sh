declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
mom_save_dir='/TOP/tmp2/cpr/fix_ncore/'
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=5)

for d in "${datasets[@]}"; do
	python3 generate_ncore_input.py --dataset ${d} --ncore ${ncores[$d]} --mom_save_dir ${mom_save_dir} --sample 3500
done

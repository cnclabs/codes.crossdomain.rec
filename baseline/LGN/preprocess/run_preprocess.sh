declare -a datasets=("hk_csjj" "spo_csj" "mt_b")
declare -A ncores
mom_save_dir='/TOP/tmp2/cpr/from_yzliu/'
ncores=(['hk_csjj']=5 ["spo_csj"]=5 ["mt_b"]=10)

for d in "${datasets[@]}"; do
	python3 generate_ncore_input.py --dataset ${d} --ncore ${ncores[$d]} --mom_save_dir ${mom_save_dir}
done

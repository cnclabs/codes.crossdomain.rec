raw_dir=$1
save_dir=$2

declare a datasets=("hk" "csjj" "spo" "csj" "mt" "b")

declare -A fullname_map
fullname_map=(\
	['hk']='Home_and_Kitchen_5.json'\
       	['csjj']='Clothing_Shoes_and_Jewelry_5.json'\
       	['spo']='Sports_and_Outdoors_5.json'\
        ['csj']='Clothing_Shoes_and_Jewelry_5.json'\
	['mt']='Movies_and_TV_5.json'\
	['b']='Books_5.json'\
)

if [[ ! -d ${save_dir} ]]
then
	mkdir -p ${save_dir}
fi

for d in "${datasets[@]}"; do
    fullname=${fullname_map[$d]}	
    python raw_to_LOO.py \
    	--raw_data_path ${raw_dir}/${fullname} \
    	--dataset_brief_name $d\
    	--save_dir $save_dir

done

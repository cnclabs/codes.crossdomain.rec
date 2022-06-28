raw_tar=Clothing_Shoes_and_Jewelry_5.json
raw_src=Home_and_Kitchen_5.json
tar=csjj
src=hk
core=5 # if you don't want to do core-filtering, set this to 0
cold_sample=4000
eval_sample=4000

# From amazon raw data to raw LOO data
# Note that this step is quite time consuming, you can skip this step by copy LOO_data_0
#./get_raw_data.sh ${raw_tar}.gz ${raw_src}.gz
#python3 preprocess/LOO_preprocess.py --raw_data ${raw_tar} --dataset_name ${tar}
#python3 preprocess/LOO_preprocess.py --raw_data ${raw_src} --dataset_name ${src}

# Filtered LOO_data & user to LOO_data_ncore & user_ncore
python3  LOO_data_0/convert_to_ncore.py \
--ncore ${core} \
--src ${src} \
--tar ${tar} \
--cold_sample ${cold_sample}

# Generated input_ncore from LOO_data_ncore & user_ncore
python3  CPR/preprocess/generate_ncore_input.py \
--ncore ${core} \
--src ${src} \
--tar ${tar}

# Train and eval
./CPR/run_smore_ncore.sh 0 ${src} ${tar} ${core} ${eval_sample}

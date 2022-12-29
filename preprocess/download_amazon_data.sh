raw_data_save_dir=$1

download_url=https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall
wget -P ${raw_data_save_dir} ${download_url}/Home_and_Kitchen_5.json.gz --no-check-certificate &
wget -P ${raw_data_save_dir} ${download_url}/Clothing_Shoes_and_Jewelry_5.json.gz --no-check-certificate &
wget -P ${raw_data_save_dir} ${download_url}/Movies_and_TV_5.json.gz --no-check-certificate &
wget -P ${raw_data_save_dir} ${download_url}/Books_5.json.gz --no-check-certificate &
wget -P ${raw_data_save_dir} ${download_url}/Sports_and_Outdoors_5.json.gz --no-check-certificate

# unzip gz file
gzip -d ${raw_data_save_dir}/*

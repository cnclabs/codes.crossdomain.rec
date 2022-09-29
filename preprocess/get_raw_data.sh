raw_data_save_dir='/TOP/tmp2/cpr/official/raw_data/'

download_url=http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/

wget -P ${raw_data_save_dir} ${download_url}/Home_and_Kitchen_5.json.gz &
wget -P ${raw_data_save_dir} ${download_url}/Clothing_Shoes_and_Jewelry_5.json.gz &
wget -P ${raw_data_save_dir} ${download_url}/Movies_and_TV_5.json.gz &
wget -P ${raw_data_save_dir} ${download_url}/Books_5.json.gz &
wget -P ${raw_data_save_dir} ${download_url}/Sports_and_Outdoors_5.json.gz

# unzip gz file
gzip -d ${raw_data_save_dir}/*

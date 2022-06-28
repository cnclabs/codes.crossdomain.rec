raw_tar=$1
raw_src=$2
download_url=http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/

# download raw data from amazon official site https://nijianmo.github.io/amazon/index.html
# wget -P ./raw_data/ ${download_url}Home_and_Kitchen_5.json.gz
# wget -P ./raw_data/ ${download_url}Clothing_Shoes_and_Jewelry_5.json.gz
wget -P ./raw_data/ ${download_url}${raw_tar}
wget -P ./raw_data/ ${download_url}${raw_src}

# unzip gz file
gzip -d ./raw_data/*

#!/bin/bash

# ====== KK ========
python3 train_kk.py \
--test_users target

python3 train_kk.py \
--test_users shared

# ====== CSJ-HK ========
python3 train_amazon_hk.py \
--test_users target

python3 train_amazon_hk.py \
--test_users shared

# ====== MT-B ========
python3 train_amazon_books.py \
--test_users target

python3 train_amazon_books.py \
--test_users shared
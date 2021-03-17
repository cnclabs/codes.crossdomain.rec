#!/bin/bash

# KK
python3 CMF_kk.py \
--test_users target \
--output_file ./result/CMF_kk_target_result.txt \
--k 50

python3 CMF_kk.py \
--test_users shared \
--output_file ./result/CMF_kk_shared_result.txt \
--k 50

python3 CMF_kk_cold.py \
--output_file ./result/CMF_kk_cold_result.txt \
--k 50

# CSJ-HK
python3 CMF_csj_hk.py \
--test_users target \
--output_file ./result/CMF_csj_hk_target_result.txt \
--k 50

python3 CMF_csj_hk.py \
--test_users shared \
--output_file ./result/CMF_csj_hk_shared_result.txt \
--k 50

python3 CMF_csj_hk_cold.py \
--output_file ./result/CMF_csj_hk_cold_result.txt \
--k 50

# MT-B
python3 CMF_mt_books.py \
--test_users target \
--output_file ./result/CMF_mt_books_target_result.txt \
--k 50

python3 CMF_mt_books.py \
--test_users shared \
--output_file ./result/CMF_mt_books_shared_result.txt \
--k 50

python3 CMF_mt_books_cold.py \
--output_file ./result/CMF_mt_books_cold_result.txt \
--k 50






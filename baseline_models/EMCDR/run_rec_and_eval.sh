#!/bin/bash

# ======== KK ========
python3 rec_and_eval_tv_vod.py \
--test_users target \
--output_file ./tv_vod/result/tv_vod_target_result.txt

python3 rec_and_eval_tv_vod.py \
--test_users shared \
--output_file ./tv_vod/result/tv_vod_shared_result.txt

python3 rec_and_eval_tv_vod_cold.py \
--output_file ./tv_vod/result/tv_vod_cold_result.txt


# ======== CSJ-HK ========
python3 rec_and_eval_csj_hk.py \
--test_users target \
--output_file ./csj_hk/result/csj_hk_target_result.txt

python3 rec_and_eval_csj_hk.py \
--test_users shared \
--output_file ./csj_hk/result/csj_hk_shared_result.txt

python3 rec_and_eval_csj_hk_cold.py \
--output_file ./csj_hk/result/csj_hk_cold_result.txt

# ======== MT-B ========
python3 rec_and_eval_mt_books.py \
--test_users target \
--output_file ./mt_books/result/mt_books_target_result.txt

python3 rec_and_eval_mt_books.py \
--test_users shared \
--output_file ./mt_books/result/mt_books_shared_result.txt

python3 rec_and_eval_mt_books_cold.py \
--output_file ./mt_books/result/mt_books_cold_result.txt
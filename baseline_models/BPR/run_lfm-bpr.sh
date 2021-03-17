#!/bin/bash

## ================================= KK dataset =================================

### vod
python3 lfm-bpr.py \
--train ../../CPR/input/vod_train_input.txt \
--save ./graph/vod_lightfm_bpr_10e-5.txt \
--dim 100 \
--iter 100 \
--worker 50 \
--item_alpha 0.00001 \
--user_alpha 0.00001

python3 rec_and_eval_kk.py \
--test_users target \
--output_file ./result/vod_lightfm_bpr_target_result.txt \
--graph_file ./graph/vod_lightfm_bpr_10e-5.txt

python3 rec_and_eval_kk.py \
--test_users shared \
--output_file ./result/vod_lightfm_bpr_shared_result.txt \
--graph_file ./graph/vod_lightfm_bpr_10e-5.txt

### tv
python3 lfm-bpr.py \
--train ../../CPR/input/all_tv_train_input.txt \
--save ./graph/tv_lightfm_bpr_10e-5.txt \
--dim 100 \
--iter 100 \
--worker 50 \
--item_alpha 0.00001 \
--user_alpha 0.00001

### vod cold
python3 lfm-bpr.py \
--train  ../../CPR/input/cold_vod_train_input.txt \
--save ./graph/cold_vod_lightfm_bpr_10e-5.txt \
--dim 100 \
--iter 100 \
--worker 50 \
--item_alpha 0.00001 \
--user_alpha 0.00001

## ================================= CSJ-HK =================================

### hk
python3 lfm-bpr.py \
--train ../../CPR/input/hk_train_input.txt \
--save ./graph/hk_lightfm_bpr_10e-5.txt \
--dim 100 \
--iter 100 \
--worker 50 \
--item_alpha 0.00001 \
--user_alpha 0.00001

python3 rec_and_eval_csj_hk.py \
--test_users target \
--output_file ./result/hk_lightfm_bpr_target_result.txt \
--graph_file ./graph/hk_lightfm_bpr_10e-5.txt

python3 rec_and_eval_csj_hk.py \
--test_users shared \
--output_file ./result/hk_lightfm_bpr_shared_result.txt \
--graph_file ./graph/hk_lightfm_bpr_10e-5.txt

### csj
python3 lfm-bpr.py \
--train ../../CPR/input/all_csj_train_input.txt \
--save ./graph/csj_lightfm_bpr_10e-5.txt \
--dim 100 \
--iter 100 \
--worker 50 \
--item_alpha 0.00001 \
--user_alpha 0.00001

### hk cold
python3 lfm-bpr.py \
--train ../../CPR/input/cold_hk_train_input.txt \
--save ./graph/cold_hk_lightfm_bpr_10e-5.txt \
--dim 100 \
--iter 100 \
--worker 50 \
--item_alpha 0.00001 \
--user_alpha 0.00001

## ================================= MT-B =================================

### books
python3 lfm-bpr.py \
--train ../../CPR/input/books_train_input.txt \
--save ./graph/books_lightfm_bpr_10e-5.txt \
--dim 100 \
--iter 100 \
--worker 50 \
--item_alpha 0.00001 \
--user_alpha 0.00001

python3 rec_and_eval_mt_books.py \
--test_users target \
--output_file ./result/books_lightfm_bpr_target_result.txt \
--graph_file ./graph/books_lightfm_bpr_10e-5.txt

python3 rec_and_eval_mt_books.py \
--test_users shared \
--output_file ./result/books_lightfm_bpr_target_result.txt \
--graph_file ./graph/books_lightfm_bpr_10e-5.txt

### mt
python3 lfm-bpr.py \
--train ../../CPR/input/all_mt_train_input.txt \
--save ./graph/mt_lightfm_bpr_10e-5.txt \
--dim 100 \
--iter 100 \
--worker 50 \
--item_alpha 0.00001 \
--user_alpha 0.00001

### books cold
python3 lfm-bpr.py \
--train ../../CPR/input/cold_books_train_input.txt \
--save ./graph/cold_books_lightfm_bpr_10e-5.txt \
--dim 100 \
--iter 100 \
--worker 50 \
--item_alpha 0.00001 \
--user_alpha 0.00001






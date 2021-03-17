#!/bin/bash

# ======== KK ========
python3 MLP.py \
--epoch_log ./tv_vod/model_log/MLP_lightfm_tv_vod.txt \
--model_save_dir ./tv_vod/model \
--Us ./tv_vod/lightfm_bpr_Us.pickle \
--Ut ./tv_vod/lightfm_bpr_Ut.pickle

# cold
python3 MLP.py \
--epoch_log ./tv_vod/model_log_cold/MLP_lightfm_tv_vod_cold.txt \
--model_save_dir ./tv_vod/model_cold \
--Us ./tv_vod/lightfm_bpr_Us_cold.pickle \
--Ut ./tv_vod/lightfm_bpr_Ut_cold.pickle

# ======== CSJ-HK ========
python3 MLP.py \
--epoch_log ./csj_hk/model_log/MLP_lightfm_csj_hk.txt \
--model_save_dir ./csj_hk/model \
--Us ./csj_hk/lightfm_bpr_Us.pickle \
--Ut ./csj_hk/lightfm_bpr_Ut.pickle

# cold
python3 MLP.py \
--epoch_log ./csj_hk/model_log_cold/MLP_lightfm_csj_hk_cold.txt \
--model_save_dir ./csj_hk/model_cold \
--Us ./csj_hk/lightfm_bpr_Us_cold.pickle \
--Ut ./csj_hk/lightfm_bpr_Ut_cold.pickle

# ======== MT-B ========
python3 MLP.py \
--epoch_log ./mt_books/model_log/MLP_lightfm_mt_books.txt \
--model_save_dir ./mt_books/model \
--Us ./mt_books/lightfm_bpr_Us.pickle \
--Ut ./mt_books/lightfm_bpr_Ut.pickle

# cold
python3 MLP.py \
--epoch_log ./mt_books/model_log_cold/MLP_lightfm_mt_books_cold.txt \
--model_save_dir ./mt_books/model_cold \
--Us ./mt_books/lightfm_bpr_Us_cold.pickle \
--Ut ./mt_books/lightfm_bpr_Ut_cold.pickle



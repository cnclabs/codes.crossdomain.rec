#!/bin/bash

# ======== KK ========
python3 infer_Us.py \
--meta_path tv_vod/model/mlp_epoch_1000/mlp_epoch_1000.meta \
--ckpt_path tv_vod/model/mlp_epoch_1000 \
--dataset_name tv_vod 

# cold
python3 infer_Us_cold.py \
--meta_path tv_vod/model_cold/mlp_epoch_1000/mlp_epoch_1000.meta \
--ckpt_path tv_vod/model_cold/mlp_epoch_1000 \
--dataset_name tv_vod 

# ======== CSJ-HK ========
python3 infer_Us.py \
--meta_path csj_hk/model/mlp_epoch_1000/mlp_epoch_1000.meta \
--ckpt_path csj_hk/model/mlp_epoch_1000 \
--dataset_name csj_hk 

# cold
python3 infer_Us_cold.py \
--meta_path csj_hk/model_cold/mlp_epoch_1000/mlp_epoch_1000.meta \
--ckpt_path csj_hk/model_cold/mlp_epoch_1000 \
--dataset_name csj_hk 

# ======== MT-B ========
python3 infer_Us.py \
--meta_path mt_books/model/mlp_epoch_1000/mlp_epoch_1000.meta \
--ckpt_path mt_books/model/mlp_epoch_1000 \
--dataset_name mt_books 

# cold
python3 infer_Us_cold.py \
--meta_path mt_books/model_cold/mlp_epoch_1000/mlp_epoch_1000.meta \
--ckpt_path mt_books/model_cold/mlp_epoch_1000 \
--dataset_name mt_books 

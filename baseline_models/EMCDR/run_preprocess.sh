#!/bin/bash

# tv_vod
python3 preprocess.py \
--dataset_name tv_vod

python3 preprocess_cold.py \
--dataset_name tv_vod

# csj_hk
python3 preprocess.py \
--dataset_name csj_hk

python3 preprocess_cold.py \
--dataset_name csj_hk

# mt_books
python3 preprocess.py \
--dataset_name mt_books

python3 preprocess_cold.py \
--dataset_name mt_books
#!/bin/bash

python3 LOO_preprocess_tv.py
python3 LOO_preprocess_vod.py
python3 LOO_preprocess_csj.py
python3 LOO_preprocess_hk.py
python3 LOO_preprocess_mt.py
python3 LOO_preprocess_books.py
python3 user_preprocess.py

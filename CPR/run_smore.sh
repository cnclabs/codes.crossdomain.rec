#!/bin/bash


## ================================= KK dataset =================================
./ADS_crossDomainRec/smore-stack/cpr \
-train_ut ./input/vod_train_input.txt \
-train_us ./input/all_tv_train_input.txt \
-train_ust ./input/all_cpr_train_u_tv+vod.txt \
-save ./graph/all_tv_vod_cpr_ug_0.0025_ig_0.0025.txt \
-dimension 100 -update_times 200 -worker 16 -init_alpha 0.025 -user_reg 0.0025 -item_reg 0.0025

python3 rec_and_eval_kk.py \
--test_users target \
--output_file ./result/all_tv_vod_cpr_target_result.txt \
--graph_file ./graph/all_tv_vod_cpr_ug_0.0025_ig_0.0025.txt

python3 rec_and_eval_kk.py \
--test_users shared \
--output_file ./result/all_tv_vod_cpr_shared_result.txt \
--graph_file ./graph/all_tv_vod_cpr_ug_0.0025_ig_0.0025.txt

# cold
./ADS_crossDomainRec/smore-stack/cpr \
-train_ut ./input/cold_vod_train_input.txt \
-train_us ./input/all_tv_train_input.txt \
-train_ust ./input/cold_cpr_train_u_tv+vod.txt \
-save ./graph/cold_all_tv_vod_cpr_ug_0.0025_ig_0.0025.txt \
-dimension 100 -update_times 200 -worker 16 -init_alpha 0.025 -user_reg 0.0025 -item_reg 0.0025


python3 rec_and_eval_kk.py \
--test_users cold \
--output_file ./result/all_tv_vod_cpr_cold_result.txt \
--graph_file ./graph/cold_all_tv_vod_cpr_ug_0.0025_ig_0.0025.txt

## ================================= Amazon dataset (CSJ-HK) =================================
./ADS_crossDomainRec/smore-stack/cpr \
-train_ut ./input/hk_train_input.txt \
-train_us ./input/all_csj_train_input.txt \
-train_ust ./input/all_cpr_train_u_csj+hk.txt \
-save ./graph/all_csj_hk_cpr_ug_0.0025_ig_0.0025.txt \
-dimension 100 -update_times 200 -worker 16 -init_alpha 0.025 -user_reg 0.0025 -item_reg 0.0025

python3 rec_and_eval_csj_hk.py \
--test_users target \
--output_file ./result/all_csj_hk_cpr_target_result.txt \
--graph_file ./graph/all_csj_hk_cpr_ug_0.0025_ig_0.0025.txt

python3 rec_and_eval_csj_hk.py \
--test_users shared \
--output_file ./result/all_csj_hk_cpr_shared_result.txt \
--graph_file ./graph/all_csj_hk_cpr_ug_0.0025_ig_0.0025.txt

# cold
./ADS_crossDomainRec/smore-stack/cpr \
-train_ut ./input/cold_hk_train_input.txt \
-train_us ./input/all_csj_train_input.txt \
-train_ust ./input/cold_cpr_train_u_csj+hk.txt \
-save ./graph/cold_all_csj_hk_cpr_ug_0.0025_ig_0.0025.txt \
-dimension 100 -update_times 200 -worker 16 -init_alpha 0.025 -user_reg 0.0025 -item_reg 0.0025

python3 rec_and_eval_csj_hk.py \
--test_users cold \
--output_file ./result/all_csj_hk_cpr_cold_result.txt \
--graph_file ./graph/cold_all_csj_hk_cpr_ug_0.0025_ig_0.0025.txt

## ================================= Amazon dataset (MT-B) =================================
./ADS_crossDomainRec/smore-stack/cpr \
-train_ut ./input/books_train_input.txt \
-train_us ./input/all_mt_train_input.txt \
-train_ust ./input/all_cpr_train_u_mt+books.txt \
-save ./graph/all_mt_books_cpr_ug_0.0025_ig_0.0025.txt \
-dimension 100 -update_times 200 -worker 16 -init_alpha 0.025 -user_reg 0.0025 -item_reg 0.0025

python3 rec_and_eval_mt_books.py \
--test_users target \
--output_file ./result/all_mt_books_cpr_target_result.txt \
--graph_file ./graph/all_mt_books_cpr_ug_0.0025_ig_0.0025.txt

python3 rec_and_eval_mt_books.py \
--test_users shared \
--output_file ./result/all_mt_books_cpr_shared_result.txt \
--graph_file ./graph/all_mt_books_cpr_ug_0.0025_ig_0.0025.txt

# cold
./ADS_crossDomainRec/smore-stack/cpr \
-train_ut ./input/cold_books_train_input.txt \
-train_us ./input/all_mt_train_input.txt \
-train_ust ./input/cold_cpr_train_u_mt+books.txt \
-save ./graph/cold_all_mt_books_cpr_ug_0.0025_ig_0.0025.txt \
-dimension 100 -update_times 200 -worker 16 -init_alpha 0.025 -user_reg 0.0025 -item_reg 0.0025

python3 rec_and_eval_mt_books.py \
--test_users cold \
--output_file ./result/all_mt_books_cpr_cold_result.txt \
--graph_file ./graph/cold_all_mt_books_cpr_ug_0.0025_ig_0.0025.txt


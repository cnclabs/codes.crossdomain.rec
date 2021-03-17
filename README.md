# codes.crossdomain.rec

1. Put the following raw data under the raw_data directory
    - interaction_log_v6_20200724_train.parquet
    - Clothing_Shoes_and_Jewelry_5.json
    - Home_and_Kitchen_5.json
    - Movies_and_TV_5.json
    - Books_5.json
3. cd to ./preprocess, and then use `bash run_preprocess.sh` to run the script

5. cd back to the previous directory
6. CPR model
    (1) cd to ./CPR 
    (2) cd to ./preprocess, and then use `python3 generate_input.py` to run the script
    (3) cd back to the previous directory 
    (4) use `bash run_smore.sh` to run the script
5. cd back to the previous directory
6. BPR model
    (1) cd to baseline_models/BPR
    (2) use `bash run_lfm-bpr.sh` to run the script
7. cd back to the previous directory
8. DDTCDR
    (1) cd to ./DDTCDR
    (2) run `preprocess_DDTCDR_input.ipynb`
    (3) use `bash run.sh` to run the script
9. cd back to the previous directory
10. CMF (run on cfda3)
    (1) cd to./CMF
    (2) requirement: `cmfrec==2.4.1` 
    (3) use `bash run_CMF.sh` to run the script 
    
11. cd back to the previous directory
12. EMCDR
    (1) cd to ./EMCDR
    (2) requirement: ```
python==3.6, 
tensorflow==1.14```
    (3) use `bash run_preprocess.sh` to run the script
    (4) use `bash run_MLP.sh` to run the script
    (5) use  `bash run_infer_Us.sh` to run the script
    (6) use `bash run_rec_and_eval.sh` to run the script

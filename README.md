# CPR-2023

## 0. Environment
a. `Environment-A` (CPR, BPR, EMCDR's evaluation)  
- docker image: `nvcr.io/nvidia/pytorch:22.05-py3`
- pip install -r `requirements_env_a.txt`

b. `Environment-B` (LightGCN, Bi-TGCF)  
- docker image: `nvcr.io/nvidia/tensorflow:22.08-tf2-py3`
- pip install -r `requirements_env_b.txt`

c. `Environment-C` (EMCDR's training)  
- docker image: `tensorflow/tensorflow:1.14.0-gpu-py3`
- pip install -r `requirements_env_c.txt`

## 1. Dataset & Preprocessing
We use 3 pairs of datasets **(Source_Target)**:
* MT_B (Movies_and_TV_5.json + Books_5.json)
* SPO_CSJ (Sports_and_Outdoors_5.json + Clothing_Shoes_and_Jewelry_5.json)
* HK_CSJJ (Home_and_Kitchen_5.json + Clothing_Shoes_and_Jewelry_5.json)
```
[Step-1] Gen input
$ cd preprocess
$ bash run_gen_input.sh {raw_data_dir} {processed_data_dir}

e.g., 
$ bash run_gen_input.sh /TOP/tmp2/cpr/from_yzliu/ /TOP/tmp2/cpr/fix_ncore_test

[Step-2] Pre-sample negative pairs for target/shared/cold testing users
$ cd ..
$ bash run_pre_sample_all.sh {processed_data_dir}

e.g.,
$ bash run_pre_sample_all.sh /TOP/tmp2/cpr/fix_ncore_test/
```


## 2. Model Training & Evaluation
### a. CPR
Use `Environment-A`
```
$ cd CPR 
$ ./run_smore_ncore.sh {processed_data_dir} {model_save_dir}
$ ./run_eval.sh {processed_data_dir} {model_save_dir} {exp_record_dir}

e.g.,
$ bash run_smore_ncore.sh /TOP/tmp2/cpr/fix_ncore_test/ /TOP/tmp2/cpr/fix_ncore_test/experiments/cpr/
$ bash run_eval.sh /TOP/tmp2/cpr/fix_ncore_test/ /TOP/tmp2/cpr/fix_ncore_test/experiments/cpr/ /TOP/tmp2/cpr/exp_record_test/
```

### b. Bi-TGCF
Use `Environment-B`
``` 
$ cd baseline/Bi-TGCF
$ bash run_all.sh {processed_data_dir} {exp_record_dir} {mode}
e.g.,
$ bash run_all.sh /TOP/tmp2/cpr/fix_ncore_test/ /TOP/tmp2/cpr/exp_record_test/ traineval
```

### c. LightGCN
Use `Environment-B`
```
$ cd baseline/LGN
$ ./build_cython.sh
$ cd preprocess
$ bash run_preprocess.sh {processed_data_dir}
e.g.,
$ bash run_preprocess.sh /TOP/tmp2/cpr/fix_ncore_test/


$ cd ..
$ bash run_all.sh {processed_data_dir} {exp_record_dir} {mode}
e.g.,
$ bash run_all.sh /TOP/tmp2/cpr/fix_ncore_test/ /TOP/tmp2/cpr/exp_record_test/ traineval
```



### d. BPR
Use `Environment-A`
```
$ cd baseline/BPR_related
$ ./run_lfm-bpr.sh {processed_data_dir} {exp_record_dir} {mode}

e.g.,
$ bash run_lfm-bpr.sh /TOP/tmp2/cpr/fix_ncore_test/ /TOP/tmp2/cpr/exp_record_test/ traineval

```


### e. EMCDR (BPR's graph is required)
Use `Environment-C` for training  
Use `Environment-A` for evaluation
```
$ cd baseline/BPR_related/EMCDR
(must wait until BPR finish training since the following preprocess need BPR's output)
$ bash run_preprocess.sh {processed_data_dir}
$ bash run_train.sh {processed_data_dir}

e.g., 
$ bash run_preprocess.sh /TOP/tmp2/cpr/fix_ncore_test/
$ bash run_train.sh /TOP/tmp2/cpr/fix_ncore_test/

(Change to Environment-A)
$ cd baseline/BPR_related/EMCDR
$ ./run_rec_and_eval.sh {processed_data_dir} {exp_record_dir}

e.g.,
$ ./run_rec_and_eval.sh /TOP/tmp2/cpr/fix_ncore_test/ /TOP/tmp2/cpr/exp_record_test/

```

## 3. Generate Latex Score Table

```
$ python gen_latex_table.py {exp_record_dir}

e.g.,
$ python gen_latex_table.py --exp_record_dir /TOP/tmp2/cpr/exp_record_test/
```

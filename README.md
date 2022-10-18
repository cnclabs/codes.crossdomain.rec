# CPR-2023

## 0. Environment
a. `CPR`, `BPR`, evaluation
```
  docker image: nvcr.io/nvidia/pytorch:22.05-py3  
  pip install faiss-gpu==1.7.2
```
b. `LightGCN`, `Bi-TGCF`
```
  docker image: nvcr.io/nvidia/tensorflow:22.08-tf2-py3
```
c. `EMCDR`
```
  docker image: tensorflow/tensorflow:1.14.0-gpu-py3
```

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
```
$ cd CPR 
$ ./run_smore_ncore.sh {processed_data_dir} {model_save_dir} {exp_record_dir}
$ ./run_eval.sh {processed_data_dir} {model_save_dir} {exp_record_dir}

e.g.,
$ bash run_smore_ncore.sh /TOP/tmp2/cpr/fix_ncore_test/ /TOP/tmp2/cpr/fix_ncore_test/experiments/cpr/ /TOP/tmp2/cpr/exp_record_test/
$ bash run_eval.sh /TOP/tmp2/cpr/fix_ncore_test/ /TOP/tmp2/cpr/fix_ncore_test/experiments/cpr/ /TOP/tmp2/cpr/exp_record_test/
```

### b. Bi-TGCF
```
$ pip install faiss-gpu 
$ cd baseline/Bi-TGCF
$ bash run_all.sh {processed_data_dir} {exp_record_dir}

e.g.,
$ bash run_all.sh /TOP/tmp2/cpr/fix_ncore_test/ /TOP/tmp2/cpr/exp_record_test/ traineval
```

### c. LightGCN
```
$ pip install faiss-gpu
$ cd baseline/LGN
$ ./build_cython.sh
$ cd preprocess
$ bash run_preprocess.sh /TOP/tmp2/cpr/fix_ncore_test/
$ cd ..

$ bash run_all.sh /TOP/tmp2/cpr/fix_ncore_test/ /TOP/tmp2/cpr/exp_record_test/ traineval
```



### d. BPR
```
$ cd baseline/BPR_related
$ pip install lightfm==1.16
$ ./run_lfm-bpr.sh {data_dir} {exp_record_dir} {mode}

e.g.,
$ bash run_lfm-bpr.sh /TOP/tmp2/cpr/fix_ncore_test/ /TOP/tmp2/cpr/exp_record_test/ traineval

```


### e. EMCDR (BPR's graph is required)
```
$ cd baseline/BPR_related/EMCDR
$ pip install pandas
$ bash run_preprocess.sh {data_dir}
$ bash run_train.sh {data_dir}

e.g., 
$ bash run_preprocess.sh /TOP/tmp2/cpr/fix_ncore_test/
$ bash run_train.sh /TOP/tmp2/cpr/fix_ncore_test/

Change docker image to: nvcr.io/nvidia/pytorch:22.05-py3
$ cd baseline/BPR_related/EMCDR
$ ./run_rec_and_eval.sh /TOP/tmp2/cpr/fix_ncore_test/ /TOP/tmp2/cpr/exp_record_test/

```

## 3. Generate Latex Score Table

```
$ python gen_latex_table.py /TOP/tmp2/cpr/exp_record_test/
```

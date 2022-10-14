# CPR-2023

## 0. Environment
a. `CPR`, `BPR`, `CMF`, evaluation
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
  docker image: tensorflow/twnsorflow:1.14.0-gpu-py3
```

## 1. Dataset & Preprocessing
We use 3 pairs of datasets **(Source_Target)**:
* MT_B (Movies_and_TV_5.json + Books_5.json)
* SPO_CSJ (Sports_and_Outdoors_5.json + Clothing_Shoes_and_Jewelry_5.json)
* HK_CSJJ (Home_and_Kitchen_5.json + Clothing_Shoes_and_Jewelry_5.json)
```
$ cd preprocess
$ bash run_gen_input.sh {raw_data_dir} {processed_data_dir}

e.g., 
$ bash run_gen_input.sh /TOP/tmp2/cpr/from_yzliu/ /TOP/tmp2/cpr/fix_ncore_test
```

## 2. Model Training & Evaluation
### a. CPR
```
$ cd CPR 
$ ./run_smore_ncore.sh {processed_data_dir} {model_save_dir} {exp_record_dir}

e.g.,
$ bash run_smore_ncore.sh /TOP/tmp2/cpr/fix_ncore_test/ /TOP/tmp2/cpr/fix_ncore_test/experiments/cpr/ /TOP/tmp2/cpr/exp_record_test/
```

### b. Bi-TGCF
```
$ cd baseline/Bi-TGCF
$ bash run_all.sh {processed_data_dir} {exp_record_dir}

e.g.,
$ bash run_all.sh /TOP/tmp2/cpr/fix_ncore_test/ /TOP/tmp2/cpr/exp_record_test/
```

### c. LightGCN
```
$ cd baseline/LGN
$ ./build_cython.sh
$ bash preprocess/run_preprocess.sh

# Option-1: run sequentially
$ ./run_LGN.sh

# Option-2: run parallelly 
$ bash run_all.sh
```



### d. BPR
```
$ cd baseline/BPR_related
$ pip install lightfm==1.16
$ ./run_lfm-bpr.sh {data_dir} {exp_record_dir}

e.g.,
$ bash run_lfm-bpr.sh /TOP/tmp2/cpr/fix_ncore_test/ /TOP/tmp2/cpr/exp_record_test/

```

### e. CMF (BPR's graph is required)
```
$ cd baseline/BPR_related/CMF
$ pip install cmfrec==3.4.3
$ ./run_CMF.sh

Result path: baseline/BPR_related/CMF/result
```

### f. EMCDR (BPR's graph is required)
```
$ cd baseline/BPR_related/EMCDR
$ pip install pandas
$ ./run_preprocess.sh
$ ./run_train.sh

Change docker image to: nvcr.io/nvidia/pytorch:22.05-py3
$ cd baseline/BPR_related/EMCDR
$ ./run_rec_and_eval.sh

Result path /baseline/BPR_related/EMCDR/result
```

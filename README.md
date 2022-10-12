# CPR-2023
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
docker image: nvcr.io/nvidia/pytorch:22.05-py3

$ cd CPR 
$ pip install -r ./requirments.txt
$ ./run_smore_ncore.sh {processed_data_dir}

e.g.,
$ bash run_smore_ncore.sh /TOP/tmp2/cpr/fix_ncore_test/
```

### b. Bi-TGCF
```
docker image: nvcr.io/nvidia/tensorflow:22.08-tf2-py3

$ cd baseline/Bi-TGCF
$ pip install -r requirements.txt
$ ./run_BiTGCF.sh
```

### c. LightGCN
```
docker image: nvcr.io/nvidia/tensorflow:22.08-tf2-py3

$ cd baseline/LGN
$ ./build_cython.sh
$ pip install -r requirements.txt
$ bash preprocess/run_preprocess.sh

# Option-1: run sequentially
$ ./run_LGN.sh

# Option-2: run parallelly 
$ bash run_all.sh
```



### d. BPR
```
Docker image: nvcr.io/nvidia/pytorch:22.05-py3

$ cd baseline/BPR_related
$ pip install faiss-gpu lightfm==1.16
$ ./run_lfm-bpr.sh

Result path: /baseline/BPR_related/lfm_bpr_result
Result embedding: /baseline/BPR_related/lfm_bpr_graphs
```

### e. CMF (BPR's graph is required)
```
Docker image: nvcr.io/nvidia/pytorch:22.05-py3

$ cd baseline/BPR_related/CMF
$ pip install faiss-gpu cmfrec==3.4.3
$ ./run_CMF.sh

Result path: baseline/BPR_related/CMF/result
```

### f. EMCDR (BPR's graph is required)
```
Change docker image to: tensorflow/twnsorflow:1.14.0-gpu-py3
$ cd baseline/BPR_related/EMCDR
$ pip install pandas
$ ./run_preprocess.sh
$ ./run_train.sh

Change docker image to: nvcr.io/nvidia/pytorch:22.05-py3
$ cd baseline/BPR_related/EMCDR
$ pip install faiss-gpu
$ ./run_rec_and_eval.sh

Result path /baseline/BPR_related/EMCDR/result
```

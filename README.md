# CPR-2023
## Dataset 
We use 3 pairs of datasets **(Source_Target)**:
* MT_B (Movies_and_TV_5.json + Books_5.json)
* SPO_CSJ (Sports_and_Outdoors_5.json + Clothing_Shoes_and_Jewelry_5.json)
* HK_CSJJ (Home_and_Kitchen_5.json + Clothing_Shoes_and_Jewelry_5.json)
### Preprocessing 
 
```
$ cd preprocess
$ bash run_gen_input.sh {raw_data_dir} {save_dir}

e.g., 
$ bash run_gen_input.sh /TOP/tmp2/cpr/from_yzliu/ /TOP/tmp2/cpr/fix_ncore_test
```

## Models
### CPR

Environment  
docker image: `nvcr.io/nvidia/pytorch:22.05-py3`
```
$ cd CPR 
(use virtualenv and install ./requirments.txt)
$ ./run_smore_ncore.sh

Find evaluated score in ./result

```

### Bi-TGCF
docker image: `nvcr.io/nvidia/tensorflow:22.08-tf2-py3`
```
$ cd baseline/Bi-TGCF
$ pip install -r requirements.txt
$ ./run_BiTGCF.sh
```
### LGN (LightGCN)
We use [NeuRec](https://github.com/wubinzzu/NeuRec) for LGN. Preporcess inputs for NeuRec format first.

docker image: `nvcr.io/nvidia/tensorflow:22.08-tf2-py3`

```
$ cd baseline/LGN
$ ./build_cython.sh
$ pip install -r requirements.txt

# Option-1: run sequentially
$ ./run_LGN.sh

# Option-2: run parallelly 
$ bash run_all.sh
```


Docker image: nvcr.io/nvidia/pytorch:22.05-py3
### BPR
```
$ cd baseline/BPR_related
$ pip install faiss-gpu lightfm==1.16
$ ./run_lfm-bpr.sh
Result path: /baseline/BPR_related/lfm_bpr_result
Result embedding: /baseline/BPR_related/lfm_bpr_graphs
```
### CMF (BPR's graph is required)
```
$ cd baseline/BPR_related/CMF
$ pip install faiss-gpu cmfrec==3.4.3
$ ./run_CMF.sh
Result path: baseline/BPR_related/CMF/result
```
### EMCDR (BPR's graph is required)
```
Environment: 
	docker image: tensorflow/tensorflow:1.14.0-gpu-py3
		      nvcr.io/nvidia/pytorch:22.05-py3

Change docker image to tensorflow/twnsorflow:1.14.0-gpu-py3
$ cd baseline/BPR_related/EMCDR
$ pip install pandas
$ ./run_preprocess.sh
$ ./run_train.sh
Change docker image to nvcr.io/nvidia/pytorch:22.05-py3
$ cd baseline/BPR_related/EMCDR
$ pip install faiss-gpu
$ ./run_rec_and_eval.sh
Result path /baseline/BPR_related/EMCDR/result


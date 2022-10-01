# CPR-2022
We generally divide this project into 3 parts:
* Pre-preprocess (Raw -> LOO data)
* Preprocess (LOO data -> N-core filtering -> Input)
* Training and Evaluation  

We use 3 pairs of datasets **(Source_Target)**:
* MT_B (Movies_and_TV_5.json + Books_5.json)
* SPO_CSJ (Sports_and_Outdoors_5.json + Clothing_Shoes_and_Jewelry_5.json)
* HK_CSJJ (Home_and_Kitchen_5.json + Clothing_Shoes_and_Jewelry_5.json)
## Pre-preprocess (Raw -> LOO data)
### Using processed data (Recommended)
Since converting raw data to raw LOO data is quite time consuming, we **recommend** you dowload processed LOO data if you don't plan to add new datasets:  
```
$ scp -r ${ACCOUNT}@clip4.cs.nccu.edu.tw:/tmp2/yzliu/store/CPR/LOO_data_0core .
```  

### Process from scratch
If you really need to download raw data and do leave one out (LOO):
```
$ scp -r ACCOUNT@clip4.cs.nccu.edu.tw:/tmp2/yzliu/store/CPR/raw_data .
$ python3 preprocess/raw_to_LOO.py --raw_data ${RAW_DATA_FILE_NAME} --dataset_name ${LOO_DATA_OUTPUT_FILE_NAME}
```
  
--
  
Note that:
* you could replace `clip4.cs.nccu.edu.tw` with `cfda4.citi.sinica.edu.tw`. 
* Raw data is from Amazon: http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/

## Preprocess (LOO_data -> N-core filtering -> Input)
Before running preprocess, requirements for it is under base dir.

```
$ cd preprocess
$ ./run_gen_input.sh 
```

## Training and Evaluation
### Our model
#### CPR
```
$ cd CPR
$ ./run_smore_ncore.sh
```
### Baselines
Since baselines need different envs, you could manage multiple envs for them. Modules needed are list in `baseline/${MODEL_NAME}/requirements.txt`.  
Models under `baseline/BPR_related` shared the same env. 
#### Bi-TGCF
docker image: `nvcr.io/nvidia/tensorflow:22.08-tf2-py3`
```
$ cd baseline/Bi-TGCF
$ pip install -r requirements.txt
$ ./run_BiTGCF.sh
```
#### LGN (LightGCN)
We use [NeuRec](https://github.com/wubinzzu/NeuRec) for LGN. Preporcess inputs for NeuRec format first.

docker image: `nvcr.io/nvidia/tensorflow:22.08-tf2-py3`

```
$ cd baseline/LGN
$ cd preprocess
$ ./run_preprocess.sh
$ cd ..
$ ./build_cython.sh
$ pip install -r requirements.txt

# Option-1: run sequentially
$ ./run_LGN.sh

# Option-2: run parallelly 
$ bash run_all.sh
```
#### BPR related models
Run BPR first and wait for its graphs.
#### BPR
```
$ cd baseline/BPR_related
$ ./run_lfm-bpr.sh
```
#### CMF
```
$ cd baseline/BPR_related/CMF
$ ./run_CMF.sh
```
#### EMCDR
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

```
## Environment
Python >=3.7 is needed  
We use official nvidia docker image:  
[nvcr.io/nvidia/pytorch:22.05-py3](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-05.html#rel_22-05)  
One can simply enter into the container, then `pip install -r requirments.txt` to reproduce our environment.

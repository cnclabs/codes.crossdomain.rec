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
```
$ cd preprocess
$ ./run_gen_input.sh 
```

## Training and Evaluation
### Our model
#### CPR

Environment  
docker image: `nvcr.io/nvidia/pytorch:22.05-py3`
```
$ cd CPR 
(use virtualenv and install ./requirments.txt)
$ ./run_smore_ncore.sh

Find evaluated score in ./result

```
### Baselines
Since baselines need different envs, you could manage multiple envs for them. Modules needed are list in `baseline/${MODEL_NAME}/requirements.txt`.  
Models under `baseline/BPR_related` shared the same env. 
#### Bi-TGCF
```
$ cd baseline/Bi-TGCF
$ ./run_BiTGCF.sh
```
#### LGN (LightGCN)
```
$ cd baseline/LGN
$ ./build_cython.sh
$ ./run_LGN.sh
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
$ cd baseline/BPR_related/EMCDR
$ ./run_EMCDR.sh
```
## Environment
Python >=3.7 is needed  
We use official nvidia docker image:  
[nvcr.io/nvidia/pytorch:22.05-py3](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-05.html#rel_22-05)  
One can simply enter into the container, then `pip install -r requirments.txt` to reproduce our environment.

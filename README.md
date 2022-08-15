# CPR-2022

## Preparation
Since converting raw data to raw LOO data is quite time consuming, you could dowload raw LOO data if you don't plan to add new datasets:  
  
**CMD** `scp -r ACCOUNT@clip4.cs.nccu.edu.tw:/tmp2/yzliu/store/LOO_data_0core .`

## Run
Set `raw_src`, `raw_tar`, `src`, `tar`, `core`, `cold_sample`, `eval_sample` in `demo_reproduce.sh` then run to make input with ncore filtering.  
  
**Note that**  
* We defaultly skip the step which converts raw data to raw LOO data. If you need to do it, please uncomment those lines.
* `raw_src` and `raw_tar` mean the file names of raw data.
* `src` and `tar` should be the datasets existing in `LOO_data_0`.
  
**CMD** `./demo_reproduce.sh`

## Steps in `demo_reproduce.sh`
1. (Optional) From amazon raw data to raw LOO data (aka. LOO_data_0)
2. Filtered LOO_data_0 to LOO_data_ncore & user_ncore
3. Generated input_ncore from LOO_data_ncore & user_ncore
4. Train and eval

 ## Environment
 We use official nvidia docker image:  
 [nvcr.io/nvidia/pytorch:22.05-py3](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-05.html#rel_22-05)  
 One can simply enter into the container, then `pip install -r requirments.txt` to reproduce our environment.

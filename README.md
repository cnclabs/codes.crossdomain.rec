# CPR: Item Concept Embedding via Textual Information
Our codes for https://dl.acm.org/doi/abs/10.1007/978-3-031-28238-6_35

## 0. Environment
- Docker image: `nvcr.io/nvidia/pytorch:22.05-py3`
```
pip install -r requirements.txt
```

## 1. Data

- Enter preprocess directory 
```
cd preprocess/
```
- Download `raw` data
```
bash download_amazon_data.sh {raw_data_dir}
```

- Process `raw` to `loo` (It takes long time, particularly `Books_5`)
```
bash raw_to_loo.sh {raw_data_dir} {loo_data_dir}
```

- Process `loo` to `loo-5core`
```
bash loo_to_ncore.sh {loo_data_dir} {ncore_data_dir}
```

- Generate `input` for CPR 
```
bash generate_cpr_input.sh {ncore_data_dir} {cpr_input_dir}
```

## 2. Model Training & Evaluation 
- Enter CPR directory
```
$ cd models/CPR
```

- Compile cpp code
```
make
```

- Usage example
```
bash n_round_pair_cpr.sh {ncore_data_dir} {cpr_input_dir} {emb_save_dir} {score_save_dir} traineval 5
```

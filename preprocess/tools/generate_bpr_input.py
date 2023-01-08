import os
import pandas as pd
import argparse

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpr_input_dir', type=str)
    parser.add_argument('--bpr_input_dir', type=str)
    parser.add_argument('--src', type=str, help='souce name')
    parser.add_argument('--tar', type=str, help='target name')
    args=parser.parse_args()
    print(args)
    
    cpr_input_dir = args.cpr_input_dir
    bpr_input_dir = args.bpr_input_dir
    src, tar = args.src, args.tar

    if not os.path.exists(bpr_input_dir):
        os.makedirs(bpr_input_dir)
    
    print("Processing src_tar_train...")
    path = f'{cpr_input_dir}/{src}_src_train_input.txt'
    src_train = pd.read_csv(path, header=None, sep='\t')
    
    path = f'{cpr_input_dir}/{tar}_tar_train_input.txt'
    tar_train = pd.read_csv(path, header=None, sep='\t')
    
    src_tar_train = pd.concat([src_train, tar_train])
    save_path = f'{bpr_input_dir}/{src}_{tar}_src_tar_train_input.txt'
    src_tar_train.to_csv(save_path, header=False, sep='\t', index=False)
    print("Done src_tar_train!")
    
    print("Processing src_ctar_train...")
    path = f'{cpr_input_dir}/{tar}_ctar_train_input.txt'
    ctar_train = pd.read_csv(path, header=None, sep='\t')
    
    src_ctar_train = pd.concat([src_train, ctar_train])
    save_path = f'{bpr_input_dir}/{src}_{tar}_src_ctar_train_input.txt'
    src_ctar_train.to_csv(save_path, header=False, sep='\t', index=False)
    print("Done src_ctar_train!")
    

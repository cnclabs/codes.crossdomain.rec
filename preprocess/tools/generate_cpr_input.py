import pickle
import pandas as pd 
from math import log
import os
import argparse

def compute_cpr_link_weight(df, user_attr, item_attr):
    # this is for CPR' weight, for sampling and weighted sum
    out = df.groupby([user_attr, item_attr]).size().apply(lambda x: log(x+1.))
    
    return out

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Generated correspond inputs from LOO datas with n core.')
    parser.add_argument('--ncore_data_dir', type=str, help='where to get common data')
    parser.add_argument('--cpr_input_dir', type=str, help='where to save cpr input')
    parser.add_argument('--src', type=str, help='souce name')
    parser.add_argument('--tar', type=str, help='target name')
    parser.add_argument('--item_attr', type=str, help='(default for amz) attribute represents items\' ids', default='asin')
    parser.add_argument('--user_attr', type=str, help='(default for amz) attribute represents users\' ids', default='reviewerID')
    args=parser.parse_args()
    print(args)
    
    src, tar = args.src, args.tar
    item_attr, user_attr = args.item_attr, args.user_attr
    input_save_dir = args.cpr_input_dir
    
    if not os.path.isdir(input_save_dir):
        os.makedirs(input_save_dir)

    # src
    print("Processing src train...")
    with open(f'{args.ncore_data_dir}/{src}_src_train.pickle', 'rb') as pf:
        src_train = pickle.load(pf)
    src_train_graph = compute_cpr_link_weight(src_train, user_attr, item_attr) 
    src_train_graph.to_csv(os.path.join(input_save_dir, f'{src}_src_train_input.txt'), header=False, sep='\t')
    print("Done src train!")
    
    # tar
    print("Processing tar train...")
    with open(f'{args.ncore_data_dir}/{tar}_tar_train.pickle', 'rb') as pf:
        tar_train = pickle.load(pf)
    tar_train_graph = compute_cpr_link_weight(tar_train, user_attr, item_attr) 
    tar_train_graph.to_csv(os.path.join(input_save_dir,f'{tar}_tar_train_input.txt'), header=False, sep='\t')
    print("Done tar train!")
    
    # ctar
    print("Processing ctar train...")
    with open(f'{args.ncore_data_dir}/{tar}_ctar_train.pickle', 'rb') as pf:
        ctar_train = pickle.load(pf)
    ctar_train_graph = compute_cpr_link_weight(ctar_train, user_attr, item_attr) 
    ctar_train_graph.to_csv(os.path.join(input_save_dir,f'{tar}_ctar_train_input.txt'), header=False, sep='\t')
    print("Done ctar train!")

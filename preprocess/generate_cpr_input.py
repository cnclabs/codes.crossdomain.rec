import pickle
import pandas as pd 
from math import log
import os
import argparse
import random

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Generated correspond inputs from LOO datas with n core.')
    parser.add_argument('--ncore_data_dir', type=str, help='where to get materials and where to save inputs')
    parser.add_argument('--ncore', type=int, help='core number', default=5)
    parser.add_argument('--src', type=str, help='souce name')
    parser.add_argument('--tar', type=str, help='target name')
    parser.add_argument('--item_attr', type=str, help='(default for amz) attribute represents items\' ids', default='asin')
    parser.add_argument('--user_attr', type=str, help='(default for amz) attribute represents users\' ids', default='reviewerID')
    args=parser.parse_args()
    print(args)
    
    ncore = args.ncore
    src, tar = args.src, args.tar
    item_attr, user_attr = args.item_attr, args.user_attr
    
    input_save_dir = "{}/input_{}core".format(args.ncore_data_dir, ncore)
    
    if not os.path.isdir(input_save_dir):
            os.makedirs(input_save_dir)
    
    # tar
    with open(f'{args.ncore_data_dir}/loo_data_{ncore}core/{tar}_tar_train.pickle', 'rb') as pf:
        tar_train = pickle.load(pf)
    
    # this is for CPR code current limitation (or design), no difference between user, item. It's just a bag.
    tar_train[user_attr] = tar_train[user_attr].apply(lambda x: 'user_'+x)
    # this is for CPR' weight, for sampling and weighted sum
    tar_train_graph = tar_train.groupby([user_attr, item_attr]).size().apply(lambda x: log(x+1.))
    #_df = pd.concat([tar_train_graph])
    tar_train_graph.to_csv(os.path.join(input_save_dir,f'{tar}_tar_train_input.txt'), header=False, sep='\t')
    
    # src
    with open(f'{args.ncore_data_dir}/loo_data_{ncore}core/{src}_src_train.pickle', 'rb') as pf:
        src_train = pickle.load(pf)
    
    # this is for CPR code current limitation (or design), no difference between user, item. It's just a bag.
    src_train[user_attr] = src_train[user_attr].apply(lambda x: 'user_'+x)
    # this is for CPR' weight, for sampling and weighted sum
    src_train_graph = src_train.groupby([user_attr, item_attr]).size().apply(lambda x: log(x+1.))
    src_train_graph.to_csv(os.path.join(input_save_dir, f'{src}_src_train_input.txt'), header=False, sep='\t')
    
    # process global testing target/shared/cold users

    # target
    with open(f'{args.ncore_data_dir}/loo_data_{ncore}core/{src}_{tar}_src_tar_sample_testing_target_users.pickle', 'rb') as pf:
        target_users= pickle.load(pf)
    # this is for CPR code current limitation (or design), no difference between user, item. It's just a bag.
    target_users = set(map(lambda x: "user_"+x, target_users))
    
    with open(os.path.join(input_save_dir, f'{src}_{tar}_src_tar_sample_testing_target_users.pickle'), 'wb') as pf:
        pickle.dump(target_users, pf)
    
    # shared
    with open(f'{args.ncore_data_dir}/loo_data_{ncore}core/{src}_{tar}_src_tar_sample_testing_shared_users.pickle', 'rb') as pf:
        shared_users = pickle.load(pf)
    # this is for CPR code current limitation (or design), no difference between user, item. It's just a bag.
    shared_users = set(map(lambda x: "user_"+x, shared_users))
    
    with open(os.path.join(input_save_dir, f'{src}_{tar}_src_tar_sample_testing_shared_users.pickle'), 'wb') as pf:
        pickle.dump(shared_users, pf)
    
    # cold
    with open(f'{args.ncore_data_dir}/loo_data_{ncore}core/{src}_{tar}_src_tar_sample_testing_cold_users.pickle', 'rb') as pf:
        cold_users = pickle.load(pf)

    # this is for CPR code current limitation (or design), no difference between user, item. It's just a bag.
    cold_users = set(map(lambda x: "user_"+x, cold_users))
    
    with open(os.path.join(input_save_dir, f'{src}_{tar}_src_tar_sample_testing_cold_users.pickle'), 'wb') as pf:
        pickle.dump(cold_users, pf)
    
    with open(f'{args.ncore_data_dir}/loo_data_{ncore}core/{tar}_tar_train.pickle', 'rb') as pf:
        tar_train = pickle.load(pf)
    
    # cold tar
    # this is for CPR code current limitation (or design), no difference between user, item. It's just a bag.
    tar_train[user_attr] = tar_train[user_attr].apply(lambda x: 'user_'+x)
    # produce artificial cold user in target dataset
    cold_tar_train = tar_train[~tar_train[user_attr].isin(cold_users)]
    # this is for CPR' weight, for sampling and weighted sum
    cold_tar_train_graph = cold_tar_train.groupby([user_attr, item_attr]).size().apply(lambda x: log(x+1.))
    
    cold_tar_train_graph.to_csv(os.path.join(input_save_dir, f'{tar}_ctar_train_input.txt'), header=False, sep='\t')

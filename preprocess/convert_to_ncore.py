import pickle
import pandas as pd
from math import log
import os
import random
import argparse

def ncore_filter(train, user_attr, item_attr, ncore):
    ncore_train = train[(train[user_attr].isin(train[user_attr].value_counts()[train[user_attr].value_counts()>=ncore].index)) & \
            (train[item_attr].isin(train[item_attr].value_counts()[train[item_attr].value_counts()>=ncore].index))]
    return ncore_train

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Filtered LOO datas with n core.')
    parser.add_argument('--loo_data_dir', type=str, help='',default=None)
    parser.add_argument('--ncore_data_dir', type=str, help='',default=None)
    parser.add_argument('--ncore', type=int, help='core number', default=5)
    parser.add_argument('--src', type=str, help='souce name', default='hk')
    parser.add_argument('--tar', type=str, help='target name', default='csjj')
    parser.add_argument('--item_attr', type=str, help='(default for amz) attribute represents items\' ids', default='asin')
    parser.add_argument('--user_attr', type=str, help='(default for amz) attribute represents users\' ids', default='reviewerID')
    parser.add_argument('--n_testing_user', type=int)
    args=parser.parse_args()
    print(args)
    random.seed(2022)
    
    ncore = args.ncore
    src, tar = args.src, args.tar
    item_attr, user_attr = args.item_attr, args.user_attr
    
    ncore_save_dir = "{}/loo_data_{}core".format(args.ncore_data_dir, ncore)
    
    if not os.path.isdir(ncore_save_dir):
        os.makedirs(ncore_save_dir)
    
    print(f"== SOURCE-TARGET: {src.upper()}-{tar.upper()} ==")
    print(f"Start {ncore}core filtering ...")

    # src train
    with open(f'{args.loo_data_dir}/{src}_train.pickle', 'rb') as f:
        src_train = pickle.load(f)
    
    src_ncore_train = ncore_filter(src_train, user_attr, item_attr, ncore)
    while (src_ncore_train[user_attr].value_counts().min()<ncore) | (src_ncore_train[item_attr].value_counts().min()<ncore):
        src_ncore_train = ncore_filter(src_ncore_train, user_attr, item_attr, ncore)
    
    with open(os.path.join(ncore_save_dir, f'{src}_src_train.pickle'), 'wb') as f:
        pickle.dump(src_ncore_train, f)

    # src test
    with open(f'{args.loo_data_dir}/{src}_test.pickle', 'rb') as f:
        src_test = pickle.load(f)

    src_ncore_test = src_test[(src_test[user_attr].isin(src_ncore_train[user_attr].unique())) & (src_test[item_attr].isin(src_ncore_train[item_attr].unique()))]
    with open(os.path.join(ncore_save_dir, f'{src}_src_test.pickle'), 'wb') as f:
        pickle.dump(src_ncore_test, f)

    print("-"*10)
    print(f"SRC ({src.upper()}) Train {ncore}core")
    print("Before:", len(src_train))
    print("After :", len(src_ncore_train))
    print("Diff  :", len(src_train)-len(src_ncore_train))
    print("Unique Users:", len(src_ncore_train[user_attr].unique()))
    print("-"*10)
    print(f"SRC ({src.upper()}) Test {ncore}core")
    print("Before:", len(src_test))
    print("After :", len(src_ncore_test))
    print("Diff  :", len(src_test)-len(src_ncore_test))
    print("Unique Users:", len(src_ncore_test[user_attr].unique()))
    print("-"*10)

    # tar train
    with open(f'{args.loo_data_dir}/{tar}_train.pickle', 'rb') as f:
        tar_train = pickle.load(f)
    
    tar_ncore_train = ncore_filter(tar_train, user_attr, item_attr, ncore)
    while (tar_ncore_train[user_attr].value_counts().min()<ncore) | (tar_ncore_train[item_attr].value_counts().min()<ncore):
        tar_ncore_train = ncore_filter(tar_ncore_train, user_attr, item_attr, ncore)
    
    with open(os.path.join(ncore_save_dir, f'{tar}_tar_train.pickle'), 'wb') as f:
        pickle.dump(tar_ncore_train, f)

    # tar test
    with open(f'{args.loo_data_dir}/{tar}_test.pickle', 'rb') as f:
        tar_test = pickle.load(f)
    tar_ncore_test = tar_test[(tar_test[user_attr].isin(tar_ncore_train[user_attr].unique())) & (tar_test[item_attr].isin(tar_ncore_train[item_attr].unique()))]
    with open(os.path.join(ncore_save_dir, f'{tar}_tar_test.pickle'), 'wb') as f:
        pickle.dump(tar_ncore_test, f)
    
    print("-"*10)
    print(f"TAR ({tar.upper()}) Train {ncore}core")
    print("Before:", len(tar_train))
    print("After :", len(tar_ncore_train))
    print("Diff  :", len(tar_train)-len(tar_ncore_train))
    print("Unique Users:", len(tar_ncore_train[user_attr].unique()))
    print("-"*10)
    print(f"TAR ({tar.upper()}) Test {ncore}core")
    print("Before:", len(tar_test))
    print("After :", len(tar_ncore_test))
    print("Diff  :", len(tar_test)-len(tar_ncore_test))
    print("Unique Users:", len(tar_ncore_test[user_attr].unique()))
    print("-"*10)
    
    # make testing users
    print("Sample testing users ...")
    # target
    all_testing_target_users = set(tar_ncore_test[user_attr])
    print(f'all testing target users: {len(all_testing_target_users)}')
    sample_testing_target_users = random.sample(all_testing_target_users, args.n_testing_user)
    print(f'sample testing target users: {len(sample_testing_target_users)}')
   
    # shared
    all_shared_users = set(tar_ncore_train[user_attr]).intersection(set(src_ncore_train[user_attr]))
    print(f'all train(?) shared users: {len(all_shared_users)}')
    try:
        assert len(all_shared_users)>=args.n_testing_user
    except:
        print(len(all_shared_users))
    sample_testing_shared_users = random.sample(all_shared_users, args.n_testing_user)
    print(f'sample testing shared users: {len(sample_testing_shared_users)}')
    sample_testing_cold_users   = random.sample(all_shared_users, args.n_testing_user)
    print(f'sample testing cold users: {len(sample_testing_cold_users)}')
    
    # save users
    with open(os.path.join(ncore_save_dir, f'{src}_{tar}_src_tar_sample_testing_target_users.pickle'), 'wb') as pf:
        pickle.dump(sample_testing_target_users, pf)

    with open(os.path.join(ncore_save_dir, f'{src}_{tar}_src_tar_sample_testing_shared_users.pickle'), 'wb') as pf:
        pickle.dump(sample_testing_shared_users, pf)
    
    with open(os.path.join(ncore_save_dir, f'{src}_{tar}_src_tar_sample_testing_cold_users.pickle'), 'wb') as pf:
        pickle.dump(sample_testing_cold_users, pf)

    # cold tar
    # produce artificial cold user in target dataset
    cold_tar_train = tar_ncore_train[~tar_ncore_train[user_attr].isin(sample_testing_cold_users)]

    with open(os.path.join(ncore_save_dir, f'{tar}_ctar_train.pickle'), 'wb') as f:
        pickle.dump(cold_tar_train, f)

import pickle
import argparse
import os
import pandas as pd
from multiprocessing import Pool
import numpy as np
from functools import partial
import random

def get_testing_users_rec_dict(n_worker, tar_test_df, uid_u, uid_i, test_mode, data_input_dir, src, tar):
    tar_train_path = f'{data_input_dir}/{tar}_tar_train_input.txt'
    tar_train_df = pd.read_csv(tar_train_path, sep='\t', header=None, names=[uid_u, uid_i, 'xxx'])
    total_item_set = set(tar_train_df[uid_i])
    
    testing_users = get_testing_users(test_mode, data_input_dir, src, tar)

    mp = Pool(n_worker)

    print(f"Start generating testing users' postive-negative pairs... using {n_worker} workers.")
    split_datas = np.array_split(list(testing_users), n_worker)
    func = partial(process_user_pos_neg_pair, tar_train_df, tar_test_df, uid_u, uid_i, total_item_set)
    results = mp.map(func, split_datas)
    mp.close()
    
    testing_users_rec_dict = {}
    for r in results:
        testing_users_rec_dict.update(r)
    print("Done generating testing users' positive-negative pairs.")

    return testing_users_rec_dict

def get_testing_users(test_mode, data_input_dir, src, tar):
    path = f'{data_input_dir}/{src}_{tar}_src_tar_sample_testing_{test_mode}_users.pickle'
    with open(path, 'rb') as pf:
        testing_users = pickle.load(pf)

    return testing_users

def process_user_pos_neg_pair(tar_train_df, tar_test_df, uid_u, uid_i, total_item_set,  user_list):
  user_rec_dict = {}
  for user in user_list:
      pos_pool = set(tar_train_df[tar_train_df[uid_u] == user][uid_i])
      neg_pool = total_item_set - pos_pool
      neg_99 = random.sample(neg_pool, 99)
      user_rec_pool = list(neg_99) + list(tar_test_df[tar_test_df[uid_u] == user][uid_i])
      user_rec_dict[user] = user_rec_pool

  return user_rec_dict

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='groundtruth files dir')
    parser.add_argument('--test_mode', type=str, help='{target, shared, cold}')
    parser.add_argument('--ncore', type=int, help='core number', default=5)
    parser.add_argument('--n_worker', type=int, help='number of workers', default=None)
    parser.add_argument('--src', type=str, help='souce name')
    parser.add_argument('--tar', type=str, help='target name')
    parser.add_argument('--uid_i', type=str, help='(default for amz) unique id column for item', default='asin')
    parser.add_argument('--uid_u', type=str, help='(default for amz) unique id column of user', default='reviewerID')
    args=parser.parse_args()
    
    tar_test_path  = '{}/LOO_data_{ncore}core/{tar}_test.pickle'.format(args.data_dir, ncore=args.ncore, tar=args.tar) 
    with open(tar_test_path, 'rb') as pf:
        tar_test_df = pickle.load(pf)
    tar_test_df[args.uid_u]  = tar_test_df[args.uid_u].apply(lambda x: 'user_'+x)
    
    data_input_dir = os.path.join(args.data_dir, f'input_{args.ncore}core')
    testing_users_rec_dict = get_testing_users_rec_dict(args.n_worker, tar_test_df, args.uid_u, args.uid_i, args.test_mode, data_input_dir, args.src, args.tar)
    
    save_path = os.path.join(data_input_dir, f'testing_users_rec_dict_{args.src}_{args.tar}_{args.test_mode}.pickle')
    with open(save_path, 'wb') as f:
        pickle.dump(testing_users_rec_dict, f)

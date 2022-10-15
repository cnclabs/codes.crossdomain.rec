import os
import argparse
import faiss
import numpy as np
import json
import pandas as pd
import pickle
import random
import re
import sys
#sys.path.insert(1, '../../../CPR/')
import multiprocessing 
from multiprocessing import Pool
import time
import uuid

from evaluation.utility import save_exp_record, rank_and_score, generate_item_graph_df, generate_user_emb, get_testing_users

parser=argparse.ArgumentParser(description='Calculate the similarity and recommend target items')
parser.add_argument('--data_dir', type=str, help='groundtruth files dir')
parser.add_argument('--save_dir', type=str, help='dir to save cav')
parser.add_argument('--save_name', type=str, help='name to save csv')
parser.add_argument('--src', type=str, help='souce name', default='hk')
parser.add_argument('--tar', type=str, help='target name', default='csjj')
parser.add_argument('--current_epoch', type=str)
parser.add_argument('--output_file', type=str, help='output_file name')
parser.add_argument('--test_mode', type=str, help='{target, shared}')
parser.add_argument('--model_name', type=str)
parser.add_argument('--n_worker', type=int, help='number of multi-processing workers')
parser.add_argument('--seed', type=int)
parser.add_argument('--dataset_name', type=str, help='{tv_vod, csj_hk, mt_books, el_cpa, spo_csj}')
parser.add_argument('--ncore', type=int, help='core_filter', default=0)
parser.add_argument('--uid_i', type=str, help='(default for amz) unique id column for item', default='asin')
parser.add_argument('--uid_u', type=str, help='(default for amz) unique id column of user', default='reviewerID')
parser.add_argument('--top_ks', nargs='*', help='top_k to eval', default=[1, 3, 5, 10, 20], action='extend', type=int)

args=parser.parse_args()

src = args.src
tar = args.tar
ncore = args.ncore
save_dir = args.save_dir
output_file = args.output_file
test_mode = args.test_mode
save_name=args.save_name

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

random.seed(args.seed)
data_input_dir = os.path.join(args.data_dir, f'input_{ncore}core')
testing_users = get_testing_users(test_mode, data_input_dir, src, tar)

with open('{}/LOO_data_{ncore}core/{tar}_train.pickle'.format(args.data_dir, ncore=ncore, tar=tar), 'rb') as pf:
    tar_train_df = pickle.load(pf)
with open('{}/LOO_data_{ncore}core/{tar}_test.pickle'.format(args.data_dir, ncore=ncore, tar=tar), 'rb') as pf:
    tar_test_df = pickle.load(pf)

tar_train_df[args.uid_u] = tar_train_df[args.uid_u].apply(lambda x: 'user_'+x)
tar_test_df[args.uid_u]  = tar_test_df[args.uid_u].apply(lambda x: 'user_'+x)
total_item_set = set(tar_train_df[args.uid_i])

# Generate user 100 rec pool
print("Start generating user rec dict...")

def process_user_pos_neg_pair(user_list):
  user_rec_dict = {}

  for user in user_list:
      pos_pool = set(tar_train_df[tar_train_df[args.uid_u] == user][args.uid_i])
      neg_pool = total_item_set - pos_pool
      neg_99 = random.sample(neg_pool, 99)
      user_rec_pool = list(neg_99) + list(tar_test_df[tar_test_df[args.uid_u] == user][args.uid_i])
      user_rec_dict[user] = user_rec_pool

  return user_rec_dict

if args.n_worker is None:
    cpu_amount = multiprocessing.cpu_count()
    n_worker = cpu_amount - 2
else:
    n_worker = args.n_worker

print(f"Start generating testing users' postive-negative pairs... using {n_worker} workers.")
mp = Pool(n_worker)
split_datas = np.array_split(list(testing_users), n_worker)
results = mp.map(process_user_pos_neg_pair, split_datas)
mp.close()

testing_users_rec_dict = {}
for r in results:
  testing_users_rec_dict.update(r)

print("testing users rec dict generated!")


# Get emb of testing users
with open(f'/TOP/home/ythuang/CODE/tmp/refactor_eval/codes.crossdomain.rec/baseline/BPR_related/EMCDR/{src}_{tar}/shared_users_mapped_emb_dict_{args.current_epoch}.pickle', 'rb') as pf:
        shared_users_mapped_emb_dict = pickle.load(pf)

if args.test_mode == 'target':
    ## source 1 : testing users are from shared users
    with open(f'/TOP/home/ythuang/CODE/tmp/refactor_eval/codes.crossdomain.rec/baseline/BPR_related/EMCDR/{src}_{tar}/shared_users_mapped_emb_dict_{args.current_epoch}.pickle', 'rb') as pf:
        shared_users_mapped_emb_dict = pickle.load(pf)
    ## source 2 : testing users are from target domain only users
    target_users_emb_dict = {}
    with open(f'/TOP/home/ythuang/CODE/tmp/refactor_eval/codes.crossdomain.rec/baseline/BPR_related/lfm_bpr_graphs/{tar}_lightfm_bpr_{args.current_epoch}_10e-5.txt', 'r') as f:
        for line in f:
            line = line.split('\t')
            prefix = line[0]
            prefix = prefix.replace(" ", "")
            emb=line[1].split()  
            if 'user_' in prefix:
                target_users_emb_dict[prefix] = np.array(emb, dtype=np.float32)

    shared_users_amount = 0
    target_only_users_amount = 0
    user_emb = {}
    for user in testing_users:
        if user in shared_users_mapped_emb_dict.keys():
            user_emb[user] = shared_users_mapped_emb_dict[user]
            shared_users_amount += 1
        else:
            user_emb[user] = target_users_emb_dict[user]
            target_only_users_amount += 1

if args.test_mode == 'shared':
    shared_users_amount = len(testing_users)
    user_emb = {k:v for k,v in shared_users_mapped_emb_dict.items() if k in testing_users}

print("Start getting embedding for each user and item...")
_path = f'/TOP/home/ythuang/CODE/tmp/refactor_eval/codes.crossdomain.rec/baseline/BPR_related/lfm_bpr_graphs/{tar}_lightfm_bpr_{args.current_epoch}_10e-5.txt'
item_graph_df= generate_item_graph_df(_path)
print("Got embedding!")
#
top_ks = args.top_ks
total_rec, total_ndcg, count = rank_and_score(testing_users, top_ks, user_emb, testing_users_rec_dict, item_graph_df, tar_test_df, n_worker, args.uid_u, args.uid_i)

model_name = args.model_name
dataset_pair = f"{src}_{tar}"
test_mode=args.test_mode
save_exp_record(model_name, dataset_pair, test_mode, top_ks, total_rec, total_ndcg, count, save_dir, save_name, output_file)
    

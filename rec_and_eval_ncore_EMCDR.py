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

from evaluation.utility import save_exp_record, rank_and_score, generate_item_graph_df, generate_user_emb

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
source_name = args.src
target_name = args.tar
src = args.src
tar = args.tar
#target_name = dataset_name.split('_')[1]
ncore = args.ncore
save_dir=args.save_dir
output_file = args.output_file

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_name=args.save_name

# ground truth

random.seed(args.seed)
if args.test_mode == 'target':
    with open('{}/input_{ncore}core/{src}_{tar}_test_target_users.pickle'.format(args.data_dir, ncore=ncore, src=src, tar=tar), 'rb') as pf:
        testing_users = pickle.load(pf)
if args.test_mode == 'shared':
    with open('{}/input_{ncore}core/{src}_{tar}_test_shared_users.pickle'.format(args.data_dir, ncore=ncore, src=src, tar=tar), 'rb') as pf:
        testing_users = pickle.load(pf)

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
with open(f'/TOP/home/ythuang/CODE/tmp/refactor_eval/codes.crossdomain.rec/baseline/BPR_related/EMCDR/{source_name}_{target_name}/shared_users_mapped_emb_dict_{args.current_epoch}.pickle', 'rb') as pf:
        shared_users_mapped_emb_dict = pickle.load(pf)

if args.test_mode == 'target':
    ## source 1 : testing users are from shared users
    with open(f'/TOP/home/ythuang/CODE/tmp/refactor_eval/codes.crossdomain.rec/baseline/BPR_related/EMCDR/{source_name}_{target_name}/shared_users_mapped_emb_dict_{args.current_epoch}.pickle', 'rb') as pf:
        shared_users_mapped_emb_dict = pickle.load(pf)
    ## source 2 : testing users are from target domain only users
    target_users_emb_dict = {}
    with open(f'/TOP/home/ythuang/CODE/tmp/refactor_eval/codes.crossdomain.rec/baseline/BPR_related/lfm_bpr_graphs/{target_name}_lightfm_bpr_{args.current_epoch}_10e-5.txt', 'r') as f:
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

# Get emb of all (training) items
#item_emb = {}
#with open(f'../lfm_bpr_graphs/{target_name}_lightfm_bpr_{args.current_epoch}_10e-5.txt', 'r') as f:
#    for line in f:
#        line = line.split('\t')
#        prefix = line[0]
#        prefix = prefix.replace(" ", "")
#        emb=line[1].split()
#        if 'user_' in prefix:
#            continue
#        else:
#            item_emb.update({ prefix: np.array(emb, dtype=np.float32) })
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
    
#print("Got embedding!")
#
#k_amount = args.top_ks
#k_max = max(k_amount)
#
#d=100
#count = 0
#total_rec=[0, 0, 0, 0, 0]
#total_ndcg=[0, 0, 0, 0, 0]
#
#print("Start counting...")
#
#def get_top_rec_and_eval(user):
#    total_rec=[0, 0, 0, 0, 0]
#    total_ndcg=[0, 0, 0, 0, 0]
#    if user not in user_emb.keys():
#      return total_rec, total_ndcg, 0
#    user_emb_vec = np.array(list(user_emb[user]))
#    user_emb_vec_m = np.matrix(user_emb_vec)
#    user_rec_pool = user_rec_dict[user]
#    if len(user_rec_pool) == 0:
#      return total_rec, total_ndcg, 0
#    filtered_item_emb = {k:v for k,v in item_emb.items() if k in user_rec_pool}
#    item_emb_vec = np.array(list(filtered_item_emb.values()))
#    # print("item_emb_vec: ", item_emb_vec.shape)
#    index = faiss.IndexFlatIP(d)
#    index.add(item_emb_vec)
#    D, I = index.search(user_emb_vec_m, k_max)
#    item_key=np.array(list(filtered_item_emb.keys())) 
#    recomm_list = item_key[I][0]
#
#    # ground truth
#    test_data = list(tar_test_df[tar_test_df['reviewerID'] == user].asin)
#    # test_data = ['item_' + str(asin) for asin in test_data for asin in test_data]
#
#    for k in range(len(k_amount)):
#        recomm_k = recomm_list[:k_amount[k]]
#        total_rec[k]+=calculate_Recall(test_data, recomm_k)
#        total_ndcg[k]+=calculate_NDCG(test_data, recomm_k)
#    return total_rec, total_ndcg, 1
#
## print(f"absolute count: {count}")
#
## record time 
#start_time = time.time() 
#with Pool(processes=args.workers) as pool:
#    m = pool.map(get_top_rec_and_eval, testing_users)
#
#total_count = 0
#total_rec = []
#total_ndcg = []
#
#for rec, ndcg, count in m:
#  total_rec.append(rec)
#  total_ndcg.append(ndcg)
#  total_count += count 
#
#total_rec = np.sum(np.array(total_rec), axis=0)
#total_ndcg = np.sum(np.array(total_ndcg), axis=0)
#count = total_count
#
#model_name = args.model_name
#dataset_pair = f"{src}_{tar}"
#test_mode=args.test_mode
#top_ks = k_amount
#save_exp_record(model_name, dataset_pair, test_mode, top_ks, total_rec, total_ndcg, count, save_dir, save_name, output_file)

import time
import argparse
import faiss
import numpy as np
import json
import pandas as pd
import pickle
import random
import re
import multiprocessing 
from multiprocessing import Pool
import os
import uuid

from evaluation.utility import save_exp_record, rank_and_score, generate_item_graph_df, generate_user_emb


os.chdir(os.path.dirname(os.path.realpath(__file__)))

parser=argparse.ArgumentParser(description='Calculate the similarity and recommend VOD items')
parser.add_argument('--data_dir', type=str, help='groundtruth files dir')
parser.add_argument('--save_dir', type=str, help='dir to save cav')
parser.add_argument('--save_name', type=str, help='name to save csv')
parser.add_argument('--output_file', type=str, help='output_file name')
parser.add_argument('--graph_file', type=str, help='graph_file')
parser.add_argument('--test_mode', type=str, help='{target, shared, cold}')
parser.add_argument('--ncore', type=int, help='core number', default=5)
parser.add_argument('--seed', type=int, help='random seed', default=3)
parser.add_argument('--n_worker', type=int, help='number of workers', default=None)
parser.add_argument('--src', type=str, help='souce name')
parser.add_argument('--tar', type=str, help='target name')
parser.add_argument('--model_name', type=str, help='cpr, lgn, lgn_s, bpr, bpr_s, emcdr, bitgcf')
parser.add_argument('--uid_i', type=str, help='(default for amz) unique id column for item', default='asin')
parser.add_argument('--uid_u', type=str, help='(default for amz) unique id column of user', default='reviewerID')
parser.add_argument('--top_ks', nargs='*', help='top_k to eval', default=[1, 3, 5, 10, 20], action='extend', type=int)

args=parser.parse_args()
print(args)

save_dir=args.save_dir

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_name=args.save_name
output_file=args.output_file
graph_file=args.graph_file
ncore = args.ncore
src, tar = args.src, args.tar

## sample testing users
random.seed(args.seed)
if args.test_mode == 'target':
    with open('{}/input_{ncore}core/{src}_{tar}_test_target_users.pickle'.format(args.data_dir, ncore=ncore, src=src, tar=tar), 'rb') as pf:
        testing_users = pickle.load(pf)
if args.test_mode == 'shared':
    with open('{}/input_{ncore}core/{src}_{tar}_test_shared_users.pickle'.format(args.data_dir, ncore=ncore, src=src, tar=tar), 'rb') as pf:
        testing_users = pickle.load(pf)
if args.test_mode == 'cold':
    with open('{}/input_{ncore}core/{src}_{tar}_test_cold_users.pickle'.format(args.data_dir, ncore=ncore, src=src, tar=tar), 'rb') as pf:
        testing_users = pickle.load(pf)

# rec pool
## load csjj_train_df
with open('{}/LOO_data_{ncore}core/{tar}_train.pickle'.format(args.data_dir, ncore=ncore, tar=tar), 'rb') as pf:
    tar_train_df = pickle.load(pf)
with open('{}/LOO_data_{ncore}core/{tar}_test.pickle'.format(args.data_dir, ncore=ncore, tar=tar), 'rb') as pf:
    tar_test_df = pickle.load(pf)

tar_train_df[args.uid_u] = tar_train_df[args.uid_u].apply(lambda x: 'user_'+x)
tar_test_df[args.uid_u]  = tar_test_df[args.uid_u].apply(lambda x: 'user_'+x)
total_item_set = set(tar_train_df[args.uid_i])

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

print("Done generating testing users' positive-negative pairs.")

print("Start getting embedding for each user and item...")
user_emb = generate_user_emb(graph_file)
item_graph_df= generate_item_graph_df(graph_file)
print("Got embedding!")

top_ks = args.top_ks
total_rec, total_ndcg, count = rank_and_score(testing_users, top_ks, user_emb, testing_users_rec_dict, item_graph_df, tar_test_df, n_worker, args.uid_u, args.uid_i)

model_name = args.model_name
dataset_pair = f"{src}_{tar}"
test_mode=args.test_mode
save_exp_record(model_name, dataset_pair, test_mode, top_ks, total_rec, total_ndcg, count, save_dir, save_name, output_file)
    

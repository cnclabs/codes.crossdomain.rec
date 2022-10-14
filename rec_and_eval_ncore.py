import time
import argparse
import faiss
import numpy as np
import json
import pandas as pd
import pickle
import random
from utility import calculate_Recall, calculate_NDCG
import re
import multiprocessing 
from multiprocessing import Pool
import os
import uuid

from evaluation.tools import save_exp_record

def generate_item_graph_df(graph_file):
    st = time.time()
    def _text_to_array(cell):
        emb = np.array(cell.split(), dtype=np.float32)
        emb = np.expand_dims(emb, axis=-1)
        
        return emb
        
    graph_df = pd.read_csv(graph_file, sep='\t', header=None, names=['node_id', 'embed'])
    item_graph_df = graph_df[~graph_df['node_id'].str.startswith('user_')]
    item_graph_df['embed'] = item_graph_df['embed'].apply(_text_to_array)

    print('item graph shape:', item_graph_df.shape)
    print('Finished gen item graph df!', time.time() - st)
    
    return item_graph_df

def generate_user_emb(graph_file):
    user_emb={}
    with open(graph_file, 'r') as f:
        for line in f:
            line = line.split('\t')
            prefix = line[0]
            emb=line[1].split()
            #if prefix in testing_users:
            if "user_" in prefix:
                user_emb.update({ prefix: np.array(emb, dtype=np.float32) })
    return user_emb

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


print("Testing users amount = {}".format(len(testing_users)))
print("Start counting...")
k_amount = args.top_ks
k_max = max(k_amount)
def get_top_rec_and_eval(user):
    total_rec=[0, 0, 0, 0, 0]
    total_ndcg=[0, 0, 0, 0, 0]
    d=100
    if user not in user_emb.keys():
      return total_rec, total_ndcg, 0

    user_emb_vec = np.array(list(user_emb[user]))
    user_emb_vec_m = np.matrix(user_emb_vec)
    
    user_rec_pool = testing_users_rec_dict[user]
    
    _tmp_df = item_graph_df[item_graph_df['node_id'].isin(user_rec_pool)]
    item_emb_vec = np.concatenate(list(_tmp_df['embed']), axis = -1).T
    item_emb_vec = item_emb_vec.copy(order='C')
    
    item_key = np.array(list(_tmp_df['node_id']))

    if len(user_rec_pool) == 0:
      return total_rec, total_ndcg, 0
    index = faiss.IndexFlatIP(d)
    index.add(item_emb_vec)
    D, I = index.search(user_emb_vec_m, k_max)
    recomm_list = item_key[I][0]

    # ground truth
    test_data = list(tar_test_df[tar_test_df[args.uid_u] == user][args.uid_i])

    for k in range(len(k_amount)):
        recomm_k = recomm_list[:k_amount[k]]
        total_rec[k]+=calculate_Recall(test_data, recomm_k)
        total_ndcg[k]+=calculate_NDCG(test_data, recomm_k)

    return total_rec, total_ndcg, 1

st = time.time()
with Pool(processes=args.n_worker) as pool:
    m = pool.map(get_top_rec_and_eval, testing_users)

print('Done counting.', time.time() - st)

total_rec = []
total_ndcg = []
for rec, ndcg, _ in m:
  total_rec.append(rec)
  total_ndcg.append(ndcg)
count = len(total_rec) 
total_rec = np.sum(np.array(total_rec), axis=0)
total_ndcg = np.sum(np.array(total_ndcg), axis=0)

model_name = args.model_name
dataset_pair = f"{src}_{tar}"
test_mode=args.test_mode
top_ks = k_amount
save_exp_record(model_name, dataset_pair, test_mode, top_ks, total_rec, total_ndcg, count, save_dir, save_name, output_file)

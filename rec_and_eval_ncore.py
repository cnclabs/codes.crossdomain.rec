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

os.chdir(os.path.dirname(os.path.realpath(__file__)))

parser=argparse.ArgumentParser(description='Calculate the similarity and recommend VOD items')
parser.add_argument('--mom_save_dir', type=str, help='groundtruth files dir')
parser.add_argument('--output_file', type=str, help='output_file name')
parser.add_argument('--graph_file', type=str, help='graph_file')
parser.add_argument('--test_users', type=str, help='{target, shared, cold}')
parser.add_argument('--ncore', type=int, help='core number', default=5)
parser.add_argument('--sample', type=int, help='sample amount to eval', default=4000)
parser.add_argument('--n_worker', type=int, help='number of workers', default=None)
parser.add_argument('--src', type=str, help='souce name')
parser.add_argument('--tar', type=str, help='target name')
parser.add_argument('--uid_i', type=str, help='(default for amz) unique id column for item', default='asin')
parser.add_argument('--uid_u', type=str, help='(default for amz) unique id column of user', default='reviewerID')


args=parser.parse_args()
print(args)

output_file=args.output_file
graph_file=args.graph_file
sample_amount = args.sample
ncore = args.ncore
src, tar = args.src, args.tar

user_emb={}
item_emb={}

# ground truth, csjj is target
with open('{}/LOO_data_{ncore}core/{tar}_test.pickle'.format(args.mom_save_dir, ncore=ncore, tar=tar), 'rb') as pf:
    tar_test_df = pickle.load(pf)
tar_test_df[args.uid_u] = tar_test_df[args.uid_u].apply(lambda x: 'user_'+x)
# csjj_test_df unique reviewerID = 451806

## sample testing users
random.seed(3)
if args.test_users == 'target':
  testing_users = random.sample(set(tar_test_df[args.uid_u]), sample_amount)
if args.test_users == 'shared':
  with open('{}/user_{ncore}core/{src}_{tar}_shared_users.pickle'.format(args.mom_save_dir, ncore=ncore, src=src, tar=tar), 'rb') as pf:
    shared_users = pickle.load(pf)
  testing_users = random.sample(set(shared_users), sample_amount)
  testing_users = set(map(lambda x: "user_"+x, testing_users))
if args.test_users == 'cold':
  with open('{}/user_{ncore}core/{src}_{tar}_cold_users.pickle'.format(args.mom_save_dir, ncore=ncore, src=src, tar=tar), 'rb') as pf:
    cold_users = pickle.load(pf)
  testing_users = cold_users 
  testing_users = set(map(lambda x: "user_"+x, testing_users))


# rec pool
## load csjj_train_df
with open('{}/LOO_data_{ncore}core/{tar}_train.pickle'.format(args.mom_save_dir, ncore=ncore, tar=tar), 'rb') as pf:
  tar_train_df = pickle.load(pf)
tar_train_df[args.uid_u] = tar_train_df[args.uid_u].apply(lambda x: 'user_'+x)
total_item_set = set(tar_train_df[args.uid_i])


# Generate user 100 rec pool

def process_user_pos_neg_pair(user_list):
  user_rec_dict = {}

  for user in user_list:
      pos_pool = set(tar_train_df[tar_train_df[args.uid_u] == user][args.uid_i])
      neg_pool = total_item_set - pos_pool
      random.seed(5)
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

user_emb={}
item_emb={}

# get emb of testing users and all (training) items
print("Start getting embedding for each user and item...")
with open(graph_file, 'r') as f:
    for line in f:
        line = line.split('\t')
        prefix = line[0]
        emb=line[1].split()
        #if prefix in testing_users:
        if "user_" in prefix:
            user_emb.update({ prefix: np.array(emb, dtype=np.float32) })
        else:
            item_emb.update({ prefix: np.array(emb, dtype=np.float32) })

print("Got embedding!")

k_amount = [1, 3, 5, 10, 20]
k_max = max(k_amount)

d=100
count = 0
total_rec=[0, 0, 0, 0, 0]
total_ndcg=[0, 0, 0, 0, 0]

print("Testing users amount = {}".format(len(testing_users)))
print("Start counting...")

for user in testing_users:
    count+=1
    user_emb_vec = np.array(list(user_emb[user]))
    user_emb_vec_m = np.matrix(user_emb_vec)
    user_rec_pool = testing_users_rec_dict[user]
    filtered_item_emb = {k:v for k,v in item_emb.items() if k in user_rec_pool}
    item_emb_vec = np.array(list(filtered_item_emb.values()))
    index = faiss.IndexFlatIP(d)
    index.add(item_emb_vec)
    D, I = index.search(user_emb_vec_m, k_max)
    item_key=np.array(list(filtered_item_emb.keys())) 
    recomm_list = item_key[I][0]

    if count%1000 == 0:
        print("{} users counted.".format(count))

    # ground truth
    test_data = list(tar_test_df[tar_test_df[args.uid_u] == user][args.uid_i])

    for k in range(len(k_amount)):
        recomm_k = recomm_list[:k_amount[k]]
        total_rec[k]+=calculate_Recall(test_data, recomm_k)
        total_ndcg[k]+=calculate_NDCG(test_data, recomm_k)

total_rec=np.array(total_rec)
total_ndcg=np.array(total_ndcg)

print("Start writing file...")
with open(output_file, 'w') as fw:
    fw.writelines(['=================================\n',
           'File: ',
            str(graph_file),
            '\n evaluated users: ',
            str(len(testing_users)),
            '\n--------------------------------',
           '\n recall@1: ',
            str(total_rec[0]/count),
           '\n NDCG@1: ',
            str(total_ndcg[0]/count),
           '\n--------------------------------',
           '\n recall@3: ',
            str(total_rec[1]/count),
           '\n NDCG@3: ',
            str(total_ndcg[1]/count),
           '\n--------------------------------',
           '\n recall@5: ',
            str(total_rec[2]/count),
           '\n NDCG@5: ',
            str(total_ndcg[2]/count),
           '\n--------------------------------',
           '\n recall@10: ',
           str(total_rec[3]/count),
           '\n NDCG@10: ',
           str(total_ndcg[3]/count),
           '\n--------------------------------',
           '\n recall@20: ',
           str(total_rec[4]/count),
           '\n NDCG@20: ',
           str(total_ndcg[4]/count),
           '\n'])

print('Finished!')

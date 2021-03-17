import argparse
import faiss
import numpy as np
import json
import pandas as pd
import pickle
import random
from utility import calculate_Recall, calculate_NDCG
import re

parser=argparse.ArgumentParser(description='Calculate the similarity and recommend VOD items')
parser.add_argument('--output_file', type=str, help='output_file name')
parser.add_argument('--graph_file', type=str, help='graph_file')
parser.add_argument('--test_users', type=str, help='{target, shared, cold}')


args=parser.parse_args()
output_file=args.output_file
graph_file=args.graph_file

user_emb={}
item_emb={}

# ground truth 
with open('../LOO_data/vod_test.pickle', 'rb') as pf:
    vod_test_df = pickle.load(pf)
vod_test_df['user_id'] = vod_test_df['user_id'].apply(lambda x: 'user_'+x)
vod_test_df['item_id'] = vod_test_df['item_id'].apply(lambda x: 'vod_'+x)
# vod_test_df unique user_id = 23744

## sample testing users
sample_amount = 2000 
random.seed(3)
if arg.test_users == 'target':
  testing_users = random.sample(set(vod_test_df.user_id), sample_amount)
if args.test_users == 'shared':
  with open('../user/tv_vod_shared_users.pickle', 'rb') as pf:
    shared_users = pickle.load(pf)
  testing_users = random.sample(set(shared_users), sample_amount)
if args.test_users == 'cold':
  with open('../user/tv_vod_cold_users.pickle', 'rb') as pf:
    cold_users = pickle.load(pf)
  testing_users = cold_users # already remain the sample amount of users

# rec pool
## load vod_train_df
with open('../LOO_data/vod_train.pickle', 'rb') as pf:
  vod_train_df = pickle.load(pf)
vod_train_df['user_id'] = vod_train_df['user_id'].apply(lambda x: 'user_'+x)
vod_train_df['item_id'] = vod_train_df['item_id'].apply(lambda x: 'vod_'+x)
total_item_set = set(vod_train_df.item_id)

# Generate user 100 rec pool
print("Start generating user rec dict...")
user_rec_dict = {}
for user in testing_users:
    watched_set = set(vod_train_df[vod_train_df['user_id'] == user].item_id)
    neg_pool = total_item_set - watched_set
    random.seed(5)
    neg_99 = random.sample(neg_pool, 99)
    user_rec_pool = list(neg_99) + list(vod_test_df[vod_test_df['user_id'] == user].item_id)
    user_rec_dict[user] = user_rec_pool
print("user rec dict generated!")

user_emb={}
item_emb={}

# get emb of testing users and all (training) items
print("Start getting embedding...")
with open(graph_file, 'r') as f:
    for line in f:
        line = line.split('\t')
        prefix = line[0]
        emb=line[1].split()
        if prefix in testing_users:
            user_emb.update({ prefix: np.array(emb, dtype=np.float32) })
        elif prefix.split('_')[0] == 'vod':
            item_emb.update({ prefix: np.array(emb, dtype=np.float32) })
        else:
            continue
print("Got embedding!")

k_amount = [1, 3, 5, 10, 20]
k_max = max(k_amount)

d=100
count = 0
total_rec=[0, 0, 0, 0, 0]
total_ndcg=[0, 0, 0, 0, 0]

print("Start counting...")

for user in testing_users:
    count+=1
    user_emb_vec = np.array(list(user_emb[user]))
    user_emb_vec_m = np.matrix(user_emb_vec)
    user_rec_pool = user_rec_dict[user]
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
    test_data = list(vod_test_df[vod_test_df['user_id'] == user].item_id)

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
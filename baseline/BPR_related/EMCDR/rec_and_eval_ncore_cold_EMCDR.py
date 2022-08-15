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
from utility import calculate_Recall, calculate_NDCG
import multiprocessing 
from multiprocessing import Pool
import time

parser=argparse.ArgumentParser(description='Calculate the similarity and recommend target items')
parser.add_argument('--current_epoch', type=str)
parser.add_argument('--output_file', type=str, help='output_file name')
parser.add_argument('--workers', type=int, help='number of multi-processing workers')
parser.add_argument('--dataset_name', type=str, help='{tv_vod, csj_hk, mt_books, el_cpa, spo_csj}')
parser.add_argument('--ncore', type=int, help='core_filter', default=0)

args=parser.parse_args()
output_file=args.output_file
source_name = args.dataset_name.split('_')[0]
target_name = args.dataset_name.split('_')[1]
ncore = args.ncore

# ground truth 
with open('../../../LOO_data_{}core/{}_test.pickle'.format(ncore, target_name), 'rb') as pf:
    tar_test_df = pickle.load(pf)
tar_test_df['reviewerID'] = tar_test_df['reviewerID'].apply(lambda x: 'user_'+x)

# testing users : cold-start users
with open('../../../user_{}core/{}_{}_cold_users.pickle'.format(ncore, source_name, target_name), 'rb') as pf:
    src_tar_cold_start_users = pickle.load(pf)
testing_users = src_tar_cold_start_users
testing_users = ['user_'+user for user in testing_users]


# rec pool
## load tar_train_df
with open('../../../LOO_data_{}core/{}_train.pickle'.format(ncore, target_name), 'rb') as pf:
  tar_train_df = pickle.load(pf)
tar_train_df['reviewerID'] = tar_train_df['reviewerID'].apply(lambda x: 'user_'+x)

item_emb = {}
with open(f'../lfm_bpr_graphs/cold_{target_name}_lightfm_bpr_{args.current_epoch}_10e-5.txt', 'r') as f:
    for line in f:
        line = line.split('\t')
        prefix = line[0]
        prefix = prefix.replace(" ", "")
        emb=line[1].split()
        if 'user_' in prefix:
            continue
        else:
            item_emb.update({ prefix: np.array(emb, dtype=np.float32) })



print("Got item embedding!")
total_item_set = set(item_emb.keys())

# Generate user 100 rec pool
print("Start generating user rec dict...")

def process_user_rec_dict(user_list):
  user_rec_dict = {}

  for user in user_list:
      watched_set = set(tar_train_df[tar_train_df['reviewerID'] == user].asin)
      neg_pool = total_item_set - watched_set
      random.seed(5)
      neg_99 = random.sample(neg_pool, 99)
      user_rec_pool = list(neg_99) + list(tar_test_df[tar_test_df['reviewerID'] == user].asin)
      user_rec_dict[user] = user_rec_pool

  return user_rec_dict

cpu_amount = multiprocessing.cpu_count()
worker = cpu_amount - 2
mp = Pool(worker)
split_datas = np.array_split(list(testing_users), worker)
results = mp.map(process_user_rec_dict ,split_datas)
mp.close()

user_rec_dict = {}
for r in results:
  user_rec_dict.update(r)

print("testing users rec dict generated!")

# Get emb of testing users 

print("Start getting embedding of testing users...")
with open('./{}_{}/cold_users_mapped_emb_dict.pickle'.format(source_name, target_name), 'rb') as pf:
    cold_start_users_mapped_emb_dict = pickle.load(pf)

user_emb = cold_start_users_mapped_emb_dict

print("Got embedding!")

k_amount = [1, 3, 5, 10, 20]
k_max = max(k_amount)

d=100
count = 0
total_rec=[0, 0, 0, 0, 0]
total_ndcg=[0, 0, 0, 0, 0]

print("Start counting...")

def get_top_rec_and_eval(user):
    total_rec=[0, 0, 0, 0, 0]
    total_ndcg=[0, 0, 0, 0, 0]
    if user not in user_emb.keys():
      return total_rec, total_ndcg, 0
    user_emb_vec = np.array(list(user_emb[user]))
    user_emb_vec_m = np.matrix(user_emb_vec)
    user_rec_pool = user_rec_dict[user]
    if len(user_rec_pool) == 0:
      return total_rec, total_ndcg, 0
    filtered_item_emb = {k:v for k,v in item_emb.items() if k in user_rec_pool}
    item_emb_vec = np.array(list(filtered_item_emb.values()))
    # print("item_emb_vec: ", item_emb_vec.shape)
    index = faiss.IndexFlatIP(d)
    index.add(item_emb_vec)
    D, I = index.search(user_emb_vec_m, k_max)
    item_key=np.array(list(filtered_item_emb.keys())) 
    recomm_list = item_key[I][0]

    # ground truth
    test_data = list(tar_test_df[tar_test_df['reviewerID'] == user].asin)
    # test_data = ['item_' + str(asin) for asin in test_data for asin in test_data]

    for k in range(len(k_amount)):
        recomm_k = recomm_list[:k_amount[k]]
        total_rec[k]+=calculate_Recall(test_data, recomm_k)
        total_ndcg[k]+=calculate_NDCG(test_data, recomm_k)
    return total_rec, total_ndcg, 1

# print(f"absolute count: {count}")

# record time 
start_time = time.time() 
with Pool(processes=args.workers) as pool:
    m = pool.map(get_top_rec_and_eval, testing_users)

total_count = 0
total_rec = []
total_ndcg = []

for rec, ndcg, count in m:
  total_rec.append(rec)
  total_ndcg.append(ndcg)
  total_count += count 

total_rec = np.sum(np.array(total_rec), axis=0)
total_ndcg = np.sum(np.array(total_ndcg), axis=0)
count = total_count

# record time
end_time = time.time()
print(f"spend {end_time - start_time} secs on rec & eval")

print("Start writing file...")
with open(output_file, 'w') as fw:
    fw.writelines(['=================================\n',
           # 'File: ',
           # str(graph_file),
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

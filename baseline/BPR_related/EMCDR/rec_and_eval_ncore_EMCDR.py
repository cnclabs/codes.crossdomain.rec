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
parser.add_argument('--mom_save_dir', type=str, help='output_file name')
parser.add_argument('--src', type=str, help='souce name', default='hk')
parser.add_argument('--tar', type=str, help='target name', default='csjj')
parser.add_argument('--current_epoch', type=str)
parser.add_argument('--output_file', type=str, help='output_file name')
parser.add_argument('--test_users', type=str, help='{target, shared}')
parser.add_argument('--workers', type=int, help='number of multi-processing workers')
parser.add_argument('--dataset_name', type=str, help='{tv_vod, csj_hk, mt_books, el_cpa, spo_csj}')
parser.add_argument('--ncore', type=int, help='core_filter', default=0)

args=parser.parse_args()
source_name = args.src
target_name = args.tar
#target_name = dataset_name.split('_')[1]
ncore = args.ncore

# ground truth
with open('{}/LOO_data_{}core/{}_test.pickle'.format(args.mom_save_dir, ncore, target_name), 'rb') as pf:
    tar_test_df = pickle.load(pf)
tar_test_df['reviewerID'] = tar_test_df['reviewerID'].apply(lambda x: 'user_'+x)

# sample testing users
sample_amount = 4000
random.seed(3)
print("test_users", args.test_users)
if args.test_users == 'target':
    testing_users = random.sample(set(tar_test_df.reviewerID), sample_amount)
    # testing_users = ['user_'+user for user in testing_users]
if args.test_users == 'shared':
    with open('{}/user_{}core/{}_{}_shared_users.pickle'.format(args.mom_save_dir, ncore, source_name, target_name), 'rb') as pf:
        shared_users = pickle.load(pf)
    testing_users = random.sample(set(shared_users), sample_amount)
    testing_users = ['user_'+user for user in testing_users]

# rec pool
with open('{}/LOO_data_{}core/{}_train.pickle'.format(args.mom_save_dir, ncore, target_name), 'rb') as pf:
    tar_train_df = pickle.load(pf)
tar_train_df['reviewerID'] = tar_train_df['reviewerID'].apply(lambda x: 'user_'+x)
total_item_set = set(tar_train_df.asin)

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
with open(f'./{source_name}_{target_name}/shared_users_mapped_emb_dict_{args.current_epoch}.pickle', 'rb') as pf:
        shared_users_mapped_emb_dict = pickle.load(pf)

if args.test_users == 'target':
    ## source 1 : testing users are from shared users
    with open(f'./{source_name}_{target_name}/shared_users_mapped_emb_dict_{args.current_epoch}.pickle', 'rb') as pf:
        shared_users_mapped_emb_dict = pickle.load(pf)
    ## source 2 : testing users are from target domain only users
    target_users_emb_dict = {}
    with open(f'../lfm_bpr_graphs/{target_name}_lightfm_bpr_{args.current_epoch}_10e-5.txt', 'r') as f:
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

if args.test_users == 'shared':
    shared_users_amount = len(testing_users)
    user_emb = {k:v for k,v in shared_users_mapped_emb_dict.items() if k in testing_users}

# Get emb of all (training) items
item_emb = {}
with open(f'../lfm_bpr_graphs/{target_name}_lightfm_bpr_{args.current_epoch}_10e-5.txt', 'r') as f:
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
with open(args.output_file, 'w') as fw:
    fw.writelines(['=================================\n',
           # 'File: ',
           # str(graph_file),
            '\n evaluated users: ',
            str(len(testing_users)),
            '\n shared users ratio: ',
            str(shared_users_amount/sample_amount),
            # '\n target only users ratio: ',
            # str(target_only_users_amount/sample_amount),
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







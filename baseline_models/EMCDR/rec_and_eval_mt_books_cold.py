import argparse
import faiss
import numpy as np
import json
import pandas as pd
import pickle
import random
import re
import sys
sys.path.insert(1, '../../CPR/')
from utility import calculate_Recall, calculate_NDCG
import multiprocessing 
from multiprocessing import Pool

parser=argparse.ArgumentParser(description='Calculate the similarity and recommend books items')
parser.add_argument('--output_file', type=str, help='output_file name')

args=parser.parse_args()
output_file=args.output_file

# ground truth 
with open('../../LOO_data/books_test.pickle', 'rb') as pf:
    books_test_df = pickle.load(pf)
books_test_df['reviewerID'] = books_test_df['reviewerID'].apply(lambda x: 'user_'+x)

# testing users : cold-start users
with open('../../user/mt_books_cold_users.pickle', 'rb') as pf:
    mt_books_cold_start_users = pickle.load(pf)
testing_users = mt_books_cold_start_users


# rec pool
## load books_train_df
with open('../../LOO_data/books_train.pickle', 'rb') as pf:
  books_train_df = pickle.load(pf)
books_train_df['reviewerID'] = books_train_df['reviewerID'].apply(lambda x: 'user_'+x)

item_emb = {}
with open('../BPR/graph/cold_books_lightfm_bpr_10e-5.txt', 'r') as f:
    skip_f = f.readlines()[1:]
    for line in skip_f:
      line = line[:-1]
      prefix = line.split(' ')[0]
      emb=line.split(' ')[3:]
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
      watched_set = set(books_train_df[books_train_df['reviewerID'] == user].asin)
      neg_pool = total_item_set - watched_set
      random.seed(5)
      neg_99 = random.sample(neg_pool, 99)
      user_rec_pool = list(neg_99) + list(books_test_df[books_test_df['reviewerID'] == user].asin)
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
with open('./mt_books/cold_users_mapped_emb_dict.pickle', 'rb') as pf:
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
    test_data = list(books_test_df[books_test_df['reviewerID'] == user].asin)

    for k in range(len(k_amount)):
        recomm_k = recomm_list[:k_amount[k]]
        total_rec[k]+=calculate_Recall(test_data, recomm_k)
        total_ndcg[k]+=calculate_NDCG(test_data, recomm_k)

total_rec=np.array(total_rec)
total_ndcg=np.array(total_ndcg)

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
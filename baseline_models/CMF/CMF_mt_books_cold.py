from cmfrec import CMF
import pickle
import numpy as np
import pandas as pd
from math import log
import argparse
import random
import sys
sys.path.insert(1, '../../CPR/')
from utility import calculate_Recall, calculate_NDCG
import multiprocessing 
from multiprocessing import Pool
import time

parser=argparse.ArgumentParser(description='CMF')
parser.add_argument('--output_file', type=str, help='output_file name')
parser.add_argument('--k', type=int, help='Number of latent factors to use in CMF model')

args=parser.parse_args()

with open('../../LOO_data/books_train.pickle', 'rb') as pk:
    books_train = pickle.load(pk)
books_train['reviewerID'] = books_train['reviewerID'].apply(lambda x: 'user_' + x)

# open mt books cold start users
with open('../../user/mt_books_cold_users.pickle', 'rb') as pf:
    mt_books_cold_start_users = pickle.load(pf)

filtered_books_train = books_train[~books_train['reviewerID'].isin(mt_books_cold_start_users)]
filtered_books_train = filtered_books_train[['reviewerID','asin','overall']]
books_ratings = filtered_books_train.groupby(['reviewerID', 'asin'])['overall'].mean().reset_index()
books_ratings.columns = ['UserId','ItemId','Rating']
print("Finished generating books_ratings...")

# getting side information

## get user information
## extract mt domain embedding of overlap_users as user_info
with open('../../LOO_data/mt_train.pickle', 'rb') as pf:
    mt_train = pickle.load(pf)
mt_train['reviewerID'] = mt_train['reviewerID'].apply(lambda x: 'user_' + x)
overlap_users = set(mt_train.reviewerID).intersection(set(filtered_books_train.reviewerID))

overlap_users_emb = {}
with open('../BPR/graph/mt_lightfm_bpr_10e-5.txt', 'r') as f:
    skip_f = f.readlines()[1:]
    for line in skip_f:
        line = line[:-1]
        prefix = line.split(' ')[0]
        emb = line.split(' ')[3:]
        if prefix in overlap_users:
            overlap_users_emb[prefix] = np.array(emb, dtype=np.float32)
print("Finished generating overlap_users_emb...")

## construct user_info
overlap_users_dict = {}
overlap_users_dict['UserId'] = list(overlap_users)
user_info = pd.DataFrame.from_dict(overlap_users_dict)
user_info['emb'] = user_info['UserId'].map(overlap_users_emb)

each_column = [i for i in range(100)]
user_info[each_column] = pd.DataFrame(user_info.emb.to_list(), index=user_info.index)
user_info = user_info.drop(columns='emb')
print("Finished constructing user_info!")


## get item embedding from target domain
item_emb_dict = {}
with open('../BPR/graph/cold_books_lightfm_bpr_10e-5.txt', 'r') as f:
    skip_f = f.readlines()[1:]
    for line in skip_f:
        line = line[:-1]
        prefix = line.split(' ')[0]
        emb = line.split(' ')[3:]
        if 'user_' not in prefix:
            item_emb_dict[prefix] = np.array(emb, dtype=np.float32)

## construct item_info
items_dict = {}
items_dict['ItemId'] = list(item_emb_dict.keys())
item_info = pd.DataFrame.from_dict(items_dict)
item_info['emb'] = item_info['ItemId'].map(item_emb_dict)
item_info[each_column] = pd.DataFrame(item_info.emb.to_list(), index=item_info.index)
item_info = item_info.drop(columns='emb')
print("Finished constructing item_info!")

# sort books_ratings 
overlap_users_books_ratings = books_ratings[books_ratings['UserId'].isin(overlap_users)]
non_overlap_users_books_ratings = books_ratings[~books_ratings['UserId'].isin(overlap_users)]
sorted_books_ratings = pd.concat([overlap_users_books_ratings, non_overlap_users_books_ratings], ignore_index=True)


print("Start fitting model...")
start_time = time.time()
model = CMF(method='als', k=args.k)
model.fit(X=sorted_books_ratings, U=user_info, I=item_info)
print("Finished fitting model!")
print("It took {} seconds to fit the model.".format(time.time() - start_time))

# ================== Rec and Eval ==================
# ground truth 
with open('../../LOO_data/books_test.pickle', 'rb') as pk:
    books_test_df = pickle.load(pk)
books_test_df['reviewerID'] = books_test_df['reviewerID'].apply(lambda x: 'user_'+x)

## testing users are cold-start users
testing_users = mt_books_cold_start_users

# generate user 100 rec pool
print("Start generating user rec dict...")

total_item_set = item_emb_dict.keys()

def process_user_rec_dict(user_list):
  user_rec_dict = {}

  for user in user_list:
      watched_set = set(books_train[books_train['reviewerID'] == user].asin)
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

k_amount = [1, 3, 5, 10, 20]
k_max = max(k_amount)
count = 0
total_rec=[0, 0, 0, 0, 0]
total_ndcg=[0, 0, 0, 0, 0]

# get cold-start users embedding from mt domain
cold_start_users_emb = {}
with open('../BPR/graph/mt_lightfm_bpr_10e-5.txt', 'r') as f:
    skip_f = f.readlines()[1:]
    for line in skip_f:
        line = line[:-1]
        prefix = line.split(' ')[0]
        emb = line.split(' ')[3:]
        if prefix in testing_users: # same as cold-start users
            cold_start_users_emb[prefix] = np.array(emb, dtype=np.float32)

print("Start counting...")
for user in testing_users:
    count += 1
    recomm_list = model.topN_cold(U=cold_start_users_emb[user], n=k_max)

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

output_file = args.output_file

print("Start writing file...")
with open(output_file, 'w') as fw:
    fw.writelines(['=================================\n',
           # 'Latent factors to use: ',
           # str(args.k),
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
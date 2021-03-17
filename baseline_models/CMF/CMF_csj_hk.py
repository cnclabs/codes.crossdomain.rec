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
import time

parser=argparse.ArgumentParser(description='CMF')
parser.add_argument('--output_file', type=str, help='output_file name')
parser.add_argument('--k', type=int, help='Number of latent factors to use in CMF model')
parser.add_argument('--test_users', type=str, help='target, shared')
args=parser.parse_args()

with open('../../LOO_data/hk_train.pickle', 'rb') as pk:
    hk_train = pickle.load(pk)

filtered_hk_train = hk_train[['reviewerID','asin','overall']]
hk_ratings = filtered_hk_train.groupby(['reviewerID', 'asin'])['overall'].mean().reset_index()
hk_ratings.columns = ['UserId','ItemId','Rating']
hk_ratings['UserId'] = hk_ratings['UserId'].apply(lambda x: 'user_'+x)
print("Finished generating hk_ratings...")

total_item_set = set(hk_ratings.ItemId)

with open('../../user/csj_hk_shared_users.pickle', 'rb') as pk:
    shared_users = pickle.load(pk)

# get user_info
# extract csj domain embedding of shared_users as user_info
shared_users_emb = {}
with open('../BPR/graph/csj_lightfm_bpr_10e-5.txt', 'r') as f:
    for line in f:
        # for lightfm bpr emb
        ## remove '\n'
        line = line[:-1]
        prefix = line.split(' ')[0]
        # ignore first two elements
        emb=line.split(' ')[3:]
        if prefix in shared_users:
            shared_users_emb[prefix] = np.array(emb, dtype=np.float32)
print("Finished generating shared_users_emb...")

# construct user_info
shared_users_dict = {}
shared_users_dict['UserId'] = list(shared_users)
user_info = pd.DataFrame.from_dict(shared_users_dict)
user_info['emb'] = user_info['UserId'].map(shared_users_emb)

each_column = [i for i in range(100)]
user_info[each_column] = pd.DataFrame(user_info.emb.to_list(), index=user_info.index)
user_info = user_info.drop(columns='emb')
print("Finished constructing user_info!")


# get item embedding from target domain
item_emb_dict = {}
with open('../../BPR/graph/hk_lightfm_bpr_10e-5.txt', 'r') as f:
    for line in f:
        # for lightfm bpr emb
        ## remove '\n'
        line = line[:-1]
        prefix = line.split(' ')[0]
        # ignore first two elements
        emb=line.split(' ')[3:]
        if 'user_' not in prefix:
            item_emb_dict[prefix] = np.array(emb, dtype=np.float32)

# construct item_info
items_dict = {}
items_dict['ItemId'] = list(item_emb_dict.keys())
item_info = pd.DataFrame.from_dict(items_dict)
item_info['emb'] = item_info['ItemId'].map(item_emb_dict)
item_info[each_column] = pd.DataFrame(item_info.emb.to_list(), index=item_info.index)
item_info = item_info.drop(columns='emb')
print("Finished constructing item_info!")

# sort hk_ratings, putting shared_users' logs to the upper part 
shared_users_hk_ratings = hk_ratings[hk_ratings['UserId'].isin(shared_users)]
non_shared_users_hk_ratings = hk_ratings[~hk_ratings['UserId'].isin(shared_users)]
sorted_hk_ratings = pd.concat([shared_users_hk_ratings, non_shared_users_hk_ratings], ignore_index=True)

print("Start fitting model...")
start_time = time.time()
model = CMF(method='als', k=args.k)
model.fit(X=sorted_hk_ratings, U=user_info, I=item_info)
print("Finished fitting model!")
print("It took {} seconds to fit the model.".format(time.time() - start_time))

# rec and eval
# ground truth 
with open('../../LOO_data/hk_test.pickle', 'rb') as pk:
    hk_test_df = pickle.load(pk)
hk_test_df['reviewerID'] = hk_test_df['reviewerID'].apply(lambda x: 'user_'+x)

## sample testing users
sample_amount = 4000 
random.seed(3)
if args.test_users == 'target':
  testing_users = random.sample(set(hk_test_df.reviewerID), sample_amount)
if args.test_users == 'shared':
  testing_users = random.sample(shared_users, sample_amount)


# Generate user 100 rec pool
print("Start generating user rec dict...")
user_rec_dict = {}
for user in testing_users:
    watched_set = set(hk_train[hk_train['reviewerID'] == user].asin)
    neg_pool = total_item_set - watched_set
    random.seed(5)
    neg_99 = random.sample(neg_pool, 99)
    user_rec_pool = list(neg_99) + list(hk_test_df[hk_test_df['reviewerID'] == user].asin)
    user_rec_dict[user] = user_rec_pool
print("user rec dict generated!")

def maxN(list, n):
    return sorted(list, reverse=True)[:n]

k_amount = [1, 3, 5, 10, 20]
k_max = max(k_amount)
count = 0
total_rec=[0, 0, 0, 0, 0]
total_ndcg=[0, 0, 0, 0, 0]

print("Start counting...")
for user in testing_users:
    count += 1
    rec_pool_scores = model.predict(user=[user]*100, item=user_rec_dict[user])
    index_list = []
    for v in maxN(rec_pool_scores, k_max):
        index_list.append(list(rec_pool_scores).index(v))
    recomm_list = np.array(user_rec_dict[user])[index_list]

    if count%1000 == 0:
        print("{} users counted.".format(count))

    # ground truth
    test_data = list(hk_test_df[hk_test_df['reviewerID'] == user].asin)

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





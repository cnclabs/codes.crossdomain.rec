from cmfrec import CMF
import pickle
import numpy as np
import pandas as pd
from math import log
import argparse
import random
rng = np.random.default_rng()
import sys
sys.path.insert(1, '../../CPR/')
from utility import calculate_Recall, calculate_NDCG
import multiprocessing 
from multiprocessing import Pool
import time
from tqdm import tqdm

parser=argparse.ArgumentParser(description='CMF')
parser.add_argument('--mom_save_dir', type=str, help='')
parser.add_argument('--current_epoch', type=str, help='current_epoch num')
parser.add_argument('--output_file', type=str, help='output_file name')
parser.add_argument('--k', type=int, help='Number of latent factors to use in CMF model')
parser.add_argument('--workers', type=int, help='number of multi-processing workers')
parser.add_argument('--dataset', type=str, help='graph_file')
parser.add_argument('--ncore', type=int, help='core_filter', default=0)
args=parser.parse_args()

source_domain, target_domain = args.dataset.split("_")
ncore = args.ncore

with open(f'{args.mom_save_dir}/LOO_data_{str(ncore)}core/{target_domain}_train.pickle', 'rb') as pk:
    target_domain_train = pickle.load(pk)
target_domain_train['reviewerID'] = target_domain_train['reviewerID'].apply(lambda x: 'user_' + x)

# open mt books cold start users
with open(f'{args.mom_save_dir}/user_{str(ncore)}core/{source_domain}_{target_domain}_cold_users.pickle', 'rb') as pf:
    dataset_cold_start_users = pickle.load(pf)
dataset_cold_start_users = ["user_"+user for user in dataset_cold_start_users]

filtered_target_domain_train = target_domain_train[~target_domain_train['reviewerID'].isin(dataset_cold_start_users)]
filtered_target_domain_train = filtered_target_domain_train[['reviewerID','asin','overall']]
target_domain_ratings = filtered_target_domain_train.groupby(['reviewerID', 'asin'])['overall'].mean().reset_index()
target_domain_ratings.columns = ['UserId','ItemId','Rating']
print("Finished generating target_domain_ratings...")

# getting side information

## get user information
## extract mt domain embedding of overlap_users as user_info
with open(f'{args.mom_save_dir}/LOO_data_{str(ncore)}core/{source_domain}_train.pickle', 'rb') as pf:
    source_domain_train = pickle.load(pf)
source_domain_train['reviewerID'] = source_domain_train['reviewerID'].apply(lambda x: 'user_' + x)
overlap_users = set(source_domain_train.reviewerID).intersection(set(filtered_target_domain_train.reviewerID))

overlap_users_emb = {}
with open(f'../lfm_bpr_graphs/{source_domain}_lightfm_bpr_{args.current_epoch}_10e-5.txt', 'r') as f:
    for line in f:
        line = line.strip("\n")
        prefix, emb = line.split('\t')
        prefix = prefix.replace(" ", "")
        emb = emb.split()
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
with open(f'../lfm_bpr_graphs/cold_{target_domain}_lightfm_bpr_{args.current_epoch}_10e-5.txt', 'r') as f:
    for line in f:
        line = line.strip("\n")
        prefix, emb = line.split('\t')
        prefix = prefix.replace(" ", "")
        emb = emb.split()
        if 'user_' not in prefix:
            item_emb_dict[prefix] = np.array(emb, dtype=np.float32)

print("prefix: ", prefix)
print(list(item_emb_dict.keys())[:10], len(emb))

## construct item_info
items_dict = {}
items_dict['ItemId'] = list(item_emb_dict.keys())
item_info = pd.DataFrame.from_dict(items_dict)
item_info['emb'] = item_info['ItemId'].map(item_emb_dict)
item_info[each_column] = pd.DataFrame(item_info.emb.to_list(), index=item_info.index)
item_info = item_info.drop(columns='emb')
print("Finished constructing item_info!")

# sort target_domain_ratings 
overlap_users_target_domain_ratings = target_domain_ratings[target_domain_ratings['UserId'].isin(overlap_users)]
non_overlap_users_target_domain_ratings = target_domain_ratings[~target_domain_ratings['UserId'].isin(overlap_users)]
sorted_target_domain_ratings = pd.concat([overlap_users_target_domain_ratings, non_overlap_users_target_domain_ratings], ignore_index=True)


print("Start fitting model...")
start_time = time.time()
model = CMF(method='als', k=args.k)
model.fit(X=sorted_target_domain_ratings, U=user_info, I=item_info)
print("Finished fitting model!")
print("It took {} seconds to fit the model.".format(time.time() - start_time))

# ================== Rec and Eval ==================
# ground truth 
with open(f'{args.mom_save_dir}/LOO_data_{str(ncore)}core/{target_domain}_test.pickle', 'rb') as pk:
    target_domain_test_df = pickle.load(pk)
target_domain_test_df['reviewerID'] = target_domain_test_df['reviewerID'].apply(lambda x: 'user_'+x)

## testing users are cold-start users
testing_users = dataset_cold_start_users

# generate user 100 rec pool
print("Start generating user rec dict...")

total_item_set = item_emb_dict.keys()

def process_user_rec_dict(user_list):
  user_rec_dict = {}

  for user in tqdm(user_list):
      watched_set = set(target_domain_train[target_domain_train['reviewerID'] == user].asin)
      neg_pool = total_item_set - watched_set
      # random.seed(5)
      neg_99 = random.sample(neg_pool, 99)
      user_rec_pool = list(neg_99) + list(target_domain_test_df[target_domain_test_df['reviewerID'] == user].asin)
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
# total_rec=[0, 0, 0, 0, 0]
# total_ndcg=[0, 0, 0, 0, 0]

# get cold-start users embedding from mt domain
cold_start_users_emb = {}
with open(f'../lfm_bpr_graphs/{source_domain}_lightfm_bpr_{args.current_epoch}_10e-5.txt', 'r') as f:
    for line in f:
        line = line.strip("\n")
        prefix, emb = line.split('\t')
        prefix = prefix.replace(" ", "")
        emb = emb.split()
        if prefix in testing_users: # same as cold-start users
            cold_start_users_emb[prefix] = np.array(emb, dtype=np.float32)

print("Start counting...")
# for user in testing_users:
def get_top_rec_and_eval(user):
    total_rec=[0, 0, 0, 0, 0]
    total_ndcg=[0, 0, 0, 0, 0]
    if user not in cold_start_users_emb.keys():
      return total_rec, total_ndcg, 0

    
    recomm_list = model.topN_cold(U=cold_start_users_emb[user], n=k_max)
    if len(recomm_list) == 0:
      return total_rec, total_ndcg, 0

    # ground truth
    test_data = list(target_domain_test_df[target_domain_test_df['reviewerID'] == user].asin)

    for k in range(len(k_amount)):
        recomm_k = recomm_list[:k_amount[k]]
        total_rec[k]+=calculate_Recall(test_data, recomm_k)
        total_ndcg[k]+=calculate_NDCG(test_data, recomm_k)
    return total_rec, total_ndcg, 1

# record time 
start_time = time.time() 
# with Pool(processes=args.workers) as pool:
#     m = pool.map(get_top_rec_and_eval, testing_users)

total_count = 0
total_rec = []
total_ndcg = []
    
for user in tqdm(testing_users):
    # m = get_top_rec_and_eval(user)
    rec, ndcg, count = get_top_rec_and_eval(user)
    # for rec, ndcg, count in m:
    total_rec.append(rec)
    total_ndcg.append(ndcg)
    total_count += count 

total_rec = np.sum(np.array(total_rec), axis=0)
total_ndcg = np.sum(np.array(total_ndcg), axis=0)
count = total_count

# record time
end_time = time.time()
print(f"spend {end_time - start_time} secs on rec & eval")


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

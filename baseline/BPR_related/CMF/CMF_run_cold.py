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

sys.path.insert(1, '../../../')
from evaluation.utility import (save_exp_record,
        rank_and_score,
        generate_item_graph_df,
        generate_user_emb,
        load_testing_users_rec_dict)

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

data_input_dir='/TOP/tmp2/cpr/fix_ncore_test/input_5core/'
test_mode='cold'
src='hk'
tar='csjj'
testing_users_rec_dict = load_testing_users_rec_dict(data_input_dir, test_mode, src, tar)
testing_users = list(testing_users_rec_dict.keys())

cold_start_users_emb = {}
with open(f'../lfm_bpr_graphs/{source_domain}_lightfm_bpr_{args.current_epoch}_10e-5.txt', 'r') as f:
    for line in f:
        line = line.strip("\n")
        prefix, emb = line.split('\t')
        prefix = prefix.replace(" ", "")
        emb = emb.split()
        if prefix in testing_users: # same as cold-start users
            cold_start_users_emb[prefix] = np.array(emb, dtype=np.float32)

with open(f'{args.mom_save_dir}/LOO_data_{str(ncore)}core/{target_domain}_train.pickle', 'rb') as pk:
    target_domain_train = pickle.load(pk)
target_domain_train['reviewerID'] = target_domain_train['reviewerID'].apply(lambda x: 'user_' + x)

filtered_target_domain_train = target_domain_train[~target_domain_train['reviewerID'].isin(testing_users)]
filtered_target_domain_train = filtered_target_domain_train[['reviewerID','asin','overall']]
filtered_target_domain_ratings = filtered_target_domain_train.groupby(['reviewerID', 'asin'])['overall'].mean().reset_index()
filtered_target_domain_ratings.columns = ['UserId','ItemId','Rating']
print("Finished generating target_domain_ratings...")

with open('/TOP/tmp2/cpr/fix_ncore_test/user_5core/hk_csjj_shared_users.pickle', 'rb') as pf:
    su = pickle.load(pf)
su = set(map(lambda x: "user_"+x, su))
overlap_users = set(su) - set(testing_users)

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
print("Finished constructing user_info!", user_info.shape)

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

print("Start fitting model...")
start_time = time.time()
model = CMF(method='als', k=args.k)
model.fit(X=filtered_target_domain_ratings, U=user_info, I=item_info)
print("Finished fitting model!")
print("It took {} seconds to fit the model.".format(time.time() - start_time))

# ================== Rec and Eval ==================
# ground truth 
with open(f'{args.mom_save_dir}/LOO_data_{str(ncore)}core/{target_domain}_test.pickle', 'rb') as pk:
    target_domain_test_df = pickle.load(pk)
target_domain_test_df['reviewerID'] = target_domain_test_df['reviewerID'].apply(lambda x: 'user_'+x)

## testing users are cold-start users
print("Start generating user rec dict...")

total_item_set = item_emb_dict.keys()


k_amount = [1, 3, 5, 10, 20]
k_max = max(k_amount)
count = 0


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

model_name='cmf'
dataset_pair='hk_csjj'
test_mode='cold'
top_ks = [1, 3, 5, 10, 20]
save_dir='/TOP/tmp2/cpr/exp_record_test/'
save_name='M_cmf_D_hk_csjj_T_cold'
save_exp_record(model_name, dataset_pair, test_mode, top_ks, total_rec, total_ndcg, count, save_dir, save_name)

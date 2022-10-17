import pickle
import pandas as pd 
from math import log
import os
import argparse
import random

os.chdir(os.path.dirname(os.path.realpath(__file__)))

parser=argparse.ArgumentParser(description='Generated correspond inputs from LOO datas with n core.')
parser.add_argument('--mom_save_dir', type=str, help='output_file name')
parser.add_argument('--sample', type=int, help='sample amount to test', default=4000)
parser.add_argument('--ncore', type=int, help='core number', default=5)
parser.add_argument('--dataset', type=str, help='dataset name, e.g. hk_csj', default='hk_csj')
parser.add_argument('--item_attr', type=str, help='(default for amz) attribute represents items\' ids', default='asin')
parser.add_argument('--user_attr', type=str, help='(default for amz) attribute represents users\' ids', default='reviewerID')
args=parser.parse_args()
print(args)


ncore = args.ncore
assert '_' in args.dataset, "Make sure your dataset name exists \'_\' to split src & target."
dataset = args.dataset.split('_')
src, tar = dataset[0], dataset[1]
item_attr, user_attr = args.item_attr, args.user_attr
sample_amount = args.sample
mom_save_dir = args.mom_save_dir

save_dir = "../dataset"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ------------ SRC-TAR ------------
# tar
with open('{mom_save_dir}/LOO_data_{ncore}core/{tar}_train.pickle'.format(mom_save_dir=mom_save_dir, ncore=ncore, tar=tar), 'rb') as pf:
    tar_train = pickle.load(pf)
tar_train[user_attr] = tar_train[user_attr].apply(lambda x: 'user_'+x)

with open('{mom_save_dir}/LOO_data_{ncore}core/{tar}_test.pickle'.format(mom_save_dir=mom_save_dir, ncore=ncore, tar=tar), 'rb') as pf:
    tar_test_df = pickle.load(pf)
tar_test_df[user_attr] = tar_test_df[user_attr].apply(lambda x: 'user_'+x)

tar_train_graph = tar_train.groupby([user_attr, item_attr]).size().reset_index().drop([0], axis=1)
_df = tar_train_graph
_df.to_csv(os.path.join(save_dir,'{src}_{tar}_lil_target.train'.format(tar=tar, src=src)), header=False, index=False, sep=',')
_df.to_csv(os.path.join(save_dir,'{src}_{tar}_lil_shared.train'.format(tar=tar, src=src)), header=False, index=False, sep=',')

target_users = random.sample(set(tar_test_df[user_attr]), sample_amount)
with open('{mom_save_dir}/user_{ncore}core/{src}_{tar}_shared_users.pickle'.format(mom_save_dir=mom_save_dir, ncore=ncore, src=src, tar=tar), 'rb') as pf:
    shared_users = pickle.load(pf)
shared_users = random.sample(set(shared_users), sample_amount)
shared_users = set(map(lambda x: "user_"+x, shared_users))
with open('{mom_save_dir}/user_{ncore}core/{src}_{tar}_cold_users.pickle'.format(mom_save_dir=mom_save_dir, ncore=ncore, src=src, tar=tar), 'rb') as pf:
    cold_users = pickle.load(pf)
cold_users = set(map(lambda x: "user_"+x, cold_users))

# make tests
target_test_df = tar_test_df[tar_test_df[user_attr].isin(target_users)].groupby([user_attr, item_attr]).size().reset_index().drop([0], axis=1)
shared_test_df = tar_test_df[tar_test_df[user_attr].isin(shared_users)].groupby([user_attr, item_attr]).size().reset_index().drop([0], axis=1)
cold_test_df = tar_test_df[tar_test_df[user_attr].isin(cold_users)].groupby([user_attr, item_attr]).size().reset_index().drop([0], axis=1)
target_test_df.to_csv(os.path.join(save_dir,'{src}_{tar}_lil_target.test'.format(tar=tar, src=src)), header=False, index=False, sep=',')
shared_test_df.to_csv(os.path.join(save_dir,'{src}_{tar}_lil_shared.test'.format(tar=tar, src=src)), header=False, index=False, sep=',')
target_test_df.to_csv(os.path.join(save_dir,'{src}_{tar}_big_target.test'.format(tar=tar, src=src)), header=False, index=False, sep=',')
shared_test_df.to_csv(os.path.join(save_dir,'{src}_{tar}_big_shared.test'.format(tar=tar, src=src)), header=False, index=False, sep=',')
cold_test_df.to_csv(os.path.join(save_dir,'{src}_{tar}_big_cold.test'.format(tar=tar, src=src)), header=False, index=False, sep=',')

# src (source)
with open('{mom_save_dir}/LOO_data_{ncore}core/{src}_train.pickle'.format(mom_save_dir=mom_save_dir, ncore=ncore, src=src), 'rb') as pf:
    src_train = pickle.load(pf)

src_train[user_attr] = src_train[user_attr].apply(lambda x: 'user_'+x)
src_train_graph = src_train.groupby([user_attr, item_attr]).size().reset_index().drop([0], axis=1)
#src_train_graph.to_csv(os.path.join(save_dir, 'all_{src}_train_input.txt'.format(src=src)), header=False, index=False, sep='\t')
pd.concat([tar_train_graph, src_train_graph]).to_csv(os.path.join(save_dir, '{src}_{tar}_big_target.train'.format(tar=tar, src=src)), header=False, index=False, sep=',')
pd.concat([tar_train_graph, src_train_graph]).to_csv(os.path.join(save_dir, '{src}_{tar}_big_shared.train'.format(tar=tar, src=src)), header=False, index=False, sep=',')

# for cold start
with open('{mom_save_dir}/user_{ncore}core/{src}_{tar}_cold_users.pickle'.format(mom_save_dir=mom_save_dir, ncore=ncore, tar=tar, src=src), 'rb') as pf:
    src_tar_cold_users = pickle.load(pf)
with open('{mom_save_dir}/LOO_data_{ncore}core/{tar}_train.pickle'.format(mom_save_dir=mom_save_dir, ncore=ncore, tar=tar), 'rb') as pf:
    tar_train = pickle.load(pf)

cold_tar_train = tar_train[~tar_train[user_attr].isin(src_tar_cold_users)]
cold_tar_train[user_attr] = cold_tar_train[user_attr].apply(lambda x: 'user_'+x)
cold_tar_train_graph = cold_tar_train.groupby([user_attr, item_attr]).size().reset_index().drop([0], axis=1)
#cold_tar_train_graph.to_csv(os.path.join(save_dir, 'cold_{tar}_train_input.txt'.format(tar=tar)), header=False, index=False, sep='\t')
pd.concat([cold_tar_train_graph, src_train_graph]).to_csv(os.path.join(save_dir, '{src}_{tar}_big_cold.train'.format(tar=tar, src=src)), header=False, index=False, sep=',')





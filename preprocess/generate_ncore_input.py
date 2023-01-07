import pickle
import pandas as pd 
from math import log
import os
import argparse
import random

parser=argparse.ArgumentParser(description='Generated correspond inputs from LOO datas with n core.')
parser.add_argument('--ncore_data_dir', type=str, help='where to save inputs', default=None)
parser.add_argument('--ncore', type=int, help='core number', default=5)
parser.add_argument('--n_testing_user', type=int, default=4000)
parser.add_argument('--src', type=str, help='souce name', default='hk')
parser.add_argument('--tar', type=str, help='target name', default='csjj')
parser.add_argument('--item_attr', type=str, help='(default for amz) attribute represents items\' ids', default='asin')
parser.add_argument('--user_attr', type=str, help='(default for amz) attribute represents users\' ids', default='reviewerID')
args=parser.parse_args()
print(args)

ncore = args.ncore
src, tar = args.src, args.tar
item_attr, user_attr = args.item_attr, args.user_attr

input_save_dir = "{}/input_{}core".format(args.ncore_data_dir, ncore)

if not os.path.isdir(input_save_dir):
        os.mkdir(input_save_dir)

# ------------ SRC-TAR ------------
# tar
with open('{}/LOO_data_{ncore}core/{tar}_train.pickle'.format(args.ncore_data_dir, ncore=ncore, tar=tar), 'rb') as pf:
    tar_train = pickle.load(pf)

tar_train[user_attr] = tar_train[user_attr].apply(lambda x: 'user_'+x)
tar_train_graph = tar_train.groupby([user_attr, item_attr]).size().apply(lambda x: log(x+1.))
_df = pd.concat([tar_train_graph])
_df.to_csv(os.path.join(input_save_dir,'{tar}_tar_train_input.txt'.format(tar=tar)), header=False, sep='\t')

# src (source)
with open('{}/LOO_data_{ncore}core/{src}_train.pickle'.format(args.ncore_data_dir, ncore=ncore, src=src), 'rb') as pf:
    src_train = pickle.load(pf)

src_train[user_attr] = src_train[user_attr].apply(lambda x: 'user_'+x)
src_train_graph = src_train.groupby([user_attr, item_attr]).size().apply(lambda x: log(x+1.))
src_train_graph.to_csv(os.path.join(input_save_dir, '{src}_src_train_input.txt'.format(src=src)), header=False, sep='\t')
#pd.concat([tar_train_graph, src_train_graph]).to_csv(os.path.join(input_save_dir, 'all_cpr_train_u_{src}+{tar}.txt'.format(tar=tar, src=src)), header=False, sep='\t')


# process global test target/shared/cold users

# target
with open('{}/LOO_data_{ncore}core/{tar}_test.pickle'.format(args.ncore_data_dir, ncore=ncore, tar=tar), 'rb') as pf:
    tar_test_df = pickle.load(pf)
all_target_users = tar_test_df[user_attr]
target_users = random.sample(set(all_target_users), args.n_testing_user)
target_users = set(map(lambda x: "user_"+x, target_users))

with open(os.path.join(input_save_dir, '{src}_{tar}_src_tar_test_target_users.pickle'.format(src=src, tar=tar)), 'wb') as pf:
    pickle.dump(target_users, pf)

# shared
with open('{}/user_{ncore}core/{src}_{tar}_shared_users.pickle'.format(args.ncore_data_dir, ncore=ncore, tar=tar, src=src), 'rb') as pf:
    all_shared_users = pickle.load(pf)
shared_users = random.sample(set(all_shared_users), args.n_testing_user)
shared_users = set(map(lambda x: "user_"+x, shared_users))

with open(os.path.join(input_save_dir, '{src}_{tar}_src_tar_test_shared_users.pickle'.format(src=src, tar=tar)), 'wb') as pf:
    pickle.dump(shared_users, pf)

# cold
with open('{}/user_{ncore}core/{src}_{tar}_cold_users.pickle'.format(args.ncore_data_dir, ncore=ncore, tar=tar, src=src), 'rb') as pf:
    cold_users = pickle.load(pf)
cold_users = set(map(lambda x: "user_"+x, cold_users))

with open(os.path.join(input_save_dir, '{src}_{tar}_src_tar_test_cold_users.pickle'.format(src=src, tar=tar)), 'wb') as pf:
    pickle.dump(cold_users, pf)

with open('{}/LOO_data_{ncore}core/{tar}_train.pickle'.format(args.ncore_data_dir, ncore=ncore, tar=tar), 'rb') as pf:
    tar_train = pickle.load(pf)


tar_train[user_attr] = tar_train[user_attr].apply(lambda x: 'user_'+x)
cold_tar_train = tar_train[~tar_train[user_attr].isin(cold_users)]
cold_tar_train_graph = cold_tar_train.groupby([user_attr, item_attr]).size().apply(lambda x: log(x+1.))

cold_tar_train_graph.to_csv(os.path.join(input_save_dir, '{tar}_ctar_train_input.txt'.format(tar=tar)), header=False, sep='\t')

#pd.concat([cold_tar_train_graph, src_train_graph]).to_csv(os.path.join(input_save_dir, 'cold_cpr_train_u_{src}+{tar}.txt'.format(tar=tar, src=src)), header=False, sep='\t')

import pickle
import pandas as pd 
from math import log
import os
import argparse

os.chdir(os.path.dirname(os.path.realpath(__file__)))

parser=argparse.ArgumentParser(description='Generated correspond inputs from LOO datas with n core.')
parser.add_argument('--mom_save_dir', type=str, help='', default=None)
parser.add_argument('--save_dir', type=str, help='where to save inputs', default=None)
parser.add_argument('--ncore', type=int, help='core number', default=5)
parser.add_argument('--src', type=str, help='souce name', default='hk')
parser.add_argument('--tar', type=str, help='target name', default='csjj')
parser.add_argument('--item_attr', type=str, help='(default for amz) attribute represents items\' ids', default='asin')
parser.add_argument('--user_attr', type=str, help='(default for amz) attribute represents users\' ids', default='reviewerID')
args=parser.parse_args()
print(args)


ncore = args.ncore
src, tar = args.src, args.tar
item_attr, user_attr = args.item_attr, args.user_attr

if args.save_dir:
    save_dir = "{}/input_{}core".format(args.save_dir, ncore)
else:
    save_dir = "{}/input_{}core".format(args.mom_save_dir, ncore)

if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

# ------------ SRC-TAR ------------
# tar
with open('{}/LOO_data_{ncore}core/{tar}_train.pickle'.format(args.save_dir, ncore=ncore, tar=tar), 'rb') as pf:
    tar_train = pickle.load(pf)

tar_train[user_attr] = tar_train[user_attr].apply(lambda x: 'user_'+x)
tar_train_graph = tar_train.groupby([user_attr, item_attr]).size().apply(lambda x: log(x+1.))
_df = pd.concat([tar_train_graph])
_df.to_csv(os.path.join(save_dir,'{tar}_train_input.txt'.format(tar=tar)), header=False, sep='\t')

# src (source)
with open('{}/LOO_data_{ncore}core/{src}_train.pickle'.format(args.save_dir, ncore=ncore, src=src), 'rb') as pf:
    src_train = pickle.load(pf)

src_train[user_attr] = src_train[user_attr].apply(lambda x: 'user_'+x)
src_train_graph = src_train.groupby([user_attr, item_attr]).size().apply(lambda x: log(x+1.))
src_train_graph.to_csv(os.path.join(save_dir, 'all_{src}_train_input.txt'.format(src=src)), header=False, sep='\t')
pd.concat([tar_train_graph, src_train_graph]).to_csv(os.path.join(save_dir, 'all_cpr_train_u_{src}+{tar}.txt'.format(tar=tar, src=src)), header=False, sep='\t')

# for cold start
with open('{}/user_{ncore}core/{src}_{tar}_cold_users.pickle'.format(args.save_dir, ncore=ncore, tar=tar, src=src), 'rb') as pf:
    src_tar_cold_users = pickle.load(pf)
with open('{}/LOO_data_{ncore}core/{tar}_train.pickle'.format(args.save_dir, ncore=ncore, tar=tar), 'rb') as pf:
    tar_train = pickle.load(pf)

cold_tar_train = tar_train[~tar_train[user_attr].isin(src_tar_cold_users)]
cold_tar_train[user_attr] = cold_tar_train[user_attr].apply(lambda x: 'user_'+x)
cold_tar_train_graph = cold_tar_train.groupby([user_attr, item_attr]).size().apply(lambda x: log(x+1.))
cold_tar_train_graph.to_csv(os.path.join(save_dir, 'cold_{tar}_train_input.txt'.format(tar=tar)), header=False, sep='\t')
pd.concat([cold_tar_train_graph, src_train_graph]).to_csv(os.path.join(save_dir, 'cold_cpr_train_u_{src}+{tar}.txt'.format(tar=tar, src=src)), header=False, sep='\t')





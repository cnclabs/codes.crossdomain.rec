import pickle
import pandas as pd
from math import log
import os
import random
import argparse

os.chdir(os.path.dirname(os.path.realpath(__file__)))

parser=argparse.ArgumentParser(description='Filtered LOO datas with n core.')
parser.add_argument('--mom_save_dir', type=str, help='',default=None)
parser.add_argument('--save_dir', type=str, help='where to save LOOs', default=None)
parser.add_argument('--user_save_dir', type=str, help='where to save users', default=None)
parser.add_argument('--ncore', type=int, help='core number', default=5)
parser.add_argument('--src', type=str, help='souce name', default='hk')
parser.add_argument('--tar', type=str, help='target name', default='csjj')
parser.add_argument('--item_attr', type=str, help='(default for amz) attribute represents items\' ids', default='asin')
parser.add_argument('--user_attr', type=str, help='(default for amz) attribute represents users\' ids', default='reviewerID')
parser.add_argument('--cold_sample', type=int, help='# cold users selected from shared users', default=4000)
args=parser.parse_args()
print(args)


ncore = args.ncore
src, tar = args.src, args.tar
item_attr, user_attr = args.item_attr, args.user_attr

if args.save_dir:
    save_dir = "{}/LOO_data_{}core".format(args.save_dir, ncore)
else:
    save_dir = "{}/LOO_data_{}core".format(args.mom_save_dir, ncore)

if args.user_save_dir:
    user_save_dir = "{}/user_{}core".format(args.user_save_dir, ncore)
else:
    user_save_dir = "{}/user_{}core".format(args.mom_save_dir, ncore)

if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
if not os.path.isdir(user_save_dir):
        os.mkdir(user_save_dir)


# ----------------- SPO-CSJ -------------
# valid_users = set()
# valid_items = set()

# tar
print("== {src}-{tar} ==".format(src=src.upper(), tar=tar.upper()))
print("Start TARGET / {} & SRC / {} core filtering ...".format(tar.upper(), src.upper()))
with open('{}/LOO_data_0core/{tar}_train.pickle'.format(args.mom_save_dir, tar=tar), 'rb') as f:
    tar_train = pickle.load(f)

def filter(train):
    ncore_train = train[(train[user_attr].isin(train[user_attr].value_counts()[train[user_attr].value_counts()>=ncore].index)) & \
            (train[item_attr].isin(train[item_attr].value_counts()[train[item_attr].value_counts()>=ncore].index))]
    return ncore_train

tar_ncore_train = filter(tar_train)
while (tar_ncore_train[user_attr].value_counts().min()<ncore) | (tar_ncore_train[item_attr].value_counts().min()<ncore):
    tar_ncore_train = filter(tar_ncore_train)

# src
with open('{}/LOO_data_0core/{src}_train.pickle'.format(args.mom_save_dir, src=src), 'rb') as f:
    src_train = pickle.load(f)

src_ncore_train = filter(src_train)
while (src_ncore_train[user_attr].value_counts().min()<ncore) | (src_ncore_train[item_attr].value_counts().min()<ncore):
    src_ncore_train = filter(src_ncore_train)

# valid_users.update(list(tar_ncore_train[user_attr].unique())+list(src_ncore_train[user_attr].unique()))
# valid_items.update(list(tar_ncore_train[item_attr].unique())+list(src_ncore_train[item_attr].unique()))

####

# tar_ncore_train = tar_train[(tar_train[user_attr].isin(valid_users)) & (tar_train[item_attr].isin(valid_items))]
# src_ncore_train = src_train[(src_train[user_attr].isin(valid_users)) & (src_train[item_attr].isin(valid_items))]


with open(os.path.join(save_dir, '{tar}_train.pickle'.format(tar=tar)), 'wb') as f:
    pickle.dump(tar_ncore_train, f)
print("-"*10)
print("(TAR / {tar} Train) Before {ncore}core:".format(tar=tar.upper(), ncore=ncore), len(tar_train))
print("(TAR / {tar} Train) After {ncore}core:".format(tar=tar.upper(), ncore=ncore), len(tar_ncore_train))
print("(TAR / {tar} Train) Diff:".format(tar=tar.upper()), len(tar_train)-len(tar_ncore_train))
print("-"*10)
# tar test
with open('{}/LOO_data_0core/{tar}_test.pickle'.format(args.mom_save_dir, tar=tar), 'rb') as f:
    tar_test = pickle.load(f)
tar_ncore_test = tar_test[(tar_test[user_attr].isin(tar_ncore_train[user_attr].unique())) & (tar_test[item_attr].isin(tar_ncore_train[item_attr].unique()))]
with open(os.path.join(save_dir, '{tar}_test.pickle'.format(tar=tar)), 'wb') as f:
    pickle.dump(tar_ncore_test, f)
print("(TAR / {tar} Test) Before {ncore}core:".format(tar=tar.upper(), ncore=ncore), len(tar_test))
print("(TAR / {tar} Test) After {ncore}core:".format(tar=tar.upper(), ncore=ncore), len(tar_ncore_test))
print("(TAR / {tar} Test) Diff:".format(tar=tar.upper()),len(tar_test)-len(tar_ncore_test))
print("-"*10)
print("Finished TARGET / {} core filtering ...".format(tar.upper()))

with open(os.path.join(save_dir, '{src}_train.pickle'.format(src=src)), 'wb') as f:
    pickle.dump(src_ncore_train, f)
print("-"*10)
print("(SRC / {src} Train) Before {ncore}core:".format(src=src.upper(), ncore=ncore), len(src_train))
print("(SRC / {src} Train) After {ncore}core:".format(src=src.upper(), ncore=ncore), len(src_ncore_train))
print("(SRC / {src} Train) Diff:".format(src=src.upper()), len(src_train)-len(src_ncore_train))
print("-"*10)
# src test
with open('{}/LOO_data_0core/{src}_test.pickle'.format(args.mom_save_dir, src=src), 'rb') as f:
    src_test = pickle.load(f)
src_ncore_test = src_test[(src_test[user_attr].isin(src_ncore_train[user_attr].unique())) & (src_test[item_attr].isin(src_ncore_train[item_attr].unique()))]
with open(os.path.join(save_dir, '{src}_test.pickle'.format(src=src)), 'wb') as f:
    pickle.dump(src_ncore_test, f)
print("(SRC / {src} Test) Before {ncore}core:".format(src=src.upper(), ncore=ncore), len(src_test))
print("(SRC / {src} Test) After {ncore}core:".format(src=src.upper(), ncore=ncore), len(src_ncore_test))
print("(SRC / {src} Test) Diff:".format(src=src.upper()), len(src_test)-len(src_ncore_test))
print("-"*10)
print("Finished SRC / {} ...".format(src.upper()))

# make users
print("Start selecting COLD users from SHARED users ...")
shared_users = set(tar_ncore_train[user_attr]).intersection(set(src_ncore_train[user_attr]))
sample_amount = args.cold_sample
random.seed(3)
try:
    assert len(shared_users)>=sample_amount
except:
    print(len(shared_users))
cold_users = random.sample(shared_users, sample_amount)

# save users
with open(os.path.join(user_save_dir, '{src}_{tar}_shared_users.pickle'.format(src=src, tar=tar)), 'wb') as pf:
    pickle.dump(shared_users, pf)
with open(os.path.join(user_save_dir,'{src}_{tar}_cold_users.pickle'.format(src=src, tar=tar)), 'wb') as pf:
    pickle.dump(cold_users, pf)
print("Finished Users ... # Cold sample from shared: {} & # of shared: {}".format(sample_amount, len(shared_users)))


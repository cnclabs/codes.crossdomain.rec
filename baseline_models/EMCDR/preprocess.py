import argparse
import numpy as np
import pandas as pd
import pickle

parser=argparse.ArgumentParser(description='EMCDR')
parser.add_argument('--dataset_name', type=str, help='{tv_vod, csj_hk, mt_books}')
args=parser.parse_args()

with open('../../user/' + args.dataset_name + '_' + 'shared_users.pickle', 'rb') as pf:
    shared_users = pickle.load(pf)

source_name = args.dataset_name.split('_')[0]
target_name = args.dataset_name.split('_')[1]

target_user_array = []
target_item_array = []
target_item_list = []

with open('../BPR/graph/'+ target_name + '_' + 'lightfm_bpr_10e-5.txt', 'r') as f:
    skip_f = f.readlines()[1:]
    for line in skip_f:
        line = line[:-1]
        prefix = line.split(' ')[0]
        # ignore first two elements
        emb=line.split(' ')[3:]
        if prefix in shared_users:
            target_user_array.append(np.array(emb, dtype=np.float32))
        if 'user_' not in prefix:
            target_item_list.append(prefix)
            target_item_array.append(np.array(emb, dtype=np.float32))

source_user_array = []
source_item_array = []

with open('../BPR/graph/' + source_name + '_' + 'lightfm_bpr_10e-5.txt', 'r') as f:
    skip_f = f.readlines()[1:]
    for line in skip_f:
        line = line[:-1]
        prefix = line.split(' ')[0]
        # ignore first two elements
        emb=line.split(' ')[3:]
        if prefix in shared_users:
            source_user_array.append(np.array(emb, dtype=np.float32))
        if 'user_' not in prefix:
            source_item_array.append(np.array(emb, dtype=np.float32))


Us = np.array(source_user_array).T
print("Us shape = {}".format(Us.shape))
Vs = np.array(source_item_array).T
print("Vs shape = {}".format(Vs.shape))
Ut = np.array(target_user_array).T
print("Ut shape = {}".format(Ut.shape))
Vt = np.array(target_item_array).T
print("Vt shape = {}".format(Vt.shape))

with open('./' + args.dataset_name + '/' + 'lightfm_bpr_Us.pickle', 'wb') as pf:
    pickle.dump(Us, pf)
with open('./' + args.dataset_name + '/' + 'lightfm_bpr_Vs.pickle', 'wb') as pf:
    pickle.dump(Vs, pf)
with open('./' + args.dataset_name + '/' + 'lightfm_bpr_Ut.pickle', 'wb') as pf:
    pickle.dump(Ut, pf)
with open('./' + args.dataset_name + '/' + 'lightfm_bpr_Vt.pickle', 'wb') as pf:
    pickle.dump(Vt, pf)

with open('./' + args.dataset_name + '/' + target_name + '_' + 'items.pickle', 'wb') as pf:
    pickle.dump(target_item_list, pf)

















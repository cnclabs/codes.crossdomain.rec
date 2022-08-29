import argparse
import numpy as np
import pandas as pd
import pickle

parser=argparse.ArgumentParser(description='EMCDR')
parser.add_argument('--current_epoch', type=str)
parser.add_argument('--dataset_name', type=str, help='{tv_vod, csj_hk, mt_books, el_cpa, spo_csj}')
parser.add_argument('--ncore', type=int, help='core_filter', default=0)
args=parser.parse_args()

source_name = args.dataset_name.split('_')[0]
target_name = args.dataset_name.split('_')[1]
ncore = args.ncore

with open('../../../user_{}core/'.format(ncore) + args.dataset_name + '_' + 'shared_users.pickle', 'rb') as pf:
    shared_users = pickle.load(pf)
shared_users = ["user_"+user for user in shared_users]
print("shared users: ", shared_users[:10])

target_user_array = []
target_item_array = []
target_item_list = []

with open('../lfm_bpr_graphs/'+ target_name + '_' + f'lightfm_bpr_{args.current_epoch}_10e-5.txt', 'r') as f:
    # print("prefix: ")
    for line in f:
        line = line.strip("\n")
        prefix, emb = line.split('\t')
        prefix = prefix.replace(" ", "")
        # print(prefix)
        emb = emb.split()
        if prefix in shared_users:
            target_user_array.append(np.array(emb, dtype=np.float32))
        if 'user_' not in prefix:
            target_item_list.append(prefix)
            target_item_array.append(np.array(emb, dtype=np.float32))

source_user_array = []
source_item_array = []

with open('../lfm_bpr_graphs/' + source_name + '_' + f'lightfm_bpr_{args.current_epoch}_10e-5.txt', 'r') as f:
    for line in f:
        line = line.strip("\n")
        prefix, emb = line.split('\t')
        prefix = prefix.replace(" ", "")
        emb = emb.split()
        # print("prefix: ", prefix)
        # print("shared_users: ", shared_users)
        if prefix in shared_users:
            source_user_array.append(np.array(emb, dtype=np.float32))
        if 'user_' not in prefix:
            source_item_array.append(np.array(emb, dtype=np.float32))

print("source: ", source_name)
print("target:", target_name)
Us = np.array(source_user_array).T
print("Us shape = {}".format(Us.shape))
Vs = np.array(source_item_array).T
print("Vs shape = {}".format(Vs.shape))
Ut = np.array(target_user_array).T
print("Ut shape = {}".format(Ut.shape))
Vt = np.array(target_item_array).T
print("Vt shape = {}".format(Vt.shape))

with open('./' + args.dataset_name + '/' + f'lightfm_bpr_Us_{args.current_epoch}.pickle', 'wb') as pf:
    pickle.dump(Us, pf)
with open('./' + args.dataset_name + '/' + f'lightfm_bpr_Vs_{args.current_epoch}.pickle', 'wb') as pf:
    pickle.dump(Vs, pf)
with open('./' + args.dataset_name + '/' + f'lightfm_bpr_Ut_{args.current_epoch}.pickle', 'wb') as pf:
    pickle.dump(Ut, pf)
with open('./' + args.dataset_name + '/' + f'lightfm_bpr_Vt_{args.current_epoch}.pickle', 'wb') as pf:
    pickle.dump(Vt, pf)

with open('./' + args.dataset_name + '/' + target_name + '_' + f'items_{args.current_epoch}.pickle', 'wb') as pf:
    pickle.dump(target_item_list, pf)

















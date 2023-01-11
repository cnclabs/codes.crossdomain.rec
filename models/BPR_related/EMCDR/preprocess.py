import argparse
import numpy as np
import pandas as pd
import pickle

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='EMCDR')
    parser.add_argument('--mom_save_dir', type=str, help='')
    parser.add_argument('--current_epoch', type=str)
    parser.add_argument('--dataset_name', type=str, help='{tv_vod, csj_hk, mt_books, el_cpa, spo_csj}')
    parser.add_argument('--ncore', type=int, help='core_filter', default=0)
    args=parser.parse_args()
    
    source_name = args.dataset_name.split('_')[0]
    target_name = args.dataset_name.split('_')[1]
    ncore = args.ncore
    
    share_user_path = .pickle
    with open(share_user_path, 'rb') as pf:
        shared_users = pickle.load(pf)
    print("shared users: ", shared_users[:10])
    
    cold_user_path = .pickle
    if cold:
        with open(cold_user_path, 'rb') as pf:
            cold_users = pickle.load(pf)
        
        to_map_users = set(shared_users) - set(cold_users)
    else:
        to_map_users = set(shared_users)
    
        print(f"num of shared users: {len(shared_users)}, num of cold users: {len(cold_users)}, num of to_map_users: {len(to_map_users)}")
    
    target_user_array = []
    target_item_array = []
    target_item_list = []
    
    target_emb_path = emb.txt
    with open(target_emb_path, 'r') as f:
        for line in f:
            line = line.strip("\n")
            prefix, emb = line.split('\t')
            prefix = prefix.replace(" ", "")
            emb = emb.split()
            if prefix in shared_users:
                target_user_array.append(np.array(emb, dtype=np.float32))
            if 'user_' not in prefix:
                target_item_list.append(prefix)
                target_item_array.append(np.array(emb, dtype=np.float32))
    
    source_user_array = []
    source_item_array = []
    
    source_emb_path = emb.txt
    with open(source_emb_path, 'r') as f:
        for line in f:
            line = line.strip("\n")
            prefix, emb = line.split('\t')
            prefix = prefix.replace(" ", "")
            emb = emb.split()
            if prefix in to_map_users:
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
    
    print("Saving out...")
    Us_save_path =  .pickle
    Vs_save_path = 
    Ut_save_path = 
    Vt_save_path = 
    target_item_list_save_path = .pickle
    
    with open(Us_save_path, 'wb') as pf:
        pickle.dump(Us, pf)
    with open(Vs_save_path, 'wb') as pf:
        pickle.dump(Vs, pf)
    with open(Ut_save_path, 'wb') as pf:
        pickle.dump(Ut, pf)
    with open(Vt_save_path, 'wb') as pf:
        pickle.dump(Vt, pf)
    with open(target_item_list_save_path, 'wb') as pf:
        pickle.dump(target_item_list, pf)
    print("Done!")

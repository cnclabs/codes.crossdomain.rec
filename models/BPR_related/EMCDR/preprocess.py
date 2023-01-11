import argparse
import numpy as np
import pandas as pd
import pickle

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='EMCDR')
    parser.add_argument('--share_user_path', type=str, required=True)
    parser.add_argument('--cold_user_path', type=str, default=None)
    parser.add_argument('--pretrained_source_emb_path', type=str, required=True)
    parser.add_argument('--pretrained_target_emb_path', type=str, required=True)
    parser.add_argument('--Us_save_path', type=str, required=True)
    parser.add_argument('--Vs_save_path', type=str, required=True)
    parser.add_argument('--Ut_save_path', type=str, required=True)
    parser.add_argument('--Vt_save_path', type=str, required=True)
    parser.add_argument('--target_item_list_save_path', type=str, required=True)
    args=parser.parse_args()
    
    with open(args.share_user_path, 'rb') as pf:
        shared_users = pickle.load(pf)

    if args.cold_user_path is not None:
        print("Processing cold mode...")
        with open(args.cold_user_path, 'rb') as pf:
            cold_users = pickle.load(pf)
        to_map_users = set(shared_users) - set(cold_users)
    else:
        print("Processing normal mode...")
        to_map_users = set(shared_users)
    
    print(f"Num of shared users: {len(shared_users)}")
    print(f"Num of to_map_users: {len(to_map_users)}")
    
    source_user_array = []
    source_item_array = []
    with open(args.pretrained_source_emb_path, 'r') as f:
        for line in f:
            line = line.strip("\n")
            prefix, emb = line.split('\t')
            prefix = prefix.replace(" ", "")
            emb = emb.split()
            if prefix in to_map_users:
                source_user_array.append(np.array(emb, dtype=np.float32))
            if 'user_' not in prefix:
                source_item_array.append(np.array(emb, dtype=np.float32))

    target_user_array = []
    target_item_array = []
    target_item_list = []
    with open(args.pretrained_target_emb_path, 'r') as f:
        for line in f:
            line = line.strip("\n")
            prefix, emb = line.split('\t')
            prefix = prefix.replace(" ", "")
            emb = emb.split()
            if prefix in to_map_users:
                target_user_array.append(np.array(emb, dtype=np.float32))
            if 'user_' not in prefix:
                target_item_list.append(prefix)
                target_item_array.append(np.array(emb, dtype=np.float32))
    
    Us = np.array(source_user_array).T
    print("Us shape = {}".format(Us.shape))
    Vs = np.array(source_item_array).T
    print("Vs shape = {}".format(Vs.shape))
    Ut = np.array(target_user_array).T
    print("Ut shape = {}".format(Ut.shape))
    Vt = np.array(target_item_array).T
    print("Vt shape = {}".format(Vt.shape))
    
    print("Saving out...")
    with open(args.Us_save_path, 'wb') as pf:
        pickle.dump(Us, pf)
    with open(args.Vs_save_path, 'wb') as pf:
        pickle.dump(Vs, pf)
    with open(args.Ut_save_path, 'wb') as pf:
        pickle.dump(Ut, pf)
    with open(args.Vt_save_path, 'wb') as pf:
        pickle.dump(Vt, pf)
    with open(args.target_item_list_save_path, 'wb') as pf:
        pickle.dump(target_item_list, pf)
    print("Done!")

import argparse
import numpy as np
import pandas as pd
import pickle
import os

def generate_id_map(user_name_list):
    id_map = {}
    for remap_id, org_id in enumerate(user_name_list):
        id_map[org_id] = remap_id
    
    return id_map

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='EMCDR')
    parser.add_argument('--share_user_path', type=str, required=True)
    parser.add_argument('--cold_user_path', type=str, default=None)
    parser.add_argument('--pretrained_source_emb_path', type=str, required=True)
    parser.add_argument('--pretrained_target_emb_path', type=str, required=True)
    parser.add_argument('--Us_save_path', type=str, required=True)
    parser.add_argument('--Us_id_map_save_path', type=str, required=True)
    parser.add_argument('--Ut_save_path', type=str, required=True)
    parser.add_argument('--cold_Us_save_path', type=str, default=None)
    parser.add_argument('--cold_Us_id_map_save_path', type=str, default=None)
    args=parser.parse_args()
    
    with open(args.share_user_path, 'rb') as pf:
        shared_users = pickle.load(pf)

    if args.cold_user_path is not None:
        assert args.cold_Us_save_path is not None, "cold Us should be processsed together."
        assert args.cold_Us_id_map_save_path is not None, "cold Us should be processsed together."
        print("Processing cold mode...")
        with open(args.cold_user_path, 'rb') as pf:
            cold_users = pickle.load(pf)
        to_map_users = set(shared_users) - set(cold_users)

        #TODO (katiyth) this was done in infer_Us_cold.py from previous people code 
        # we temporarily move it to here as a data preprocessing part.for procedure consistency
        cold_users_array = []
        cold_user_name_list = []
        with open(args.pretrained_source_emb_path, 'r') as f:
            for line in f:
                line = line.strip("\n")
                prefix, emb = line.split('\t')
                prefix = prefix.replace(" ", "")
                emb = emb.split()
                if prefix in cold_users:
                    cold_users_array.append(np.array(emb, dtype=np.float32))
                    cold_user_name_list.append(prefix)
        cold_Us = np.array(cold_users_array).T
        cold_Us_id_map = generate_id_map(cold_user_name_list)
        with open(args.cold_Us_save_path, 'wb') as pf:
            pickle.dump(cold_Us, pf)
        with open(args.cold_Us_id_map_save_path, 'wb') as pf:
            pickle.dump(cold_Us_id_map, pf)
    else:
        print("Processing normal mode...")
        to_map_users = set(shared_users)
    
    print(f"Num of shared users: {len(shared_users)}")
    print(f"Num of to_map_users: {len(to_map_users)}")
    
    source_user_array = []
    source_user_name_list = []
    with open(args.pretrained_source_emb_path, 'r') as f:
        for line in f:
            line = line.strip("\n")
            prefix, emb = line.split('\t')
            prefix = prefix.replace(" ", "")
            emb = emb.split()
            if prefix in to_map_users:
                source_user_array.append(np.array(emb, dtype=np.float32))
                source_user_name_list.append(prefix)

    target_user_array = []
    target_user_name_list = []
    with open(args.pretrained_target_emb_path, 'r') as f:
        for line in f:
            line = line.strip("\n")
            prefix, emb = line.split('\t')
            prefix = prefix.replace(" ", "")
            emb = emb.split()
            if prefix in to_map_users:
                target_user_array.append(np.array(emb, dtype=np.float32))
                target_user_name_list.append(prefix)

    assert source_user_name_list == target_user_name_list, 'The order of user name is not equal.'
    Us_id_map = generate_id_map(source_user_name_list)
      
    Us = np.array(source_user_array).T
    print("Us shape = {}".format(Us.shape))
    Ut = np.array(target_user_array).T
    print("Ut shape = {}".format(Ut.shape))
    
    for _path in [args.Us_save_path, args.Ut_save_path]:
        _dir = os.path.dirname(_path)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
    print("Saving out...")
    with open(args.Us_save_path, 'wb') as pf:
        pickle.dump(Us, pf)
    with open(args.Ut_save_path, 'wb') as pf:
        pickle.dump(Ut, pf)
    with open(args.Us_id_map_save_path, 'wb') as pf:
        pickle.dump(Us_id_map, pf)

    print("Done!")

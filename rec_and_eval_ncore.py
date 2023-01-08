import argparse
import numpy as np
import pickle
import os

from evaluation.utility import (save_exp_record,
        rank_and_score,
        generate_item_graph_df,
        generate_user_emb,
        load_testing_users_neg99_pos1)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Calculate the similarity and recommend VOD items')
    parser.add_argument('--data_dir', type=str, help='groundtruth files dir')
    parser.add_argument('--save_dir', type=str, help='dir to save cav')
    parser.add_argument('--save_name', type=str, help='name to save csv')
    parser.add_argument('--user_emb_path', type=str)
    parser.add_argument('--user_emb_path_shared', type=str)
    parser.add_argument('--user_emb_path_target', type=str)
    parser.add_argument('--user_emb_path_cold', type=str)
    parser.add_argument('--item_emb_path', type=str)
    parser.add_argument('--test_mode', type=str, help='{target, shared, cold}')
    parser.add_argument('--ncore', type=int, help='core number', default=5)
    parser.add_argument('--src', type=str, help='souce name')
    parser.add_argument('--tar', type=str, help='target name')
    parser.add_argument('--model_name', type=str, help='cpr, lgn, lgn_s, bpr, bpr_s, emcdr, bitgcf')
    parser.add_argument('--uid_i', type=str, help='(default for amz) unique id column for item', default='asin')
    parser.add_argument('--uid_u', type=str, help='(default for amz) unique id column of user', default='reviewerID')
    parser.add_argument('--top_ks', nargs='*', help='top_k to eval', default=[1, 3, 5, 10, 20], action='extend', type=int)
    
    args=parser.parse_args()
    print(args)
    
    save_name = args.save_name
    ncore = args.ncore
    src, tar = args.src, args.tar
    uid_u, uid_i = args.uid_u, args.uid_i
    test_mode = args.test_mode
    model_name = args.model_name
    dataset_pair = f"{src}_{tar}"
    top_ks = args.top_ks
    save_dir=args.save_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    tar_test_path  = '{}/{tar}_tar_test.pickle'.format(args.data_dir, ncore=ncore, tar=tar) 
    with open(tar_test_path, 'rb') as pf:
        tar_test_df = pickle.load(pf)
    #
    testing_users_neg99_pos1 = load_testing_users_neg99_pos1(args.data_dir, test_mode, src, tar)
    testing_users = list(testing_users_neg99_pos1.keys())
    
    if model_name == 'emcdr':
        if args.test_mode == 'target':
            ## source 1 : testing users are from shared users
            with open(args.user_emb_path_shared, 'rb') as pf:
                shared_users_mapped_emb_dict = pickle.load(pf)
            ## source 2 : testing users are from target domain only users
            target_users_emb_dict = {}
            with open(args.user_emb_path_target, 'r') as f:
                for line in f:
                    line = line.split('\t')
                    prefix = line[0]
                    prefix = prefix.replace(" ", "")
                    emb=line[1].split()  
                    if 'user_' in prefix:
                        target_users_emb_dict[prefix] = np.array(emb, dtype=np.float32)
        
            user_emb = {}
            for user in testing_users:
                if user in shared_users_mapped_emb_dict.keys():
                    user_emb[user] = shared_users_mapped_emb_dict[user]
                else:
                    user_emb[user] = target_users_emb_dict[user]
        
        if args.test_mode == 'shared':
            with open(args.user_emb_path_shared, 'rb') as pf:
                    shared_users_mapped_emb_dict = pickle.load(pf)
            user_emb = {k:v for k,v in shared_users_mapped_emb_dict.items() if k in testing_users}
        
        if args.test_mode == 'cold':
            with open(args.user_emb_path_cold, 'rb') as pf:
                user_emb = pickle.load(pf)
    
        print("Start getting embedding for each user and item...")
        item_graph_df = generate_item_graph_df(args.item_emb_path)
        print("Got embedding!")
    else:
        print("Start getting embedding for each user and item...")
        user_emb = generate_user_emb(args.user_emb_path)
        item_graph_df = generate_item_graph_df(args.item_emb_path)
        print("Got embedding!")
    
    total_rec, total_ndcg, count = rank_and_score(testing_users, top_ks, user_emb, testing_users_neg99_pos1, item_graph_df, tar_test_df, uid_u, uid_i)
    save_exp_record(model_name, dataset_pair, test_mode, top_ks, total_rec, total_ndcg, count, save_dir, save_name)

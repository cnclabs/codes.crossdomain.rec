import pandas as pd
import os
import argparse
import pickle
import random
import multiprocessing
from multiprocessing import Pool
import numpy as np
import sys

# def
all_dataset = ["tvvod", "vodtv", "csjhk", "hkcsjj", "mtb", "elcpa", "cpael", "spocsj"]
sample_amount = 1000
cpr_path = "/tmp2/yzliu/CPR_paper" 
sys.setrecursionlimit(100000000)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Available dataset: "+str(all_dataset))
args = parser.parse_args()

dataset = args.dataset

if dataset not in all_dataset:
    print("\n\nDo nothing. Dataset",dataset,"is not available")
else:
    print("\n\n**** Dataset:", dataset, "****")
    # for multiprocessing
    cpu_amount = multiprocessing.cpu_count()
    worker = cpu_amount - 2
    mp = Pool(worker)

    def process_user_rec_dict(user_list):
        user_rec_dict = {}
        for user in user_list:
            if '_' in user:
                user = user[user.find('_')+1:]
            """
            if dataset == "tvvod" or dataset == "vodtv":
                watched_set = set(vod_train_df[vod_train_df['user_id'] == user].item_id)
            else:
                watched_set = set(vod_train_df[vod_train_df['reviewerID'] == user].asin)
            neg_pool = total_item_set - watched_set
            random.seed(5)
            neg_99 = random.sample(neg_pool, 99)
            """
            neg_99 = []
            if dataset == "tvvod" or dataset == "vodtv":
                user_rec_pool = list(neg_99) + list(vod_test_df[vod_test_df['user_id'] == user].item_id)
            else:
                user_rec_pool = list(neg_99) + list(vod_test_df[vod_test_df['reviewerID'] == user].asin)
            user_rec_dict[user] = user_rec_pool
        return user_rec_dict
    
#############################################
############################# Preparing State
#############################################

    if dataset == "tvvod" or dataset == "vodtv":
        tar_name = "vod"
        src_name = "tv"
        sample_amount = 2000
    elif dataset == "vodtv":
        tar_name = "tv"
        src_name = "vod"
        sample_amount = 2000
    elif dataset == "csjhk":
        tar_name = "hk"
        src_name = "csj"
        sample_amount = 4000
    elif dataset == "hkcsjj":
        tar_name = "csjj"
        src_name = "hk"
        sample_amount = 4000
    elif dataset == "mtb":
        tar_name = "books"
        src_name = "mt"
        sample_amount = 1000
    elif dataset == "elcpa":
        tar_name = "cpa"
        src_name = "el"
        sample_amount = 8000
    elif dataset == "cpael":
        tar_name = "el"
        src_name = "cpa"
        sample_amount = 8000
    elif dataset == "spocsj":
        tar_name = "csj"
        src_name = "spo"
        sample_amount = 8000

    with open(os.path.join(cpr_path, 'LOO_data_sampled/'+tar_name+'_test.pickle'), 'rb') as pf:
        vod_test_df = pickle.load(pf)
    with open(os.path.join(cpr_path, 'LOO_data_sampled/'+src_name+'_test.pickle'), 'rb') as pf:
        tv_test_df = pickle.load(pf)
    with open(os.path.join(cpr_path, 'LOO_data_sampled/'+tar_name+'_train.pickle'), 'rb') as pf:
        vod_train_df = pickle.load(pf)
    with open(os.path.join(cpr_path, 'LOO_data_sampled/'+src_name+'_train.pickle'), 'rb') as pf:
        tv_train_df = pickle.load(pf)
    if dataset == "tvvod" or dataset == "vodtv":
        total_item_set = set(vod_train_df.item_id)
    else:
        total_item_set = set(vod_train_df.asin)
    if dataset == "tvvod" or dataset == "vodtv":
        tv_total_item_set = set(vod_train_df.item_id)
    else:
        tv_total_item_set = set(vod_train_df.asin)

    # make target_test.txt
    print("== Start preparing data for target_test.txt ==")
    if dataset == "tvvod" or dataset == "vodtv":
        testing_users = random.sample(set(vod_test_df.user_id), sample_amount)
    else:
        testing_users = random.sample(set(vod_test_df.reviewerID), sample_amount)
    
    mp = Pool(worker)
    split_datas = np.array_split(list(testing_users), worker)
    results = mp.map(process_user_rec_dict ,split_datas)
    mp.close()
    target_user_rec_dict = {}
    for r in results:
          target_user_rec_dict.update(r)

    print("user rec dict generated!")

    # make shared_test.txt
    print("== Start preparing data for shared_test.txt ==")
    with open(os.path.join(cpr_path, 'user_sampled/'+src_name+'_'+tar_name+'_shared_users.pickle'), 'rb') as pf:
        shared_users = pickle.load(pf)
    testing_users = random.sample(set(shared_users), sample_amount)
    
    mp = Pool(worker)
    split_datas = np.array_split(list(testing_users), worker)
    results = mp.map(process_user_rec_dict ,split_datas)
    mp.close()
    shared_user_rec_dict = {}
    for r in results:
          shared_user_rec_dict.update(r)

    print("user rec dict generated!")

    #print("== Start preparing data for cold_tar_train.txt ==")
    #cold_tar_ui = [set() for i in range(len(user_remap_dic))]
    #with open(os.path.join(cpr_path, "input_0/cold_"+tar_name+"_train_input.txt"), 'r') as f:
    #    print("adding ut")
    #    raw = list(map(lambda x: x.strip('\n').split('\t'), f.readlines()))
    #    for e in raw:
    #        remap_user_id = user_remap_dic[e[0][e[0].find('_')+1:]]
    #        if dataset == "tvvod" or dataset == "vodtv":
    #            remap_item_id = item_remap_dic[e[1][e[1].find('_')+1:]]
    #        else:
    #            remap_item_id = item_remap_dic[e[1]]
    #        cold_tar_ui[int(remap_user_id)].add(remap_item_id)

    # make cold_test.txt
    print("== Start preparing data for cold_test.txt ==")
    with open(os.path.join(cpr_path, 'user_sampled/'+src_name+'_'+tar_name+'_cold_users.pickle'), 'rb') as pf:
        cold_users = pickle.load(pf)
    
    testing_users = cold_users # already remain the sample amount of users
    
    mp = Pool(worker)
    split_datas = np.array_split(list(testing_users), worker)
    results = mp.map(process_user_rec_dict ,split_datas)
    mp.close()
    cold_user_rec_dict = {}
    for r in results:
          cold_user_rec_dict.update(r)
    print("user rec dict generated!")

###########################################
############################# Writing State
###########################################
    # write train.txt
    print("== Writing State ==")
    # write target_test.txt
    with open('./'+dataset+'/'+dataset+'_big_target.test', 'w') as f:
        for uid in target_user_rec_dict:
            for iid in target_user_rec_dict[uid]:
                f.write(uid)
                f.write(','+iid)
                f.write('\n')
    print("Finishing target_test.txt")

    #print(random.choice(list(shared_user_rec_dict.items())))
    with open('./'+dataset+'/'+dataset+'_big_shared.test', 'w') as f:
        for uid in shared_user_rec_dict:
            for iid in shared_user_rec_dict[uid]:
                f.write(str(uid))
                f.write(','+str(iid))
                f.write('\n')
    print("Finishing shared_test.txt")
    #print(random.choice(list(cold_user_rec_dict.items())))
    with open('./'+dataset+'/'+dataset+'_big_cold.test', 'w') as f:
        for uid in cold_user_rec_dict:
            for iid in cold_user_rec_dict[uid]:
                f.write(uid)
                f.write(','+iid)
                f.write('\n')
    print("Finishing cold_test.txt")

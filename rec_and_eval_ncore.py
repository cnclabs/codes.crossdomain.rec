import time
import argparse
import faiss
import numpy as np
import json
import pandas as pd
import pickle
import random
import re
import os
import uuid
import multiprocessing

from evaluation.utility import save_exp_record, rank_and_score, generate_item_graph_df, generate_user_emb, get_testing_users, get_testing_users_rec_dict

os.chdir(os.path.dirname(os.path.realpath(__file__)))

parser=argparse.ArgumentParser(description='Calculate the similarity and recommend VOD items')
parser.add_argument('--data_dir', type=str, help='groundtruth files dir')
parser.add_argument('--save_dir', type=str, help='dir to save cav')
parser.add_argument('--save_name', type=str, help='name to save csv')
parser.add_argument('--output_file', type=str, help='output_file name')
parser.add_argument('--graph_file', type=str, help='graph_file')
parser.add_argument('--test_mode', type=str, help='{target, shared, cold}')
parser.add_argument('--ncore', type=int, help='core number', default=5)
parser.add_argument('--seed', type=int, help='random seed', default=3)
parser.add_argument('--n_worker', type=int, help='number of workers', default=None)
parser.add_argument('--src', type=str, help='souce name')
parser.add_argument('--tar', type=str, help='target name')
parser.add_argument('--model_name', type=str, help='cpr, lgn, lgn_s, bpr, bpr_s, emcdr, bitgcf')
parser.add_argument('--uid_i', type=str, help='(default for amz) unique id column for item', default='asin')
parser.add_argument('--uid_u', type=str, help='(default for amz) unique id column of user', default='reviewerID')
parser.add_argument('--top_ks', nargs='*', help='top_k to eval', default=[1, 3, 5, 10, 20], action='extend', type=int)

args=parser.parse_args()
print(args)

save_name = args.save_name
output_file = args.output_file
graph_file = args.graph_file
ncore = args.ncore
src, tar = args.src, args.tar
uid_u, uid_i = args.uid_u, args.uid_i
test_mode = args.test_mode
save_dir=args.save_dir

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if args.n_worker is None:
    cpu_amount = multiprocessing.cpu_count()
    n_worker = cpu_amount - 2
else:
    n_worker = args.n_worker

random.seed(args.seed)

data_input_dir = os.path.join(args.data_dir, f'input_{ncore}core')
testing_users = get_testing_users(test_mode, data_input_dir, src, tar)

# TODO:((katieyth): 
tar_test_path  = '{}/LOO_data_{ncore}core/{tar}_test.pickle'.format(args.data_dir, ncore=ncore, tar=tar) 
with open(tar_test_path, 'rb') as pf:
    tar_test_df = pickle.load(pf)
tar_test_df[uid_u]  = tar_test_df[uid_u].apply(lambda x: 'user_'+x)

tar_train_path = f'{args.data_dir}/input_{ncore}core/{tar}_train_input.txt'
tar_train_df = pd.read_csv(tar_train_path, sep='\t', header=None, names=[uid_u, uid_i, 'xxx'])

total_item_set = set(tar_train_df[uid_i])

testing_users_rec_dict = get_testing_users_rec_dict(n_worker, testing_users, tar_train_df, tar_test_df, uid_u, uid_i, total_item_set)

print("Start getting embedding for each user and item...")
user_emb = generate_user_emb(graph_file)
item_graph_df= generate_item_graph_df(graph_file)
print("Got embedding!")

top_ks = args.top_ks
total_rec, total_ndcg, count = rank_and_score(testing_users, top_ks, user_emb, testing_users_rec_dict, item_graph_df, tar_test_df, n_worker, uid_u, uid_i)

model_name = args.model_name
dataset_pair = f"{src}_{tar}"
test_mode=args.test_mode
save_exp_record(model_name, dataset_pair, test_mode, top_ks, total_rec, total_ndcg, count, save_dir, save_name, output_file)
    

import pickle
import argparse
import os

from evaluation.utility import get_testing_users_rec_dict

parser=argparse.ArgumentParser(description='Calculate the similarity and recommend VOD items')
parser.add_argument('--data_dir', type=str, help='groundtruth files dir')
parser.add_argument('--test_mode', type=str, help='{target, shared, cold}')
parser.add_argument('--ncore', type=int, help='core number', default=5)
parser.add_argument('--n_worker', type=int, help='number of workers', default=None)
parser.add_argument('--src', type=str, help='souce name')
parser.add_argument('--tar', type=str, help='target name')
parser.add_argument('--uid_i', type=str, help='(default for amz) unique id column for item', default='asin')
parser.add_argument('--uid_u', type=str, help='(default for amz) unique id column of user', default='reviewerID')
args=parser.parse_args()

tar_test_path  = '{}/LOO_data_{ncore}core/{tar}_test.pickle'.format(args.data_dir, ncore=args.ncore, tar=args.tar) 
with open(tar_test_path, 'rb') as pf:
    tar_test_df = pickle.load(pf)
tar_test_df[args.uid_u]  = tar_test_df[args.uid_u].apply(lambda x: 'user_'+x)

data_input_dir = os.path.join(args.data_dir, f'input_{args.ncore}core')
testing_users_rec_dict = get_testing_users_rec_dict(args.n_worker, tar_test_df, args.uid_u, args.uid_i, args.test_mode, data_input_dir, args.src, args.tar)

save_path = os.path.join(data_input_dir, f'testing_users_rec_dict_{args.src}_{args.tar}_{args.test_mode}.pickle')
with open(save_path, 'wb') as f:
    pickle.dump(testing_users_rec_dict, f)

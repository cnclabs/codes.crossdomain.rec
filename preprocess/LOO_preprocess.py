import pickle
import pandas as pd
import json
from datetime import datetime
import multiprocessing 
from multiprocessing import Pool
import numpy as np
import os
import argparse

os.chdir(os.path.dirname(os.path.realpath(__file__)))

parser=argparse.ArgumentParser(description='Raw data -> LOO data. Please determine the name of the raw data to be processed.')
parser.add_argument('--raw_data', type=str, help='name of raw data to be processed', default=None)
parser.add_argument('--dataset_name', type=str, help="name to be represent this dataset in the future", default=None)
parser.add_argument('--user_attr', type=str, help='(default for amz) attribute represents users\' ids', default='reviewerID')
parser.add_argument('--time_attr', type=str, help='(default for amz) attribute represents time', default='unixReviewTime')
args=parser.parse_args()
print(args)

raw_data, dataset_name = args.raw_data, args.dataset_name
assert raw_data is not None, "name of raw data can't be empty"
assert dataset_name is not None, "dataset name can't be empty"

time_attr, user_attr = args.time_attr, args.user_attr

print('>>>>>Reading raw data...')
jf = open(os.path.join('../raw_data', raw_data), 'r')
line_list = []
for line in jf.readlines():
    dic = json.loads(line)
    line_list.append(dic)
print('>>>>>Done reading raw data.')

dataset = pd.DataFrame(line_list)
print('>>>>>Applying datetime...')
dataset['date'] = dataset[time_attr].apply(lambda x:\
                                             datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
print('>>>>>Done applying datetime...')

# align with h_k, so the latest date of the data is 2018-10-04
print('>>>>>Subsetting...')
dataset_2 = dataset[(dataset['date'] >= '2016-10-05 00:00:00') & (dataset['date'] <= '2018-10-04 00:00:00')]
print('>>>>>Doen Subsetting...')

def process_train_test(user_list):
    dataset_one_log_user = []
    dataset_train_data = []
    dataset_test_data = []

    for user in user_list:

        s_user_dataset_2 = dataset_2[dataset_2[user_attr] == user]\
                                    .sort_values('date')

        user_test_log = s_user_dataset_2.tail(1)
        user_train_log = s_user_dataset_2.iloc[:(len(s_user_dataset_2)-1)]
        
        if len((dataset_2[dataset_2[user_attr] == user])) == 1:
            dataset_one_log_user.append(user)
            user_train_log = s_user_dataset_2.tail(1)

        dataset_train_data.append(user_train_log)
        dataset_test_data.append(user_test_log)
    
    return [dataset_one_log_user, dataset_train_data, dataset_test_data]


total_users = dataset_2.reviewerID.unique()
print("total user amount = {}".format(len(total_users)))

cpu_amount = multiprocessing.cpu_count() - 10
print(f'>>>>>Multiprocessing... with cpu_amount {cpu_amount}')
mp = Pool(cpu_amount)
split_datas = np.array_split(list(total_users), cpu_amount)
results = mp.map(process_train_test ,split_datas)
mp.close()
print(f'>>>>>DONE Multiprocessing... ')

print(f'>>>>>Gathering result...')
one_log_user_list = []
train_data = []
test_data = []

for r in results:
    one_log_user_list += r[0]
    train_data += r[1]
    test_data += r[2]


dataset_train_df = pd.concat(train_data)
dataset_test_df = pd.concat(test_data)
print(f'>>>>>DONE Gathering result...')


print(f'>>>>>Saving train...')
with open('../LOO_data_0/{}_train.pickle'.format(dataset_name), 'wb') as pickle_file:
    pickle.dump(dataset_train_df, pickle_file)

print(f'>>>>>Saving test...')
with open('../LOO_data_0/{}_test.pickle'.format(dataset_name), 'wb') as pickle_file:
    pickle.dump(dataset_test_df, pickle_file)

print(f'>>>>>Saving one log user...')
with open('../LOO_data_0/{}_one_log_user.pickle'.format(dataset_name), 'wb') as pickle_file:
    pickle.dump(one_log_user_list, pickle_file)












import sys
import json
import re
import pandas as pd
from math import log
import argparse
import pickle

time_start = pd.Timestamp('2020-01-12')
time_end = pd.Timestamp('2020-07-13')

print('process vod/tv data...')
print('training time start: {}'.format(time_start))
print('training time end: {}'.format(time_end))

df = pd.read_parquet('../raw_data/interaction_log_v6_20200724_train.parquet', engine='pyarrow')
df = df[ (df['client_upload_timestamp'] > time_start) & (df['client_upload_timestamp'] <= time_end) ]
vod_df = df[ (df['item_type']=='movie') | (df['item_type']=='series') ]
tv_df = df[ ~((df['item_type']=='movie') | (df['item_type']=='series')) ]

vod_df = vod_df[vod_df['interaction'] == 'play']
tv_df = tv_df[tv_df['interaction'] == 'play']

tv_one_log_user = []
tv_train_data = []
tv_test_data = []

print("total user amount = {}".format(len(tv_df.user_id.unique())))
count = 0

for user in tv_df.user_id.unique():
    count += 1
    if len(tv_df[tv_df['user_id'] == user]) == 1:
        tv_one_log_user.append(user)
        continue
    s_user_tv_df = tv_df[tv_df['user_id'] == user].sort_values('client_upload_timestamp')
    user_test_log = s_user_tv_df.tail(1)
    user_train_log = s_user_tv_df.iloc[:(len(s_user_tv_df)-1)]
    tv_test_data.append(user_test_log)
    tv_train_data.append(user_train_log)
    
    if count % 2000 == 0:
        print("Finish {} users".format(count))

tv_train_df = pd.concat(tv_train_data)
tv_test_df = pd.concat(tv_test_data)

with open('../LOO_data/tv_train.pickle', 'wb') as pickle_file:
    pickle.dump(tv_train_df, pickle_file)

with open('../LOO_data/tv_test.pickle', 'wb') as pickle_file:
    pickle.dump(tv_test_df, pickle_file)

with open('../LOO_data/tv_one_log_user.pickle', 'wb') as pickle_file:
    pickle.dump(tv_one_log_user, pickle_file)
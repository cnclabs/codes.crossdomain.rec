import pickle
import pandas as pd
import json
from datetime import datetime
import multiprocessing 
from multiprocessing import Pool
import numpy as np

jf = open('../raw_data/Home_and_Kitchen_5.json', 'r')
line_list = []
for line in jf.readlines():
    dic = json.loads(line)
    line_list.append(dic)

h_and_k = pd.DataFrame(line_list)
h_and_k['date'] = h_and_k['unixReviewTime'].apply(lambda x:\
                                             datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

h_and_k_2 = h_and_k[(h_and_k['date'] >= '2016-10-04 00:00:00') & (h_and_k['date'] <= '2018-10-04 00:00:00')]

def process_train_test(user_list):
    h_and_k_one_log_user = []
    h_and_k_train_data = []
    h_and_k_test_data = []

    for user in user_list:
        if len((h_and_k_2[h_and_k_2['reviewerID'] == user])) == 1:
            h_and_k_one_log_user.append(user)
            continue
        s_user_h_and_k_2 = h_and_k_2[h_and_k_2['reviewerID'] == user]\
                                    .sort_values('date')

        user_test_log = s_user_h_and_k_2.tail(1)
        user_train_log = s_user_h_and_k_2.iloc[:(len(s_user_h_and_k_2)-1)]
        
        h_and_k_train_data.append(user_train_log)
        h_and_k_test_data.append(user_test_log)
    
    return [h_and_k_one_log_user, h_and_k_train_data, h_and_k_test_data]

total_users = h_and_k_2.reviewerID.unique()
cpu_amount = multiprocessing.cpu_count()
mp = Pool(cpu_amount)
split_datas = np.array_split(list(total_users), cpu_amount)
results = mp.map(process_train_test ,split_datas)
mp.close()

one_log_user_list = []
train_data = []
test_data = []

for r in results:
    one_log_user_list += r[0]
    train_data += r[1]
    test_data += r[2]


h_and_k_train_df = pd.concat(train_data)
h_and_k_test_df = pd.concat(test_data)


with open('../LOO_data/hk_train.pickle', 'wb') as pickle_file:
    pickle.dump(h_and_k_train_df, pickle_file)

with open('../LOO_data/hk_test.pickle', 'wb') as pickle_file:
    pickle.dump(h_and_k_test_df, pickle_file)

with open('../LOO_data/hk_one_log_user.pickle', 'wb') as pickle_file:
    pickle.dump(one_log_user_list, pickle_file)












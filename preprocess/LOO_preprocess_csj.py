import pickle
import pandas as pd
import json
from datetime import datetime
import multiprocessing 
from multiprocessing import Pool
import numpy as np

jf = open('../raw_data/Clothing_Shoes_and_Jewelry_5.json', 'r')
line_list = []
for line in jf.readlines():
    dic = json.loads(line)
    line_list.append(dic)

cs_and_j = pd.DataFrame(line_list)
cs_and_j['date'] = cs_and_j['unixReviewTime'].apply(lambda x:\
                                             datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

# align with h_k, so the latest date of the data is 2018-10-04
cs_and_j_2 = cs_and_j[(cs_and_j['date'] >= '2016-10-05 00:00:00') & (cs_and_j['date'] <= '2018-10-04 00:00:00')]

def process_train_test(user_list):
    cs_and_j_one_log_user = []
    cs_and_j_train_data = []
    cs_and_j_test_data = []

    for user in user_list:
        if len((cs_and_j_2[cs_and_j_2['reviewerID'] == user])) == 1:
            cs_and_j_one_log_user.append(user)

        s_user_cs_and_j_2 = cs_and_j_2[cs_and_j_2['reviewerID'] == user]\
                                    .sort_values('date')

        user_test_log = s_user_cs_and_j_2.tail(1)
        user_train_log = s_user_cs_and_j_2.iloc[:(len(s_user_cs_and_j_2)-1)]
        
        cs_and_j_train_data.append(user_train_log)
        cs_and_j_test_data.append(user_test_log)
    
    return [cs_and_j_one_log_user, cs_and_j_train_data, cs_and_j_test_data]


total_users = cs_and_j_2.reviewerID.unique()
print("total user amount = {}".format(len(total_users)))

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


cs_and_j_train_df = pd.concat(train_data)
cs_and_j_test_df = pd.concat(test_data)


with open('../LOO_data/csj_train.pickle', 'wb') as pickle_file:
    pickle.dump(cs_and_j_train_df, pickle_file)

with open('../LOO_data/csj_test.pickle', 'wb') as pickle_file:
    pickle.dump(cs_and_j_test_df, pickle_file)

with open('../LOO_data/csj_one_log_user.pickle', 'wb') as pickle_file:
    pickle.dump(one_log_user_list, pickle_file)












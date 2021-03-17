import pickle
import pandas as pd
import json
from datetime import datetime
import multiprocessing 
from multiprocessing import Pool
import numpy as np

jf = open('../raw_data/Movies_and_TV_5.json', 'r')
line_list = []
for line in jf.readlines():
    dic = json.loads(line)
    line_list.append(dic)

m_and_t = pd.DataFrame(line_list)
m_and_t['date'] = m_and_t['unixReviewTime'].apply(lambda x:\
                                             datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

m_and_t_2 = m_and_t[(m_and_t['date'] >= '2016-10-01 00:00:00') & (m_and_t['date'] <= '2018-10-01 00:00:00')]

def process_train_test(user_list):
    m_and_t_one_log_user = []
    m_and_t_train_data = []
    m_and_t_test_data = []

    for user in user_list:
        if len((m_and_t_2[m_and_t_2['reviewerID'] == user])) == 1:
            m_and_t_one_log_user.append(user)
            # fixed
            continue
        s_user_m_and_t_2 = m_and_t_2[m_and_t_2['reviewerID'] == user]\
                                    .sort_values('date')

        user_test_log = s_user_m_and_t_2.tail(1)
        user_train_log = s_user_m_and_t_2.iloc[:(len(s_user_m_and_t_2)-1)]
        
        m_and_t_train_data.append(user_train_log)
        m_and_t_test_data.append(user_test_log)
    
    return [m_and_t_one_log_user, m_and_t_train_data, m_and_t_test_data]


total_users = m_and_t_2.reviewerID.unique()
print("total user amount = {}".format(len(total_users)))

cpu_amount = multiprocessing.cpu_count()
cpu_amount = cpu_amount - 2
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


m_and_t_train_df = pd.concat(train_data)
m_and_t_test_df = pd.concat(test_data)


with open('../LOO_data/mt_train.pickle', 'wb') as pickle_file:
    pickle.dump(m_and_t_train_df, pickle_file)

with open('../LOO_data/mt_test.pickle', 'wb') as pickle_file:
    pickle.dump(m_and_t_test_df, pickle_file)

with open('../LOO_data/mt_one_log_user.pickle', 'wb') as pickle_file:
    pickle.dump(one_log_user_list, pickle_file)






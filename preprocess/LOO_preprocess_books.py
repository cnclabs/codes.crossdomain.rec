import pickle
import pandas as pd
import json
from datetime import datetime
import multiprocessing 
from multiprocessing import Pool
import numpy as np

jf = open('../raw_data/Books_5.json', 'r')
line_list = []
for line in jf.readlines():
    dic = json.loads(line)
    line_list.append(dic)

books = pd.DataFrame(line_list)
books['date'] = books['unixReviewTime'].apply(lambda x:\
                                             datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

books_2 = books[(books['date'] >= '2016-10-02 00:00:00') & (books['date'] <= '2018-10-02 00:00:00')]

def process_train_test(user_list):
    books_one_log_user = []
    books_train_data = []
    books_test_data = []        

    for user in user_list:
        if len((books_2[books_2['reviewerID'] == user])) == 1:
            books_one_log_user.append(user)
            # fixed
            continue
        s_user_books_2 = books_2[books_2['reviewerID'] == user]\
                                    .sort_values('date')

        user_test_log = s_user_books_2.tail(1)
        user_train_log = s_user_books_2.iloc[:(len(s_user_books_2)-1)]
        
        books_train_data.append(user_train_log)
        books_test_data.append(user_test_log)
    
    return [books_one_log_user, books_train_data, books_test_data]


total_users = books_2.reviewerID.unique()
print("total user amount = {}".format(len(total_users)))

cpu_amount = multiprocessing.cpu_count()
cpu_amount = cpu_amount - 10
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


books_train_df = pd.concat(train_data)
books_test_df = pd.concat(test_data)


with open('../LOO_data/books_train.pickle', 'wb') as pickle_file:
    pickle.dump(books_train_df, pickle_file)

with open('../LOO_data/books_test.pickle', 'wb') as pickle_file:
    pickle.dump(books_test_df, pickle_file)

with open('../LOO_data/books_one_log_user.pickle', 'wb') as pickle_file:
    pickle.dump(one_log_user_list, pickle_file)






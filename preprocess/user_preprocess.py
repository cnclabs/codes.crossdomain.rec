import pickle
import pandas as pd
import random


# ------------ TV-VOD ------------
# vod
with open('../LOO_data/vod_train.pickle', 'rb') as pf:
    vod_train = pickle.load(pf)

vod_train['user_id'] = vod_train['user_id'].apply(lambda x: 'user_'+x)
vod_train['item_id'] = vod_train['item_id'].apply(lambda x: 'vod_'+x)

# tv (source)
with open('../LOO_data/tv_train.pickle', 'rb') as pf:
    tv_train = pickle.load(pf)

tv_train['user_id'] = tv_train['user_id'].apply(lambda x: 'user_'+x)
tv_train['item_id'] = tv_train['item_id'].apply(lambda x: 'tv_'+x)

shared_users = set(vod_train.user_id).intersection(set(tv_train.user_id))
sample_amount = 2000
random.seed(3)
cold_users = random.sample(shared_users, sample_amount)

# save
with open('../user/tv_vod_shared_users.pickle', 'wb') as pf:
    pickle.dump(shared_users, pf)
with open('../user/tv_vod_cold_users.pickle', 'wb') as pf:
    pickle.dump(cold_users, pf)

# ------------ CSJ-HK ------------
# hk
with open('../LOO_data/hk_train.pickle', 'rb') as pf:
    hk_train = pickle.load(pf)

hk_train['reviewerID'] = hk_train['reviewerID'].apply(lambda x: 'user_'+x)

# csj (source)
with open('../LOO_data/csj_train.pickle', 'rb') as pf:
    csj_train = pickle.load(pf)

csj_train['reviewerID'] = csj_train['reviewerID'].apply(lambda x: 'user_'+x)

shared_users = set(hk_train.reviewerID).intersection(set(csj_train.reviewerID))
sample_amount = 4000
random.seed(3)
cold_users = random.sample(shared_users, sample_amount)

# save
with open('../user/csj_hk_shared_users.pickle', 'wb') as pf:
    pickle.dump(shared_users, pf)
with open('../user/csj_hk_cold_users.pickle', 'wb') as pf:
    pickle.dump(cold_users, pf)

# ------------ MT-B ------------
# books
with open('../LOO_data/books_train.pickle', 'rb') as pf:
    books_train = pickle.load(pf)

books_train['reviewerID'] = books_train['reviewerID'].apply(lambda x: 'user_'+x)

# movie_tv (source)
with open('../LOO_data/mt_train.pickle', 'rb') as pf:
    mt_train = pickle.load(pf)

mt_train['reviewerID'] = mt_train['reviewerID'].apply(lambda x: 'user_'+x)

shared_users = set(books_train.reviewerID).intersection(set(mt_train.reviewerID))
sample_amount = 8000
random.seed(3)
cold_users = random.sample(shared_users, sample_amount)

# save
with open('../user/mt_books_shared_users.pickle', 'wb') as pf:
    pickle.dump(shared_users, pf)
with open('../user/mt_books_cold_users.pickle', 'wb') as pf:
    pickle.dump(cold_users, pf)







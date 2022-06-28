import pickle
import pandas as pd 
from math import log

save_dir='/tmp2/wtchiang/CPR_paper/CPR/input'

# ------------ TV-VOD ------------
# vod
with open('../../LOO_data/vod_train.pickle', 'rb') as pf:
    vod_train = pickle.load(pf)

vod_train['user_id'] = vod_train['user_id'].apply(lambda x: 'user_'+x)
vod_train['item_id'] = vod_train['item_id'].apply(lambda x: 'vod_'+x)
vod_train_graph = vod_train.groupby(['user_id', 'item_id']).size().apply(lambda x: log(x+1.))
_df = pd.concat([vod_train_graph])
_df.to_csv('../input/vod_train_input.txt', header=False, sep='\t')

# tv (source)
with open('../../LOO_data/tv_train.pickle', 'rb') as pf:
    tv_train = pickle.load(pf)

tv_train['user_id'] = tv_train['user_id'].apply(lambda x: 'user_'+x)
tv_train['item_id'] = tv_train['item_id'].apply(lambda x: 'tv_'+x)
tv_train_graph = tv_train.groupby(['user_id', 'item_id']).size().apply(lambda x: log(x+1.))
tv_train_graph.to_csv('../input/all_tv_train_input.txt', header=False, sep='\t')
pd.concat([vod_train_graph, tv_train_graph]).to_csv('../input/all_cpr_train_u_tv+vod.txt', header=False, sep='\t')

# for cold start
with open('../../user/tv_vod_cold_users.pickle', 'rb') as pf:
    tv_vod_cold_users = pickle.load(pf)

cold_vod_train = vod_train[~vod_train['user_id'].isin(tv_vod_cold_users)]
cold_vod_train_graph = cold_vod_train.groupby(['user_id', 'item_id']).size().apply(lambda x: log(x+1.))
cold_vod_train_graph.to_csv('../input/cold_vod_train_input.txt', header=False, sep='\t')
pd.concat([cold_vod_train_graph, tv_train_graph]).to_csv('../input/cold_cpr_train_u_tv+vod.txt', header=False, sep='\t')

# ------------ CSJ-HK ------------
# hk
with open('../../LOO_data/hk_train.pickle', 'rb') as pf:
    hk_train = pickle.load(pf)

hk_train['reviewerID'] = hk_train['reviewerID'].apply(lambda x: 'user_'+x)
hk_train_graph = hk_train.groupby(['reviewerID', 'asin']).size().apply(lambda x: log(x+1.))
_df = pd.concat([hk_train_graph])
_df.to_csv('../input/hk_train_input.txt', header=False, sep='\t')

# csj (source)
with open('../../LOO_data/csj_train.pickle', 'rb') as pf:
    csj_train = pickle.load(pf)

csj_train['reviewerID'] = csj_train['reviewerID'].apply(lambda x: 'user_'+x)
csj_train_graph = csj_train.groupby(['reviewerID', 'asin']).size().apply(lambda x: log(x+1.))
csj_train_graph.to_csv('../input/all_csj_train_input.txt', header=False, sep='\t')
pd.concat([hk_train_graph, csj_train_graph]).to_csv('../input/all_cpr_train_u_csj+hk.txt', header=False, sep='\t')

# for cold start
with open('../../user/csj_hk_cold_users.pickle', 'rb') as pf:
    csj_hk_cold_users = pickle.load(pf)

cold_hk_train = hk_train[~hk_train['reviewerID'].isin(csj_hk_cold_users)]
cold_hk_train_graph = cold_hk_train.groupby(['reviewerID', 'asin']).size().apply(lambda x: log(x+1.))
cold_hk_train_graph.to_csv('../input/cold_hk_train_input.txt', header=False, sep='\t')
pd.concat([cold_hk_train_graph, csj_train_graph]).to_csv('../input/cold_cpr_train_u_csj+hk.txt', header=False, sep='\t')

# ------------ MT-B ------------
# books
with open('../../LOO_data/books_train.pickle', 'rb') as pf:
    books_train = pickle.load(pf)

books_train['reviewerID'] = books_train['reviewerID'].apply(lambda x: 'user_'+x)
books_train_graph = books_train.groupby(['reviewerID', 'asin']).size().apply(lambda x: log(x+1.))
_df = pd.concat([books_train_graph])
_df.to_csv('../input/books_train_input.txt', header=False, sep='\t')

# movie_tv (source)
with open('../../LOO_data/mt_train.pickle', 'rb') as pf:
    mt_train = pickle.load(pf)

mt_train['reviewerID'] = mt_train['reviewerID'].apply(lambda x: 'user_'+x)
mt_train_graph = mt_train.groupby(['reviewerID', 'asin']).size().apply(lambda x: log(x+1.))
mt_train_graph.to_csv('../input/all_mt_train_input.txt', header=False, sep='\t')
pd.concat([books_train_graph, mt_train_graph]).to_csv('../input/all_cpr_train_u_mt+books.txt', header=False, sep='\t')

# for cold start
with open('../../user/mt_books_cold_users.pickle', 'rb') as pf:
    mt_books_cold_users = pickle.load(pf)

cold_books_train = books_train[~books_train['reviewerID'].isin(mt_books_cold_users)]
cold_books_train_graph = cold_books_train.groupby(['reviewerID', 'asin']).size().apply(lambda x: log(x+1.))
cold_books_train_graph.to_csv('../input/cold_books_train_input.txt', header=False, sep='\t')
pd.concat([cold_books_train_graph, mt_train_graph]).to_csv('../input/cold_cpr_train_u_mt+books.txt', header=False, sep='\t')





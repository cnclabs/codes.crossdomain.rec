import argparse
import pandas as pd
from engine import Engine
from data import SampleGenerator
import pickle
from multiprocessing import Pool
import multiprocessing
from utils import calculate_Recall, calculate_NDCG
import numpy as np
import torch
from torch.autograd import Variable
import random

parser = argparse.ArgumentParser('DDTCDR')
# Path Arguments
parser.add_argument('--num_epoch', type=int, default=100,help='number of epoches')
parser.add_argument('--batch_size', type=int, default=1024,help='batch size')
parser.add_argument('--lr', type=int, default=1e-2,help='learning rate')
parser.add_argument('--latent_dim', type=int, default=100,help='latent dimensions')
parser.add_argument('--alpha', type=int, default=0.03,help='dual learning rate')
parser.add_argument('--cuda', type=bool, default=False,help='use of cuda')
parser.add_argument('--test_users', type=str, help='{target, shared}')
args = parser.parse_args()

def dictionary(terms):
    term2idx = {}
    idx2term = {}
    for i in range(len(terms)):
        term2idx[terms[i]] = i
        idx2term[i] = terms[i]
    return term2idx, idx2term

mlp_config = {'num_epoch': args.num_epoch,
              'batch_size': args.batch_size,
              'optimizer': 'sgd',
              'lr': args.lr,
              'latent_dim': args.latent_dim,
              'nlayers':1,
              'alpha':args.alpha,
              'layers': [2*args.latent_dim,64,args.latent_dim],  # layers[0] is the concat of latent user vector & latent item vector
              'use_cuda': args.cuda,
              'pretrain': False,
              'device_id':0}

#Load Datasets
# userID, itemID, rating, user_embedding, item_embedding
with open('./input/mt_lightfm.pickle', 'rb') as pickle_file:
  book = pickle.load(pickle_file)
with open('./input/books_lightfm.pickle', 'rb') as pickle_file:
  movie = pickle.load(pickle_file)


book.columns = ['userId', 'itemId', 'rating', 'user_embedding', 'item_embedding']
movie.columns = ['userId', 'itemId', 'rating', 'user_embedding', 'item_embedding']

sample_book_generator = SampleGenerator(ratings=book)
sample_movie_generator = SampleGenerator(ratings=movie)

# ground truth 
with open('../../LOO_data/books_test.pickle', 'rb') as pf:
    books_test_df = pickle.load(pf)
books_test_df = books_test_df[['reviewerID', 'asin']]
books_test_df.columns = ['user_id', 'item_id']
books_test_df['user_id'] = books_test_df['user_id'].apply(lambda x: 'user_'+x)


# rec pool
## load books_train_df
books_train_df = pd.read_csv('../../CPR/input/books_train_input.txt', header=None, sep='\t')
books_train_df.columns = ['user_id', 'item_id', 'rating']
total_item_set = set(books_train_df.item_id)

## sample testing users
sample_amount = 8000
random.seed(3)
if args.test_users == 'target':
  testing_users = random.sample(set(books_train_df.user_id), sample_amount)
if args.test_users == 'shared':
  with open('../../user/mt_books_shared_users.pickle', 'rb') as pf:
    shared_users = pickle.load(pf)
  testing_users = random.sample(set(shared_users), sample_amount)


# Generate user 100 rec pool 
print("Start generating user rec dict...")

def process_user_rec_dict(user_list):
  user_rec_dict = {}

  for user in user_list:
      watched_set = set(books_train_df[books_train_df['user_id'] == user].item_id)
      neg_pool = total_item_set - watched_set
      random.seed(5)
      neg_99 = random.sample(neg_pool, 99)
      user_rec_pool = list(neg_99) + list(books_test_df[books_test_df['user_id'] == user].item_id)
      user_rec_dict[user] = user_rec_pool

  return user_rec_dict

cpu_amount = multiprocessing.cpu_count()
worker = cpu_amount - 2
mp = Pool(worker)
split_datas = np.array_split(list(testing_users), worker)
results = mp.map(process_user_rec_dict ,split_datas)
mp.close()

user_rec_pool_dict = {}
for r in results:
  user_rec_pool_dict.update(r)

print("user rec pool dict generated!")

config = mlp_config
engine = Engine(config)

# read rec_item_pool embedding
with open('./input/books_item_emb_dict.pickle', 'rb') as pickle_file:
    rec_pool_emb_dict = pickle.load(pickle_file)

# read books (target) domain emb
with open('./input/books_emb_dict.pickle', 'rb') as pickle_file:
    books_emb_dict = pickle.load(pickle_file)


def multi_eval(user_list):
  user_rec_dict = {}
  for user in user_list:
    user_score_dict = {}
    for item in user_rec_pool_dict[user]:
      # if item only shows in test data, skip it
      if item not in rec_pool_emb_dict.keys():
        continue
      if modelB.config['use_cuda'] is True:
        score = modelB(Variable(torch.FloatTensor(books_emb_dict[user])).cuda(),\
                       Variable(torch.FloatTensor(rec_pool_emb_dict[item])).cuda())
      else:
        score = modelB(Variable(torch.FloatTensor(books_emb_dict[user])),\
                       Variable(torch.FloatTensor(rec_pool_emb_dict[item])))
      user_score_dict[item] = score

    s_user_score_dict = {k:v for k,v in sorted(user_score_dict.items(), \
                                                  key=lambda x: x[1], reverse=True)}
    user_rec_dict[user] = list(s_user_score_dict.keys())[:20] #k_max

  return user_rec_dict


for epoch in range(config['num_epoch']):
    comb_user_rec_dict = {}

    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_book_loader = sample_book_generator.instance_a_train_loader(config['batch_size'])
    train_movie_loader = sample_movie_generator.instance_a_train_loader(config['batch_size'])
    engine.train_an_epoch(train_book_loader, train_movie_loader, epoch_id=epoch)
    modelB = engine.model()

    cpu_amount = multiprocessing.cpu_count()
    mp = Pool(cpu_amount-2)
    split_datas = np.array_split(testing_users, cpu_amount-2)
    results = mp.map(multi_eval, split_datas)
    
    for result in results:
      comb_user_rec_dict.update(result)

    k_amount = [1, 3, 5, 10, 20]
    k_max = max(k_amount)
    count = 0
    total_rec=[0, 0, 0, 0, 0]
    total_ndcg=[0, 0, 0, 0, 0]

    print("testing users amount = {}".format(len(testing_users)))

    for user in testing_users:
      count += 1
      # ground truth
      test_data = list(books_test_df[books_test_df['user_id']==user]['item_id'])
      
      for k in range(len(k_amount)):
        recomm_k = comb_user_rec_dict[user][:k_amount[k]]

        total_rec[k]+=calculate_Recall(test_data, recomm_k)
        total_ndcg[k]+=calculate_NDCG(test_data, recomm_k)

      if count%2000==0:   
        print(count, ' users counted.')

    total_rec=np.array(total_rec)
    total_ndcg=np.array(total_ndcg)
    
    recall_10 = total_rec[3]/count
    ndcg_10 = total_ndcg[3]/count

    print('[BOOKS Evluating Epoch {}], Recall@10 = {}, NDCG@10 = {}'.format(epoch, str(recall_10), str(ndcg_10)))
    engine.save('./model_status/mt_books/' + args.test_users + '_' + 'epoch_' + str(epoch) +'_model_')

    with open('./result/DDTCDR_amazon_books_lightfm' + '_' + args.test_users + '_' + 'result.txt', 'a') as fw:
        fw.writelines(['=================================\n',
                'epoch = ', str(epoch),
                '\n evaluated users amount: ',
                str(len(testing_users)),
                '\n--------------------------------',
               '\n recall@1: ',
                str(total_rec[0]/count),
               '\n NDCG@1: ',
                str(total_ndcg[0]/count),
               '\n--------------------------------',
               '\n recall@3: ',
                str(total_rec[1]/count),
               '\n NDCG@3: ',
                str(total_ndcg[1]/count),
               '\n--------------------------------',
               '\n recall@5: ',
                str(total_rec[2]/count),
               '\n NDCG@5: ',
                str(total_ndcg[2]/count),
               '\n--------------------------------',
               '\n recall@10: ',
               str(total_rec[3]/count),
               '\n NDCG@10: ',
               str(total_ndcg[3]/count),
               '\n--------------------------------',
               '\n recall@20: ',
               str(total_rec[4]/count),
               '\n NDCG@20: ',
               str(total_ndcg[4]/count),
               '\n'])

    print('Finished epoch {}!'.format(epoch))













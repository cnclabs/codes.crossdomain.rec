'''
source code
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)

change to Bi-TGCF
paper: Meng Liu et al. Cross Domain Recommendation via Bi-directional Transfer
Graph Collaborative Filtering Networks. In CIKM 2020.

@author: Meng Liu (sunshinel@hust.edu.cn)
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
from utility.helper import *
from collections import defaultdict
import os

def get_merge_user_dict(source_data_path, target_data_path):
    merge_user_dict = {}
    counter = 0

    with open(source_data_path, "r") as fp_s:
        for line in fp_s:
            line = line.strip("\n")
            user_id, _, _ = line.split("\t")
            if user_id not in merge_user_dict:
                merge_user_dict[user_id] = counter
                counter += 1
            else:
                continue 

    with open(target_data_path, "r") as fp_t:
        for line in fp_t:
            line = line.strip("\n")
            user_id, _, _ = line.split("\t")
            if user_id not in merge_user_dict:
                merge_user_dict[user_id] = counter
                counter += 1
            else:
                continue
    return merge_user_dict
    



class Data(object):
    def __init__(self, data_path, current_dataset, batch_size,neg_num, merge_user_dict): 
        # add preprocessed merge_user_dict to do true cross-domain
        self.neg_num = neg_num
        self.path = f"./internal_data/{current_dataset}/"
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.batch_size = batch_size
        # train_file = path +'/train.txt'
        # test_file = path + '/test.txt'
        train_file = data_path

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        self.merge_user_dict = merge_user_dict # new_index -> org_index

        self.exist_users = []


        ''' 
        with open(train_file, "r") as f:
            line = f.readline().strip('\n')
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                self.n_users = max(self.n_users, u)
                self.n_items = max(self.n_items, i)
                self.n_train += 1
                line = f.readline().strip('\n')

        self.n_items += 1
        self.n_users += 1
        '''

        # rewrite dataloader(1)
        # count n_user, n_items
        # construct reindex map
        user_counter = 0
        item_counter = 0
        # u_i_dict = defaultdict(list)
        # use a pair of list rather then dict to construct adj matrix
        # borrow from NeuRec LightGCN
        user_pair_list = []
        item_pair_list = []
        item_reindex_dict = {}
        with open(train_file, "r") as f:
            line = f.readline().strip('\n')
            while line != None and line != "":
                user_id, item_id, _ = line.split("\t")
                # since input file has no item_ prefix
                if '_' in item_id:
                    item_id = item_id[item_id.find('_')+1:]
                item_id = 'item_'+item_id
                if item_id not in item_reindex_dict:
                    item_reindex_dict[item_id] = item_counter
                    item_counter += 1
                user_pair_list.append(self.merge_user_dict[user_id])
                item_pair_list.append(item_reindex_dict[item_id])

                
                # u, i = int(arr[0]), int(arr[1])
                # self.n_users = max(self.n_users, u)
                # self.n_items = max(self.n_items, i)
                # self.n_train += 1
                line = f.readline().strip('\n')
        self.n_users = len(self.merge_user_dict.keys())
        self.n_items = item_counter
        self.n_train = user_counter + item_counter
        self.item_reindex_dict = item_reindex_dict


        # self.print_statistics()
        # self.negativeList = self.read_neg_file(path) # test on own neg dataset, no need to use
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.ratingList = []
        # self.train_items, self.test_set = {}, {}
        self.train_items = {}
        
        # rewrite dataloader(2) construct 
        for user, item in zip(user_pair_list, item_pair_list):
            self.R[user, item] = 1.0
        
        del user_pair_list, item_pair_list
        '''
        with open(train_file) as f_train:
            for l in f_train.readlines():
                if len(l) == 0: break
                l = l.strip('\n').split('\t')
                user, item, rating = int(l[0]), int(l[1]), float(l[2])
                if user in self.train_items.keys():
                    self.train_items[user].append(item)
                else:
                    self.train_items[user] = [item]
                if (rating > 0):
                    self.R[user, item] = 1.0
                    # self.R[uid][i] = 1
        '''

        ''' # not using test set here
        with open(test_file) as f_test:
            for l in f_train.readlines():
                if len(l) == 0: break
                l = l.strip('\n').split('\t')
                user, item, rating = int(l[0]), int(l[1]), float(l[2])
                if user in self.train_items.keys():
                    self.train_items[user].append(item)
                else:
                    self.train_items[user] = [item]
                if (rating > 0):
                    self.R[user, item] = 1.0
                    # self.R[uid][i] = 1

            line = f_test.readline().strip('\n')
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                if user in self.test_set.keys():
                    self.test_set[user].append(item)
                else:
                    self.test_set[user] = [item]
                self.ratingList.append([user, item])
                self.n_test += 1
                line = f_test.readline().strip('\n')
        '''

    def get_R_mat(self):
        return self.R
    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)


        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat
    
    
    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def read_neg_file(self,path):
        try:
            test_neg = path + '/test_neg.txt'
            test_neg_f = open(test_neg ,'r')
        except:
            negativeList = None
            return negativeList
        negativeList = []
        line = test_neg_f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            negatives = []
            for x in arr[1:]:  # arr[0] = (user, pos_item)
                item = int(x)
                negatives.append(item)
            negativeList.append(negatives)
            line = test_neg_f.readline()
        return negativeList
    def get_test_neg_item(self,u,negativeList):
        neg_items = negativeList[u]
        return neg_items
    def get_train_instance(self):
        user_input, item_input, labels = [],[],[]
        for (u, i) in self.R.keys():
            # positive instance
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            # negative negRatio instances
            for _ in range(self.neg_num):
                j = np.random.randint(self.n_items)
                while (u, j) in self.R.keys():
                    j = np.random.randint(self.n_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)
        return np.array(user_input),np.array(item_input),np.array(labels)

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self,save_log):
        pprint('n_users=%d, n_items=%d' % (self.n_users, self.n_items),save_log)
        pprint('n_interactions=%d' % (self.n_train + self.n_test),save_log)
        pprint('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)),save_log)


    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split('\t')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write('\t'.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state

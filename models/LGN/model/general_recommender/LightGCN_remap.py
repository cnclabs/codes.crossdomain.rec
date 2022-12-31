"""
Paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
Author: Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang
Reference: https://github.com/hexiangnan/LightGCN
"""

import scipy.sparse as sp
import tensorflow as tf
import numpy as np
from model.AbstractRecommender import AbstractRecommender
from util import timer
from util import l2_loss, inner_product, log_loss
from data import PairwiseSampler
import pickle
import os


class LightGCN(AbstractRecommender):
    def __init__(self, sess, dataset, config):
        super(LightGCN, self).__init__(dataset, config)
        self.lr = config['lr']
        self.reg = config['reg']
        self.emb_dim = config['embed_size']
        self.batch_size = config['batch_size']
        self.epochs = config["epochs"]
        self.n_layers = config['n_layers']

        self.dataset = dataset
        self.n_users, self.n_items = self.dataset.num_users, self.dataset.num_items
        self.user_pos_train = self.dataset.get_user_train_dict(by_time=False)
        self.all_users = list(self.user_pos_train.keys())

        self.norm_adj = self.create_adj_mat(config['adj_type'])
        self.sess = sess

    @timer
    def create_adj_mat(self, adj_type):
        user_list, item_list = self.dataset.get_train_interactions()
        user_np = np.array(user_list, dtype=np.int32)
        item_np = np.array(item_list, dtype=np.int32)
        ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.n_users + self.n_items
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.n_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        if adj_type == 'plain':
            adj_matrix = adj_mat
            print('use the plain adjacency matrix')
        elif adj_type == 'norm':
            adj_matrix = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
            print('use the normalized adjacency matrix')
        elif adj_type == 'gcmc':
            adj_matrix = normalized_adj_single(adj_mat)
            print('use the gcmc adjacency matrix')
        elif adj_type == 'pre':
            # pre adjcency matrix
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            adj_matrix = norm_adj_tmp.dot(d_mat_inv)
            print('use the pre adjcency matrix')
        else:
            mean_adj = normalized_adj_single(adj_mat)
            adj_matrix = mean_adj + sp.eye(mean_adj.shape[0])
            print('use the mean adjacency matrix')

        return adj_matrix

    def _create_variable(self):

        self.users = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.items = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.compat.v1.placeholder(tf.int32, shape=(None,))

        self.weights = dict()
        initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        self.weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
        self.weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')

    def build_graph(self):
        self._create_variable()
        self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(params=self.ua_embeddings, ids=self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(params=self.ia_embeddings, ids=self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(params=self.ia_embeddings, ids=self.neg_items)
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(params=self.weights['user_embedding'], ids=self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(params=self.weights['item_embedding'], ids=self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(params=self.weights['item_embedding'], ids=self.neg_items)

        """
        *********************************************************
        Inference for the testing phase.
        """
        self.item_embeddings_final = tf.Variable(tf.zeros([self.n_items, self.emb_dim]),
                                                 dtype=tf.float32, name="item_embeddings_final", trainable=False)
        self.user_embeddings_final = tf.Variable(tf.zeros([self.n_users, self.emb_dim]),
                                                 dtype=tf.float32, name="user_embeddings_final", trainable=False)

        self.assign_opt = [tf.compat.v1.assign(self.user_embeddings_final, self.ua_embeddings),
                           tf.compat.v1.assign(self.item_embeddings_final, self.ia_embeddings)]

        u_embed = tf.nn.embedding_lookup(params=self.user_embeddings_final, ids=self.users)
        self.batch_ratings = tf.matmul(u_embed, self.item_embeddings_final, transpose_a=False, transpose_b=True)
        self.user_embeddings =  tf.nn.embedding_lookup(params=self.user_embeddings_final, ids=self.users)
        self.item_embeddings =  tf.nn.embedding_lookup(params=self.item_embeddings_final, ids=self.items)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                           self.pos_i_g_embeddings,
                                                           self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss

        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _create_lightgcn_embed(self):
        adj_mat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            #try:
            side_embeddings = tf.sparse.sparse_dense_matmul(adj_mat, ego_embeddings, name="sparse_dense")
            #except:
            #dense_adj_mat = tf.sparse.to_dense(adj_mat)
            #dense_ego_embeddings = tf.sparse.to_dense(ego_embeddings)
            #side_embeddings = tf.matmul(dense_adj_mat, ego_embeddings, a_is_sparse = True, b_is_sparse = False)

            # transformed sum messages of neighbors.
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(input_tensor=all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = inner_product(users, pos_items)
        neg_scores = inner_product(users, neg_items)

        regularizer = l2_loss(self.u_g_embeddings_pre, self.pos_i_g_embeddings_pre, self.neg_i_g_embeddings_pre)

        mf_loss = tf.reduce_sum(input_tensor=log_loss(pos_scores - neg_scores))

        emb_loss = self.reg * regularizer

        return mf_loss, emb_loss

    def train_model(self, dir_path, for_count):

        data_iter = PairwiseSampler(self.dataset, neg_num=1, batch_size=self.batch_size, shuffle=True)

        self.logger.info(self.evaluator.metrics_info())
        g_epoch = ""
        g_result = ""

        ###### for get embeddin
        dataset = self.dataset
        user_ids = dataset.userids
        item_ids = dataset.itemids
        try:
            assert os.path.isdir(os.path.join(dir_path, 'dic'))
        except:
            raise Exception("Lack of dictionaries for users !!")
        if not os.path.isdir(os.path.join(dir_path, str(for_count))):
            os.mkdir(os.path.join(dir_path, str(for_count)))
            os.mkdir(os.path.join(dir_path, str(for_count)+'/evaluations'))

        dataset_attrs = dataset.dataset_name.split('_')
        d_name = dataset_attrs[0]
        d_train = dataset_attrs[1]
        d_test = dataset_attrs[2]
        
        if d_name == "tvvod" and d_train == "big":
            user_dic_path = os.path.join(dir_path, "dic/tvvod_user_sampled_dic.pickle")
            item_dic_path = os.path.join(dir_path, "dic/tvvod_item_sampled_dic.pickle")
        else:
            user_dic_path = os.path.join(dir_path, "dic/"+d_name+"_user_dic.pickle")
            item_dic_path = os.path.join(dir_path, "dic/"+d_name+"_item_dic.pickle")

        with open(user_dic_path, 'rb') as pf:
            user_dic = pickle.load(pf)
        with open(item_dic_path, 'rb') as pf:
            item_dic = pickle.load(pf)
        ###### 

        for epoch in range(self.epochs):
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                feed = {self.users: bat_users,
                        self.pos_items: bat_pos_items,
                        self.neg_items: bat_neg_items}
                self.sess.run(self.opt, feed_dict=feed)
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))
            g_epoch = epoch
            g_result = result
            if (epoch+1) % 10 == 0:
                with open(os.path.join(dir_path, str(for_count)+'/evaluations/'+dataset.dataset_name+'evaluation.txt'), 'a') as f:
                    f.write("epoch %d:\t%s\n" % (epoch, result))
            ###### write embeds
            if (epoch+1) % 100 == 0: 
                file_name = os.path.join(dir_path, str(for_count)+'/'+dataset.dataset_name+'_'+str(epoch+1)+'epoch.graph')
                self.writeEmbed(file_name, user_dic, item_dic)
            ######


    # @timer
    def evaluate_model(self):
        self.sess.run(self.assign_opt)
        return self.evaluator.evaluate(self)

    def predict(self, users, candidate_items=None):
        feed_dict = {self.users: users}
        ratings = self.sess.run(self.batch_ratings, feed_dict=feed_dict)
        if candidate_items is not None:
            ratings = [ratings[idx][u_item] for idx, u_item in enumerate(candidate_items)]
        return ratings
    def get_embeddings(self, users, items):
        feed_dict = {self.users: users, self.items: items}
        user_embeddings = self.sess.run(self.user_embeddings, feed_dict=feed_dict)
        item_embeddings = self.sess.run(self.item_embeddings, feed_dict=feed_dict)

        return user_embeddings, item_embeddings
    def findKey(self, dic, value):
        key_list = list(dic.keys())
        value_list = list(dic.values())
        return key_list[value_list.index(value)]
    def writeEmbed(self, file_name, user_dic, item_dic):
        user_embeds, item_embeds = self.get_embeddings([i for i in range(self.dataset.num_users)], [i for i in range(self.dataset.num_items)])
        with open(file_name, 'w') as f:
            print("Start Users' Embeddings ...")
            for i in range(dataset.num_users):
                userid_real = self.findKey(dataset.userids, i)
                userid_remap = self.findKey(user_dic, str(userid_real))
                f.write("user_"+str(userid_remap)+'\t')
                embed_str = ""
                for e in user_embeds[i]:
                    embed_str += str(e)+' '
                f.write(embed_str.strip(' ')+'\n')
            print("Finished Users ...")

            print("Start Items' Embeddings ...")
            for i in range(dataset.num_items):
                itemid_real = self.findKey(dataset.itemids, i)
                itemid_remap = self.findKey(item_dic, str(itemid_real))
                f.write("item_"+str(itemid_remap)+'\t')
                embed_str = ""
                for e in item_embeds[i]:
                    embed_str += str(e)+' '
                f.write(embed_str.strip(' ')+'\n')
            print("Finished Items ...")
        


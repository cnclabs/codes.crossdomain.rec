import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.contrib import layers
import numpy as np
import pickle
import argparse

parser=argparse.ArgumentParser(description='Infer Us from source domain to target domain')
parser.add_argument('--meta_path', type=str, help='the meta of the trained model')
parser.add_argument('--ckpt_path', type=str, help='the ckpt directory of the trained model')
parser.add_argument('--dataset_name', type=str, help='{tv_vod, csj_hk, mt_books}')
args=parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
meta_path = args.meta_path
ckpt_path = args.ckpt_path

if __name__ == "__main__":
    with open(args.dataset_name + '/' + 'lightfm_bpr_Us.pickle', 'rb') as pf:
        bpr_Us = pickle.load(pf)
    block_size = 10000
    block_amount = bpr_Us.shape[1] / block_size
    concat_list = []
    for i in range(int(block_amount)+1):
        bpr_Us_block = bpr_Us[:,i*block_size:(i+1)*block_size]
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
            all_vars = tf.trainable_variables()
            w1 = sess.run(all_vars[0])
            b1 = sess.run(all_vars[1])
            w2 = sess.run(all_vars[2])
            b2 = sess.run(all_vars[3]) 
            hidden1 = tf.nn.tanh(tf.matmul(w1, bpr_Us_block)+b1)
            pred = tf.nn.sigmoid(tf.matmul(w2, hidden1) + b2)
            test_output = sess.run(pred)

            print("Finsh inference...")
            print(test_output.shape)
            print("Lets Transpose")
            output = test_output.T
            print(output.shape)
            print(type(output))
            concat_list.append(output)
        concat_output = np.concatenate(concat_list)
        print(concat_output.shape)


with open('../../user/' + args.dataset_name + '_' + 'shared_users.pickle', 'rb') as pf:
    shared_users = pickle.load(pf)

shared_users_mapped_emb = {}
index=0
for i in shared_users:
    shared_users_mapped_emb[i] = concat_output[index]
    index += 1

# save
with open('./' + args.dataset_name + '/shared_users_mapped_emb_dict.pickle', 'wb') as pf:
    pickle.dump(shared_users_mapped_emb, pf)

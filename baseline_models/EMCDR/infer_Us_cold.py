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

# only infer cold_users
with open('../../user/' + args.dataset_name + '_' + 'cold_users.pickle', 'rb') as pf:
    cold_users = pickle.load(pf)

cold_users_array = []
with open('../BPR/graph/' + args.dataset_name.split('_')[0] + '_' + 'lightfm_bpr_10e-5.txt', 'r') as f:
    skip_f = f.readlines()[1:]
    for line in skip_f:
        line = line[:-1]
        prefix = line.split(' ')[0]
        emb=line.split(' ')[3:]
        if prefix in cold_users:
            cold_users_array.append(np.array(emb, dtype=np.float32))

cold_Us = np.array(cold_users_array).T

if __name__ == "__main__":
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        all_vars = tf.trainable_variables()
        w1 = sess.run(all_vars[0])
        b1 = sess.run(all_vars[1])
        w2 = sess.run(all_vars[2])
        b2 = sess.run(all_vars[3]) 
        hidden1 = tf.nn.tanh(tf.matmul(w1, cold_Us)+b1)
        pred = tf.nn.sigmoid(tf.matmul(w2, hidden1) + b2)
        test_output = sess.run(pred)
        
        print("Finsh inference...")
        print(test_output.shape)
        print("Lets Transpose")
        output = test_output.T
        print(output.shape)
        print(type(output))

cold_users_mapped_emb_dict = {}
index=0
for i in cold_users:
    cold_users_mapped_emb_dict[i] = output[index]
    index += 1

# save
with open('./' + args.dataset_name + '/cold_users_mapped_emb_dict.pickle', 'wb') as pf:
    pickle.dump(cold_users_mapped_emb_dict, pf)




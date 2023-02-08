import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.contrib import layers
import numpy as np
import pickle
import argparse
import os

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='Infer Us from source domain to target domain')
    parser.add_argument('--user_to_infer_path', type=str, required=True)
    parser.add_argument('--Us_path', type=str, required=True)
    parser.add_argument('--Us_id_map_path', type=str, required=True)
    parser.add_argument('--meta_path', type=str, help='the meta of the trained model')
    parser.add_argument('--ckpt_path', type=str, help='the ckpt directory of the trained model')
    parser.add_argument('--emb_save_path', type=str, required=True)
    args=parser.parse_args()
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    meta_path = args.meta_path
    ckpt_path = args.ckpt_path

    with open(args.user_to_infer_path, 'rb') as pf:
        users_to_infer = pickle.load(pf)

    with open(args.Us_id_map_path, 'rb') as pf:
        Us_id_map = pickle.load(pf)
    
    #_list = []
    #for k, v in Us_id_map.items():
    #    _list.append(v)

    #assert list(users_to_infer) == _list

    with open(args.Us_path, 'rb') as pf:
        Us = pickle.load(pf)
    block_size = 10000
    block_amount = Us.shape[1] / block_size
    concat_list = []
    for i in range(int(block_amount)+1):
        Us_block = Us[:,i*block_size:(i+1)*block_size]
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
            print('Restore from: ', ckpt_path)
            all_vars = tf.trainable_variables()
            w1 = sess.run(all_vars[0])
            b1 = sess.run(all_vars[1])
            w2 = sess.run(all_vars[2])
            b2 = sess.run(all_vars[3]) 
            hidden1 = tf.nn.tanh(tf.matmul(w1, Us_block)+b1)
            pred = tf.matmul(w2, hidden1) + b2
            test_output = sess.run(pred)

            print("Finsh inference...")
            print(test_output.shape)
            print("Lets Transpose")
            output = test_output.T
            print(output.shape)
            concat_list.append(output)
        concat_output = np.concatenate(concat_list)
        print(concat_output.shape)
    
    users_to_infer_mapped_emb = {}
    for org_id in users_to_infer:
        remap_id = Us_id_map[org_id]
        users_to_infer_mapped_emb[org_id] = concat_output[remap_id]
    
    # save
    with open(args.emb_save_path, 'wb') as pf:
        pickle.dump(users_to_infer_mapped_emb, pf)
    print("Done saved to: ", args.emb_save_path)

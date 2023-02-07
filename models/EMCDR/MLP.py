import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import pickle

def MLP(input_Us, input_Ut, beta, learning_rate, training_epochs, model_save_dir, display_step=100):
    '''多层感知机映射函数
    input: 
        input_Us(ndarray): 源领域矩阵
        input_Ut(ndarray): 目标领域矩阵 
        beta(float): 正则化参数
        learning_rate(float): 学习率
        training_epochs(int): 最大迭代次数
    output: 
        U, V: 分解后的矩阵
    '''
    # =====
    # =====
    k, m = np.shape(input_Us)
    # with tf.device('/gpu:0'): 
    with tf.device('/cpu:0'):
        # 1. 初始化参数
        w1 = tf.Variable(tf.truncated_normal([2 * k, k], stddev = 0.001), name="w1")
        b1 = tf.Variable(tf.truncated_normal([2 * k, 1], stddev = 0.001), name="b1")

        w2 = tf.Variable(tf.truncated_normal([k, 2 * k], stddev = 0.001), name="w2")
        b2 = tf.Variable(tf.truncated_normal([k, 1], stddev = 0.001), name="b2")

        Us = tf.placeholder(tf.float32,[k, None])
        Ut = tf.placeholder(tf.float32,[k, None])

        # 2. 构建模型
        hidden1 = tf.nn.tanh(tf.matmul(w1, Us)+b1)

        reg_w1 = layers.l2_regularizer(beta)(w1)
        reg_w2 = layers.l2_regularizer(beta)(w2)

        pred = tf.nn.sigmoid(tf.matmul(w2, hidden1) + b2)
        cost = tf.reduce_mean(tf.square(Ut - pred)) + reg_w1 + reg_w2
        train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

        init = tf.global_variables_initializer()

    # 3. 开始训练
    print("Start training...")
    with tf.Session(config=cfg) as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            #print("Start epoch {}".format(epoch+1))
            sess.run(train_step, feed_dict={Us: input_Us, Ut: input_Ut})

            if (epoch + 1) % display_step == 0:
                avg_cost = sess.run(cost, feed_dict={Us: input_Us, Ut: input_Ut})
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
           
                #os.makedirs(os.path.dirname(args.epoch_log), exist_ok=True)
                #with open(args.epoch_log, 'a') as file:
                #    file.writelines(["Epoch:", '%04d' % (epoch + 1), ", cost=", "{:.9f}".format(avg_cost), "\n"])

            #print("Finish epoch {}".format(epoch+1))

        # 打印变量
        variable_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable:", k)
            print("Shape: ", v.shape)
            print(v)
        
        # 保存模型
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(model_save_dir, 'mlp'))
        print("Optimization Finished!")
    
if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    cfg = tf.ConfigProto(log_device_placement=True)
    cfg.gpu_options.per_process_gpu_memory_fraction = 0.1
    
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    
    parser = argparse.ArgumentParser(description='EMCDR MLP mapping using lightfm embedding')
    #parser.add_argument('--epoch_log', type=str, help='epoch log saving path')
    parser.add_argument('--model_save_dir', type=str, required=True)
    parser.add_argument('--Us', type=str, required=True)
    parser.add_argument('--Ut', type=str, required=True)
    args=parser.parse_args()
    
    with open(args.Us, 'rb') as pf:
        Us = pickle.load(pf)
    print("Finish loading source...")
    print("Us shape = {}".format(np.shape(Us)))

    with open(args.Ut, 'rb') as pf:
        Ut = pickle.load(pf)
    print("Finish loading target...")
    print("Uv shape = {}".format(np.shape(Ut)))

    beta = 0.001
    learning_rate = 0.01
    training_epochs = 200
    display_step = 10
    
    MLP(Us, Ut, beta, learning_rate, training_epochs, args.model_save_dir, display_step)

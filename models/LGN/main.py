import os
import random
import numpy as np
import tensorflow as tf
import importlib
from data.dataset import Dataset
from util import Configurator, tool
import pickle


np.random.seed(2021)
random.seed(2021)
tf.compat.v1.set_random_seed(2020)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    conf = Configurator("CPR_LightGCN.properties", default_section="hyperparameters")
    gpu_id = str(conf["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    recommender = conf["recommender"]
    # num_thread = int(conf["rec.number.thread"])

    # if Tool.get_available_gpus(gpu_id):
    #     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    dataset = Dataset(conf)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = conf["gpu_mem"]
    
    for_max = 1
    dir_path = "./graph"
    write_ep = 100
    assert os.path.isdir(dir_path)
    for for_count in range(for_max):

        with tf.compat.v1.Session(config=config) as sess:
            if importlib.util.find_spec("model.general_recommender." + recommender) is not None:
                my_module = importlib.import_module("model.general_recommender." + recommender)
            
            elif importlib.util.find_spec("model.social_recommender." + recommender) is not None:
            
                my_module = importlib.import_module("model.social_recommender." + recommender)
            
            else:
                my_module = importlib.import_module("model.sequential_recommender." + recommender)
        
            #print("after eagerly:",tf.executing_eagerly())
            MyClass = getattr(my_module, recommender)
            model = ""
            model = MyClass(sess, dataset, conf)

            model.build_graph()
            sess.run(tf.compat.v1.global_variables_initializer())
            if for_max == 1:
                model.train_model(dir_path, write_ep)
            else:
                model.train_model(dir_path, write_ep, for_count)

import statistics
import math

import os
import pandas as pd
import uuid
import datetime
import faiss
#from multiprocessing import Pool
import numpy as np
import time

def save_exp_record(model_name, dataset_pair, test_mode, top_ks, total_rec, total_ndcg, count, save_dir, save_name, output_file):
    uuid_str = uuid.uuid4().hex
    record_row_save_path = os.path.join(save_dir, save_name +'_' +uuid_str+'.csv')
    txt_contents = []
    record_row = {}
    record_row['model_name'] = model_name
    record_row['dataset_pair'] = dataset_pair
    record_row['test_mode'] = test_mode
    record_row['time_stamp'] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    record_row['save_path'] = record_row_save_path
    for idx, k in enumerate(top_ks):
        _recall = total_rec[idx]/count
        _ndcg   = total_ndcg[idx]/count
        _content = [
               '\n--------------------------------',
               f'\n recall@{k}: ',
                str(_recall),
               f'\n NDCG@{k}: ',
                str(_ndcg)]
        txt_contents.append(_content)
        record_row[f'recall@{k}'] = _recall
        record_row[f'NDCG@{k}'] = _ndcg
    
    record_row = pd.DataFrame([record_row]) 
    record_row.to_csv(record_row_save_path, index=False)
    
    print("Start writing file...")
    with open(output_file, 'w') as fw:
        fw.writelines(['=================================\n',
                '\n evaluated users: ',
                str(count)])
        for _content in txt_contents:
            fw.writelines(_content)
    print('Finished!')

def rank_and_score(testing_users, top_ks, user_emb, testing_users_rec_dict, item_graph_df, tar_test_df, n_worker, uid_u, uid_i):
    st = time.time()
    d=100
    count = 0
    total_rec=[0, 0, 0, 0, 0]
    total_ndcg=[0, 0, 0, 0, 0]
    k_max = max(top_ks)
    for user in testing_users:
        count+=1
        user_emb_vec = np.array(list(user_emb[user]))
        user_emb_vec_m = np.matrix(user_emb_vec)
        user_rec_pool = testing_users_rec_dict[user]
    
        _tmp_df = item_graph_df[item_graph_df['node_id'].isin(user_rec_pool)]
        item_emb_vec = np.concatenate(list(_tmp_df['embed']), axis = -1).T
        item_emb_vec = item_emb_vec.copy(order='C')
        
        item_key = np.array(list(_tmp_df['node_id']))
    
        index = faiss.IndexFlatIP(d)
        index.add(item_emb_vec)
        D, I = index.search(user_emb_vec_m, k_max)
        recomm_list = item_key[I][0]
    
        if count%1000 == 0:
            print("{} users counted.".format(count))
    
        # ground truth
        test_data = list(tar_test_df[tar_test_df[uid_u] == user][uid_i])
    
        for k in range(len(top_ks)):
            recomm_k = recomm_list[:top_ks[k]]
            total_rec[k]+=calculate_Recall(test_data, recomm_k)
            total_ndcg[k]+=calculate_NDCG(test_data, recomm_k)
    
    total_rec=np.array(total_rec)
    total_ndcg=np.array(total_ndcg)
    print('Done counting.', time.time() - st)
   
    return total_rec, total_ndcg, count

def calculate_Recall(active_watching_log, topk_program):
    unique_played_amount = len(set(active_watching_log))
    hit = 0

    for program in topk_program:
        if program in active_watching_log:
            hit += 1
            
    if unique_played_amount == 0:
        return 0
    else:
        return hit / unique_played_amount

def calculate_Precision(active_watching_log, topk_program):
    recommend_amount = len(topk_program)
    hit = 0
    for program in topk_program:
        if program in active_watching_log:
            hit += 1
    return hit / recommend_amount
    
def calculate_NDCG(active_watching_log, topk_program):
    dcg = 0
    idcg = 0
    ideal_length = min(len(active_watching_log), len(topk_program))
    #dcg
    for i in range(len(topk_program)):
        if topk_program[i] in active_watching_log:
            dcg += (1/math.log2(i+2))
    #idcg
    for i in range(ideal_length):
        idcg += (1/math.log2(i+2))
    
    if idcg == 0:
        return 0
    else:
        return float(dcg/idcg)


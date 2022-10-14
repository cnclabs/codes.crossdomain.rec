import os
import pandas as pd
import uuid
import datetime

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
    

import os
import pandas as pd
import argparse

def get_org_id_remap_id_df(df, node_type):
    assert node_type in ['user', 'item'], 'node_type must be either user or item'
    _df = pd.DataFrame({'org_id':df[node_type].unique()})
    _df['remap_id'] = _df.index
    
    return _df

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_data_path', type=str)
    parser.add_argument('--lgn_input_dir', type=str)
    parser.add_argument('--save_file_name', type=str)
    args=parser.parse_args()
    print(args)

    print(f"Processing...{args.save_file_name}")
    df = pd.read_csv(args.input_data_path, names=['user', 'item', 'w'], sep='\t')
    user_id_map = get_org_id_remap_id_df(df, 'user')
    item_id_map = get_org_id_remap_id_df(df, 'item') 
    df['user_id'] = df['user'].map(user_id_map.set_index('org_id')['remap_id'])
    df['item_id'] = df['item'].map(item_id_map.set_index('org_id')['remap_id'])
    user_items_df = df.groupby('user_id')['item_id'].apply(list).apply(lambda col: str(col))
    user_items_df = pd.DataFrame(user_items_df)
    user_items_df['item_id'] = user_items_df['item_id'].str.replace('[\[,\]]', '')

    if not os.path.exists(args.lgn_input_dir):
        os.makedirs(args.lgn_input_dir)

    save_path = os.path.join(args.lgn_input_dir, args.save_file_name+'_train_input.txt')
    user_items_df.to_csv(save_path, sep=' ', header=None)

    save_path = os.path.join(args.lgn_input_dir, args.save_file_name+'_user_id_map.txt')
    user_id_map.to_csv(save_path, sep=' ', index=False) 

    save_path = os.path.join(args.lgn_input_dir, args.save_file_name+'_item_id_map.txt')
    item_id_map.to_csv(save_path, sep=' ', index=False) 
    print(f"Done {args.save_file_name}!")

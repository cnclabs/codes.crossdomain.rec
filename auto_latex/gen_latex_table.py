import os
import pandas as pd
import numpy as np
import argparse

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--exp_record_dir', type=str)
    args=parser.parse_args()

    select_columns = ['test_mode','model_name', 'dataset_pair', 'time_stamp', 'recall@10', 'NDCG@10']
    exp_dir = args.exp_record_dir
    fns = os.listdir(exp_dir)

    row_list = []
    for fn in fns:
        if '.csv' in fn:
            path = os.path.join(exp_dir, fn)
            row = pd.read_csv(path)
            row_list.append(row)
    result_df = pd.concat(row_list)

    mode_list = ['target',
            'shared',
            'cold']
    model_list = ['bpr',
                  'bpr_s',
                  'lgn_lil',
                   'lgn_big',
                 'emcdr',
                 'bitgcf',
                 'cpr']
    dataset_list = ['hk_csjj',
                   'mt_b',
                   'spo_csj']
    metric_list = ['recall@10',
                  'NDCG@10']
    
    model_show_name_map = {
        'bpr': 'BPR',
        'bpr_s': 'BPR$^+$',
        'lgn_lil': 'LightGCN',
        'lgn_big': 'LightGCN$^+$',
        'emcdr': 'EMCDR',
        'bitgcf': 'Bi-TGCF',
        'cpr': 'CPR'
    }

    def gen_latex_table(mode_name, path, model_row_map):
        bpr = model_row_map['bpr']
        bpr_s = model_row_map['bpr_s']
        lgn_lil = model_row_map['lgn_lil']
        lgn_big = model_row_map['lgn_big']
        emcdr = model_row_map['emcdr']
        bitgcf = model_row_map['bitgcf']
        cpr = model_row_map['cpr']
        imp = model_row_map['imp']
        
        with open(path, 'w') as fw:
            contents = [
               r'\begin' + '{table}[t]\n',
               r'\begin' + '{center}\n',
               r'    \begin' + '{tabular}{l rrr rrr}\n',
               r'    \toprule'+ '\n',
               r'    & \multicolumn{2}{c}{\texttt{HK-CSJ}} & \multicolumn{2}{c}{\texttt{MT-B}} & \multicolumn{2}{c}{\texttt{SPO-CSJ}} \\' +'\n',
               r'    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}'+ '\n',
               r'    & HR@10 & NDCG@10 & HR@10 & NDCG@10 & HR@10 & NDCG@10\\' + '\n',
               r'    \midrule'+ '\n',
               f'    {bpr}\n',
               f'    {bpr_s}\n',
               f'    {lgn_lil}\n',
               f'    {lgn_big}\n',
               r'    \midrule'+ '\n',
               f'    {emcdr}\n',
               f'    {bitgcf}\n',
               f'    {cpr}\n',
               r'    \midrule'+ '\n',
               f'    {imp}\n',
               r'    \midrule'+ '\n',
               r'    \bottomrule' + '\n',
               r'    \end{tabular}' + '\n',
               '\n',
               r'\caption{' + f'Test users from {mode_name} users'+ r'}' + '\n',
               r'\label{' + f'tab:{mode_name}_table'+ r'}%' + '\n',
               r'\vspace*{-0.6cm}' + '\n',
               r'\end{center}' + '\n',
               r'\end{table}%' + '\n',
               ]
            fw.writelines(contents)

    cross = '{$^\dagger$}'
    _b = '\\textbf{'
    b_ = '}'
    
    for mode_name in mode_list:
        model_row_map = {}
        cpr_list = []
        strong_list = []
        print(f'==============[{mode_name}]==============')
        for model_name in model_list:
            show_name = model_show_name_map[model_name]
            model_row = f'{show_name}\t'
            
            # TODO
            clean_row = f'{model_name}\t'
            for dataset_name in dataset_list:
                for metric_name in metric_list:
                    _df1 = result_df[
                        (result_df['test_mode']==mode_name) \
                        & (result_df['model_name']==model_name) \
                        & (result_df['dataset_pair']==dataset_name) \
                    ]
    
                    _df2 = result_df[
                        (result_df['test_mode']==mode_name) \
                        & ~(result_df['model_name']=='cpr') \
                        & (result_df['dataset_pair']==dataset_name) \
                    ]
                    _max = max(_df2[metric_name].values)
    
    
                    score = _df1[metric_name].values
                    if len(score) == 1:
                        score = score[0]
                        if model_name == 'cpr':
                            model_row += f"& {_b}"
                            model_row += f'{score:.4f}{b_}'
                            cpr_list.append(score)
                            strong_list.append(_max)
                        else:
                            if score == _max:
                                model_row += f'& {cross}{score:.4f}'
                            else:
                                model_row += f'& {score:.4f}'
                        # TODO
                        clean_row += f'{score:.4f}\t'
                    else:
                        model_row += f'& - '
                        # TODO
                        clean_row += f'-\t'
    
            model_row += " \\"
            model_row += "\\"
            model_row_map[model_name] = model_row
            #print(model_row)
            print(clean_row)
        cpr_list = np.array(cpr_list)
        strong_list = np.array(strong_list)
        improve_list = (cpr_list - strong_list) / strong_list
    
        imp = 'Improv. '

        # TODO
        clean_imp = 'Improv. ' 
        for i in improve_list:
            imp+= f'& {i*100:.2f}\% '
            clean_imp+= f'{i*100:.2f}%\t'
        imp += "\\"
        model_row_map['imp'] = imp
        print(clean_imp) 
        path = f'./{mode_name}.txt'
        gen_latex_table(mode_name, path, model_row_map)
        


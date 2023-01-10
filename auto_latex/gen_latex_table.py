import scipy.stats as stats
import os
import pandas as pd
import numpy as np
import argparse

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
        
def get_best_baseline_score(result_df, mode_name, dataset_name, metric_name):
    _df2 = result_df[
        (result_df['test_mode']==mode_name) \
        & ~(result_df['model_name']=='cpr') \
        & (result_df['dataset_pair']==dataset_name) \
    ]
    _scores = _df2[metric_name].values
    if len(_scores) >= 1:
        _max = max(_scores)
    else:
        _max = -1
    best_baseline_model = _df2['model_name'].values[0]
    best_baseline_scores = _df2[_df2['model_name']==best_baseline_model][metric_name].values
    return _max, best_baseline_scores, best_baseline_model

def do_pair_ttest(A_set, B_set, type1_err_alpha = 0.05):
    _, pvalue = stats.shapiro(A_set)
    if pvalue <= type1_err_alpha:
        A_normal = False
    else:
        A_normal = True
    _, pvalue = stats.shapiro(B_set)
    if pvalue <= type1_err_alpha:
        B_normal = False
    else:
        B_normal = True
    assert (A_normal is True) and (B_normal is True), 'violate the normality assumption'
    
    _, pvalue = stats.ttest_rel(A_set, B_set)
    if pvalue <= type1_err_alpha:
        return True
    else:
        return False

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

    cross = '{$^\dagger$}'
    _b = '\\textbf{'
    b_ = '}'
    
    for mode_name in mode_list:
        latex_row_map = {}
        cpr_list = []
        strong_list = []
        print(f'==============[{mode_name}]==============')
        for model_name in model_list:
            show_name = model_show_name_map[model_name]
            model_row = f'{show_name}\t'
            # TODO
            show_on_terminal = f'{model_name}\t'
            for dataset_name in dataset_list:
                for metric_name in metric_list:
                    _max, ttest_B, _ = get_best_baseline_score(result_df, mode_name, dataset_name, metric_name)
                    
                    _df1 = result_df[
                        (result_df['test_mode']==mode_name) \
                        & (result_df['model_name']==model_name) \
                        & (result_df['dataset_pair']==dataset_name) \
                    ]
                    
                    scores = _df1[metric_name].values
                    if len(scores) >= 1:
                        # we only consider everyone's best score
                        score = max(scores)
                        if model_name == 'cpr':
                            #TODO:(katiyth) fix ad-hoc
                            x = min(len(scores), len(ttest_B))
                            print(len(scores), len(ttest_B), x)
                            significant = do_pair_ttest(scores[:x], ttest_B[:x])
                            # to bold-face our CPR
                            model_row += f"& {_b}"
                            if significant:
                                model_row += f'{score:.4f}*{b_}'
                            else:
                                model_row += f'{score:.4f}{b_}'
                            cpr_list.append(score)
                            strong_list.append(_max)
                            
                        else:
                            if score == _max:
                                # add cross for best baseline 
                                model_row += f'& {cross}{score:.4f}'
                            else:
                                model_row += f'& {score:.4f}'
                        # TODO
                        show_on_terminal += f'{score:.4f}\t'
                    
                    # there's no score for this model
                    else:
                        model_row += f'& - '
                        # TODO
                        show_on_terminal += f'-\t'
                    
            
            # the first one: to escape
            model_row += " \\"
            # the second one: to change line
            model_row += "\\"
            latex_row_map[model_name] = model_row
    #         print('model?:', model_row)
            print(show_on_terminal)
        cpr_list = np.array(cpr_list)
        strong_list = np.array(strong_list)
        print('\t', cpr_list, '(cpr scores)')
        print('\t', strong_list, '(best baseline score)')
        improve_list = (cpr_list - strong_list) / strong_list
    
        imp_row = 'Improv. '
    
        # TODO
        show_on_terminal_imp = 'Improv. ' 
        for i in improve_list:
            imp_row+= f'& {i*100:.2f}\% '
            show_on_terminal_imp+= f'{i*100:.2f}%\t'
        imp_row += r" \\"
        latex_row_map['imp'] = imp_row
        print(show_on_terminal_imp) 
        path = f'./{mode_name}.txt'
        gen_latex_table(mode_name, path, latex_row_map)

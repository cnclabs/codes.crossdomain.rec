'''
Created on Aug 19, 2016
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
__author__ = "xiangwang"
import os
import re
# from progress.bar import Bar


# class ProgressBar(Bar):
#     message = 'Loading'
#     fill = '#'
#     suffix = '%(percent).1f%% | ETA: %(eta)ds'

def txt2list(file_src):
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def pprint(_str,file):
    print(_str)
    print(_str,file=file)


def uni2str(unicode_str):
    return str(unicode_str.encode('ascii', 'ignore')).replace('\n', '').strip()

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def delMultiChar(inputString, chars):
    for ch in chars:
        inputString = inputString.replace(ch, '')
    return inputString

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def early_stopping(log_hr, best_hr,stopping_step , flag_step=100):
    # early stopping strategy:
    # assert expected_order in ['acc', 'dec']

    if log_hr >= best_hr :
        stopping_step = 0
        best_hr = log_hr
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_hr))
        should_stop = True
    else:
        should_stop = False
    return best_hr, stopping_step, should_stop

def search_index_from_file(string):
    p1 = re.compile(r'hit=[[](.*?)[]]', re.S)
    p2 = re.compile(r'ndcg=[[](.*?)[]]',re.S)
    p3 = re.compile(r'=(\d*\.\d*?) \+',re.S)
    p4 = re.compile(r'\+ (\d*\.\d*?)[],]',re.S)
    return re.findall(p1, string), re.findall(p2, string),re.findall(p3, string),re.findall(p4, string)


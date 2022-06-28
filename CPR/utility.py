import statistics
import math

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


"""
    Some handy functions for pytroch model training ...
"""
import torch
import math


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=params['lr'], weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr=params['lr'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['lr'],
                                        alpha=params['alpha'],
                                        momentum=params['momentum'])
    return optimizer


# added evaluation official metric

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

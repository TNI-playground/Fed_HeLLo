import torch
import random
import numpy as np
import copy
from utils.model_utils import model_clip
from collections import defaultdict

def average(args, global_model, local_updates):
    '''
    simple average
    '''
    model_update = {k: local_updates[0][k] *0.0 for k in local_updates[0].keys()}
    for i in range(len(local_updates)):
        model_update = {k: model_update[k] +  local_updates[i][k] for k in global_model.keys()}

    global_model = {k: global_model[k] +  model_update[k]/ len(local_updates) for k in global_model.keys()}

    return global_model

def average_lora(args, global_model, local_updates):
    '''
    simple average
    '''
    model_update = {k: local_updates[0][k] *0.0 for k in local_updates[0].keys()}
    for i in range(len(local_updates)):
        if args.peft == 'lora':
            model_update = {k: model_update[k] +  local_updates[i][k] for k in global_model.keys() if 'lora' in k}
        else:
            model_update = {k: model_update[k] +  local_updates[i][k] for k in global_model.keys()}
    if args.peft == 'lora':
        for k in global_model.keys():
            if 'lora' in k:
                global_model[k] = global_model[k].detach().cpu() +  model_update[k]/ len(local_updates)
    else:
        global_model = {k: global_model[k].detach().cpu() +  model_update[k]/ len(local_updates) for k in global_model.keys()}

    return global_model

def average_lora_depthfl(args, global_model, loc_updates):
    '''
    hetero average
    '''
    model_update_avg_dict = {}

    for k in global_model.keys():
        if 'lora' in k or 'classifier' in k:
            for loc_update in loc_updates:
                if k in loc_update:
                    if k in model_update_avg_dict:
                        model_update_avg_dict[k].append(loc_update[k])
                    else:
                        model_update_avg_dict[k] = []
                        model_update_avg_dict[k].append(loc_update[k])

    for k in global_model.keys():
        if k in model_update_avg_dict:
            global_model[k] = global_model[k].detach().cpu() +  sum(model_update_avg_dict[k]) / len(model_update_avg_dict[k])

    return global_model

def weighted_average_lora_depthfl(args, global_model, loc_updates, num_samples):
    '''
    weighted hetero average
    '''
    model_update_avg_dict = {}
    model_weights_cnt = {}
    model_weights_list = {}

    for k in global_model.keys():
        if 'lora' in k or 'classifier' in k: # classifier is not included
            for client_i, loc_update in enumerate(loc_updates):
                if k in loc_update:
                    if k in model_update_avg_dict:
                        model_update_avg_dict[k].append(loc_update[k])
                        model_weights_cnt[k] += num_samples[client_i]
                        model_weights_list[k].append(num_samples[client_i])
                    else:
                        model_update_avg_dict[k] = []
                        model_update_avg_dict[k].append(loc_update[k])
                        model_weights_cnt[k] = 0.0
                        model_weights_cnt[k] += num_samples[client_i]
                        model_weights_list[k] = []
                        model_weights_list[k].append(num_samples[client_i])

    for k in global_model.keys():
        if k in model_update_avg_dict:
            weight_based = model_weights_cnt[k]
            model_weights_list[k] = [item/weight_based for item in model_weights_list[k]]
            global_model[k] = global_model[k].detach().cpu() + sum([model*weight for model, weight in zip(model_update_avg_dict[k], model_weights_list[k])])

    return global_model

    '''
    hetero average
    '''
    update_weights = []
    for updates in loc_updates:
        svd_weights = []
        keys = list(updates.keys())
        for i in range(0, len(keys), 2):
            keyA = keys[i]
            keyB = keys[i+1]
            tensorA = updates[keyA]
            tensorB = updates[keyB]
            tensorSVD = torch.matmul(tensorB, tensorA)
            svd_weights.append(tensorSVD)
        svd_weights = torch.stack(svd_weights)
        frobenius_norm = np.linalg.norm(tensorSVD, 'fro')
        update_weights.append(frobenius_norm)
    update_weights = update_weights / np.sum(update_weights)
    args.logger.info(update_weights, main_process_only=True)
    for index, (u, w) in enumerate(zip(loc_updates, update_weights)):
        loc_updates[index] = {key: value * w for key, value in u.items()}

    model_update_avg_dict = {}
    for k in global_model.keys():
        if 'lora' in k or 'classifier' in k:
            for loc_update in loc_updates:
                if k in loc_update:
                    if k in model_update_avg_dict:
                        model_update_avg_dict[k].append(loc_update[k])
                    else:
                        model_update_avg_dict[k] = []
                        model_update_avg_dict[k].append(loc_update[k])

    for k in global_model.keys():
        if k in model_update_avg_dict:
            if 'lora_A' in k:
                for i in range(global_model[k].size(0)):
                    temp = []
                    for tensor in model_update_avg_dict[k]:
                        if i < tensor.size(0):
                            temp.append(tensor[i, :])
                    if temp:
                        # global_model[k][i, :] = global_model[k][i, :].detach().cpu() + torch.stack(temp).mean(dim=0)
                        global_model[k][i, :] = global_model[k][i, :].detach().cpu() + torch.stack(temp).sum(dim=0)
            elif 'lora_B' in k:
                for i in range(global_model[k].size(1)):
                    temp = []
                    for tensor in model_update_avg_dict[k]:
                        if i < tensor.size(1):
                            temp.append(tensor[:, i])
                    if temp:
                        # global_model[k][:, i] = global_model[k][:, i].detach().cpu() + torch.stack(temp).mean(dim=0)
                        global_model[k][:, i] = global_model[k][:, i].detach().cpu() + torch.stack(temp).sum(dim=0)
    return global_model


    if args.privc == 'dp_randk' or args.privc == 'dp_topk' or args.privc == 'randk' or args.privc == 'topk':
        # error compensate, clipping, privately compress and average
        # lr_g = args.global_lr * 1/(t+1)**0.5
        pris_local_updates =[]
        norm_para=torch.zeros(args.num_users).to(args.device)
        for i in idxs_users:
            if (t+1) == args.tau:
                local_models[i] = {k:local_models[i][k] - global_model[k] for k in global_model.keys()}
                local_models[i], norm_para[i] = model_clip(local_models[i], args.clip)
            else:
                local_models[i] = {k:local_models[i][k] - global_model[k] + errors[i][k] for k in global_model.keys()}
                local_models[i], norm_para[i] = model_clip(local_models[i], args.clip)
                            
            pris_update = private_com(local_models[i], args)
            pris_local_updates.append(pris_update)
        print('weight norm', sum(norm_para)/len(idxs_users))
        
        global_update = {k: global_model[k] *0.0 for k in global_model.keys()}
        for i in len(idxs_users):
            global_update = {k: global_update[k] +  pris_local_updates[i][k] for k in global_update.keys()}
        
        global_update = {k: global_update[k] +  pris_local_updates[i][k] for k in global_update.keys()}
        global_model = {k: global_model[k] + global_update[k] for k in global_model.keys()}
    else:
        exit('Error: unrecognized private compressor for pefed.') 
        # global_model = aggregation_avg(local_models, idxs_users)
    return global_model
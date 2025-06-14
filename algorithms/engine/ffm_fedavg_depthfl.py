import copy
import numpy as np
import time, math
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.data_pre_process import load_partition, DatasetSplit
from utils.model_utils import model_setup
from utils.log_utils import set_log_path
from test import test, test_vit, test_ledgar

from ..solver.local_solver import LocalUpdate
from ..solver.global_aggregator import average, average_lora, average_lora_depthfl, weighted_average_lora_depthfl
import gc
from fractions import Fraction
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
from dataclasses import dataclass
import random

def vit_collate_fn(examples):
    pixel_values = torch.stack([example[2] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "correct_answer_num"
        labels = [torch.tensor(int(feature[label_name])-1) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        flattened_features = [
            [{k: v[i] for k, v in feature.items() if k in ['input_ids', 'token_type_ids', 'attention_mask']} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

def ffm_fedavg_depthfl(args):
    ################################### hyperparameter setup ########################################
    args.logger.info("{:<50}".format("-" * 15 + " data setup " + "-" * 50)[0:60], main_process_only=True)
    args, dataset_train, dataset_test, dataset_val, dataset_public, dict_users, dataset_fim = load_partition(args)
    args.logger.info('length of dataset:{}'.format(len(dataset_train) + len(dataset_test)), main_process_only=True)
    args.logger.info('num. of training data:{}'.format(len(dataset_train)), main_process_only=True)
    args.logger.info('num. of testing data:{}'.format(len(dataset_test)), main_process_only=True)
    # print('num. of validation data:{}'.format(len(dataset_val)))
    # print('num. of public data:{}'.format(len(dataset_public)))
    args.logger.info('num. of users:{}'.format(len(dict_users)), main_process_only=True)
    sample_per_users = int(sum([ len(dict_users[i]) for i in range(len(dict_users))])/len(dict_users))
    args.logger.info('average num. of samples per user:{}'.format(sample_per_users), main_process_only=True)

    args.logger.info("{:<50}".format("-" * 15 + " log path " + "-" * 50)[0:60], main_process_only=True)
    if args.accelerator.is_local_main_process:
        writer = SummaryWriter(args.log_path)
    args.logger.info(args.log_path, main_process_only=True)

    args.logger.info("{:<50}".format("-" * 15 + " model setup " + "-" * 50)[0:60], main_process_only=True)
    args, net_glob, global_model, args.dim = model_setup(args)
    
    args.logger.info('model dim: '+str(args.dim), main_process_only=True)

    ###################################### model initialization ###########################
    t1 = time.time()
    args.logger.info("{:<50}".format("-" * 15 + " training... " + "-" * 50)[0:60], main_process_only=True)
    # initialize data loader for training and/or public dataset
    data_loader_list = []
    for i in range(args.num_users):
        dataset = DatasetSplit(dataset_train, dict_users[i], args)
        
        if 'vit' in args.model:
            ldr_train = DataLoader(dataset, shuffle=True, collate_fn=vit_collate_fn, batch_size=args.batch_size)
        elif 'sst2' in args.dataset or 'qqp' in args.dataset or 'qnli' in args.dataset or 'ledgar' in args.dataset:
            ldr_train = DataLoader(dataset, shuffle=True, collate_fn=args.data_collator, batch_size=args.batch_size)
        elif 'belebele' in args.dataset:
            ldr_train = DataLoader(dataset, shuffle=True, collate_fn=DataCollatorForMultipleChoice(tokenizer=args.tokenizer), batch_size=args.batch_size)

        data_loader_list.append(ldr_train)

    # heterogenity
    group_num = len(args.heterogeneous_group)
    group_cnt = []
    for g in range(group_num):
        if g == (group_num - 1):
            remind_cnt = args.num_users
            for c in group_cnt:
                remind_cnt -= c
            group_cnt.append(remind_cnt)
        else:    
            group_cnt.append(int(args.num_users * float(Fraction(args.heterogeneous_group[g]))))

    args.user_groupid_list = []
    for id, c in enumerate(group_cnt):
        args.user_groupid_list += [id] * c
    # block ids for each clients
    if hasattr(args, 'heterogeneous_group0_lora'):
        if isinstance(getattr(args, 'heterogeneous_group0_lora'), int):
            args.block_ids_list = []
            for id in args.user_groupid_list:
                args.block_ids_list.append(sorted(random.sample(range(args.lora_layer), getattr(args, 'heterogeneous_group'+str(id)+'_lora'))))
        elif isinstance(getattr(args, 'heterogeneous_group0_lora'), list):
            args.block_ids_list = []
            if hasattr(args, 'probability_structure'):
                id_list = set(args.user_groupid_list)
                prob_distribution = []
                for k in id_list:
                    prob_distribution.extend(getattr(args, 'heterogeneous_group'+str(k)+'_lora'))
                total_elements = len(prob_distribution)
                prob_distribution = [count/total_elements for _, count in Counter(sorted(prob_distribution)).items()]
                for id in args.user_groupid_list:
                    args.block_ids_list.append(sorted(np.random.choice(np.arange(len(getattr(args, 'heterogeneous_group'+str(list(id_list)[-1])+'_lora'))),
                                            size=len(getattr(args, 'heterogeneous_group'+str(id)+'_lora')),
                                            p=prob_distribution,
                                            replace=False)))
            else:
                for id in args.user_groupid_list:
                    args.block_ids_list.append(getattr(args, 'heterogeneous_group'+str(id)+'_lora'))

    best_test_acc = 0.0
    best_test_f1 = 0.0
    best_test_macro_f1 = 0.0
    best_test_micro_f1 = 0.0
    metric_keys = {'Accuracy': 0, 'F1': 0, 'Macro_F1': 0, "Micro_F1": 0}

    for t in range(args.round):
        args.logger.info('Round: ' + str(t) + '/' + str(args.round), main_process_only=True)
        ## learning rate decaying
        if t+1 % args.lr_step_size == 0: # shakespear: 50, femnist: x, other: 1
            args.local_lr = args.local_lr * args.decay_weight
        
        ############################################################# FedAvg ##########################################
        ## user selection
        selected_idxs = list(np.random.choice(range(args.num_users), args.num_selected_users, replace=False))
        ## copy global model
        net_glob.train()
        ## local training
        local_solver = LocalUpdate(args=args)
        local_models, local_losses, local_updates, delta_norms, num_samples = [], [], [], [], []
        for num_index, i in enumerate(selected_idxs):
            if args.peft == 'lora':
                local_model, local_loss, no_weight_lora =  local_solver.lora_tuning(model=copy.deepcopy(net_glob),
                                                                                    ldr_train=data_loader_list[i],
                                                                                    args=args,
                                                                                    client_index=num_index,
                                                                                    client_real_id=i,
                                                                                    round=t,
                                                                                    hete_group_id=args.user_groupid_list[i])
            if local_loss:
                local_losses.append(local_loss)
            # compute model update
            model_update = {}
            if args.peft == 'lora':
                for k in global_model.keys():
                    if 'lora' in k: # no classifier
                        if int(re.findall(r"\d+", k)[0]) not in no_weight_lora:
                            model_update[k] = local_model[k].detach().cpu() - global_model[k].detach().cpu() 
            else:
                model_update = {k: local_model[k].detach().cpu() - global_model[k].detach().cpu() for k in global_model.keys()}
            # compute model update norm
            norm_updates = []
            for k in model_update.keys():
                norm_updates.append(torch.flatten(model_update[k]))
            if len(norm_updates) > 0:
                delta_norm = torch.norm(torch.cat(norm_updates))
            else:
                delta_norm = None
            if delta_norm:
                delta_norms.append(delta_norm)
            local_updates.append(model_update)
            num_samples.append(len(data_loader_list[i]))

        if len(local_updates) == 0:
            args.logger.info('The number of trainable client is 0, skip the round for average')
            continue
        # metrics
        if len(delta_norms) > 0:
            norm = torch.median(torch.stack(delta_norms)).cpu()
        else:
            norm = 100
        if len(local_losses) > 0:
            train_loss = sum(local_losses) / len(local_losses)
        else:
            train_loss = 100
        if args.accelerator.is_local_main_process:
            writer.add_scalar('norm', norm, t)
            writer.add_scalar('train_loss', train_loss, t)

        if hasattr(args, 'aggregation'):
            if args.aggregation ==  'weighted_average':
                global_model = weighted_average_lora_depthfl(args, global_model, local_updates, num_samples)
        else:
            global_model = average_lora_depthfl(args, global_model, local_updates)

        # test global model on server side   
        net_glob.load_state_dict(global_model)
        net_glob.eval()
        if 'vit' in args.model:
            test_acc, test_loss = test_vit(copy.deepcopy(net_glob), dataset_test, args, t)
            # metrics
            if args.accelerator.is_local_main_process:
                writer.add_scalar('test_acc', test_acc, t)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    metric_keys['Accuracy'] = 1
            args.logger.info('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
                format(t, train_loss, norm, test_acc), main_process_only=True)
        elif 'bert' in args.model:
            if 'sst2' in args.dataset:
                test_acc, test_loss = test_sst2(copy.deepcopy(net_glob), dataset_test, args, t)
                # metrics
                if args.accelerator.is_local_main_process:
                    writer.add_scalar('test_acc', test_acc, t)
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        metric_keys['Accuracy'] = 1
                args.logger.info('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
                    format(t, train_loss, norm, test_acc), main_process_only=True)
            elif 'qqp' in args.dataset:
                test_f1, test_acc, test_loss = test_qqp(copy.deepcopy(net_glob), dataset_test, args, t)
                # metrics
                if args.accelerator.is_local_main_process:
                    writer.add_scalar('test_acc', test_acc, t)
                    writer.add_scalar('test_f1', test_f1, t)

                    if test_f1 > best_test_f1:
                        best_test_f1 = test_f1
                        # best performance based on f1
                        best_test_acc = test_acc
                        metric_keys['Accuracy'] = 1
                        metric_keys['F1'] = 1
                args.logger.info('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_f1 = {:.3f}, test_acc = {:.3f}'.
                    format(t, train_loss, norm, test_f1, test_acc), main_process_only=True)
            elif 'qnli' in args.dataset:
                test_acc, test_loss = test_qnli(copy.deepcopy(net_glob), dataset_test, args, t)
                # metrics
                if args.accelerator.is_local_main_process:
                    writer.add_scalar('test_acc', test_acc, t)
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        metric_keys['Accuracy'] = 1
                args.logger.info('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
                    format(t, train_loss, norm, test_acc), main_process_only=True)
            elif 'ledgar' in args.dataset:
                test_macro_f1, test_micro_f1, test_loss = test_ledgar(copy.deepcopy(net_glob), dataset_test, args, t)
                # metrics
                if args.accelerator.is_local_main_process:
                    writer.add_scalar('test_macro_f1', test_macro_f1, t)
                    writer.add_scalar('test_micro_f1', test_micro_f1, t)
                    if test_micro_f1 > best_test_micro_f1:
                        best_test_micro_f1 = test_micro_f1
                        best_test_macro_f1 = test_macro_f1
                        metric_keys['Macro_F1'] = 1
                        metric_keys['Micro_F1'] = 1

                args.logger.info('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_macro_f1 = {:.3f}, test_micro_f1 = {:.3f}'.
                    format(t, train_loss, norm, test_macro_f1, test_micro_f1), main_process_only=True)
            elif 'belebele' in args.dataset:
                test_acc, test_loss = test_belebele(copy.deepcopy(net_glob), dataset_test, args, t)
                # metrics
                if args.accelerator.is_local_main_process:
                    writer.add_scalar('test_acc', test_acc, t)
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        metric_keys['Accuracy'] = 1
                args.logger.info('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
                    format(t, train_loss, norm, test_acc), main_process_only=True)
        else:
            test_acc, test_loss = test(copy.deepcopy(net_glob), dataset_test, args)
            # metrics
            if args.accelerator.is_local_main_process:
                writer.add_scalar('test_acc', test_acc, t)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    metric_keys['Accuracy'] = 1
            args.logger.info('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
                format(t, train_loss, norm, test_acc), main_process_only=True)

        args.accelerator.wait_for_everyone()

    return (best_test_acc, best_test_f1, best_test_macro_f1, best_test_micro_f1), metric_keys
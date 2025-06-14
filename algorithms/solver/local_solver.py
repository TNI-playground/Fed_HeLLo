import torch
from torch import nn
import copy

import numpy as np
from tqdm import tqdm
import gc
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import time

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

class LocalUpdate(object):
    def __init__(self, args):
        self.args = args
        if args.data_type == 'image':
            self.loss_func = nn.CrossEntropyLoss()
        elif args.data_type == 'text':
            self.loss_func = nn.CrossEntropyLoss()
        elif args.data_type == 'sentiment':
            self.loss_func = nn.NLLLoss()

    def lora_tuning(self, model, ldr_train, args, client_index, client_real_id, round, hete_group_id):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

        if isinstance(getattr(args, 'heterogeneous_group'+str(hete_group_id)+'_lora'), list):
            no_weight_lora = list(set(range(args.lora_layer)) - set(getattr(args, 'heterogeneous_group'+str(hete_group_id)+'_lora')))
        elif isinstance(getattr(args, 'heterogeneous_group'+str(hete_group_id)+'_lora'), int):
            no_weight_lora = list(set(range(args.lora_layer)) - set(args.block_ids_list[client_real_id]))

        # early stop for exclusive training
        if len(no_weight_lora) == args.lora_layer:
            return model.state_dict(), None, no_weight_lora

        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(str(nd) in n for nd in no_weight_lora)]
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(str(nd) in n for nd in no_weight_lora)],
                    'lr': 0.0
                }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.local_lr)
        # # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, ldr_train)
        total_loss = []
        for t_au in range(self.args.tau):
            with accelerator.accumulate(model):
                for step, batch in tqdm(enumerate(train_dataloader), desc='Local Training Client '+str(client_index)+' Tau: '+str(t_au), total=len(train_dataloader), disable=(not accelerator.is_local_main_process)):
                    optimizer.zero_grad()
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    # lr_scheduler.step()
                    if accelerator.is_local_main_process:
                        total_loss.append(loss.detach().float().cpu())
                    accelerator.wait_for_everyone()
                    
        args.logger.info(f'Total local training loss is: {np.mean(total_loss)}', main_process_only=True)

        # optimizer.zero_grad()
        return accelerator.unwrap_model(model).state_dict(), np.mean(total_loss), no_weight_lora

    def lora_tuning_extradata_ft(self, model, ldr_train, args, client_index=0):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

        no_weight_lora = []

        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(str(nd) in n for nd in no_weight_lora)]
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(str(nd) in n for nd in no_weight_lora)],
                    'lr': 0.0
                }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.local_lr)
        # # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, ldr_train)
        total_loss = []
        for t_au in range(self.args.tau):
            with accelerator.accumulate(model):
                for step, batch in tqdm(enumerate(train_dataloader), desc='Local Training Client '+str(client_index)+' Tau: '+str(t_au), total=len(train_dataloader), disable=(not accelerator.is_local_main_process)):
                    optimizer.zero_grad()
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    # lr_scheduler.step()
                    if accelerator.is_local_main_process:
                        total_loss.append(loss.detach().float().cpu())
                    accelerator.wait_for_everyone()
                    
        args.logger.info(f'Total local training loss is: {np.mean(total_loss)}', main_process_only=True)

        # optimizer.zero_grad()
        return accelerator.unwrap_model(model).state_dict(), np.mean(total_loss), no_weight_lora

    def lora_tuning_extradata(self, model, ldr_train, args):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        
        no_weight_lora = []

        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(str(nd) in n for nd in no_weight_lora)]
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(str(nd) in n for nd in no_weight_lora)],
                    'lr': 0.0
                }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.local_lr)
        # # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, ldr_train)
        total_loss = []
        for t_au in range(self.args.tau):
            with accelerator.accumulate(model):
                for step, batch in tqdm(enumerate(train_dataloader), desc='Extra data finetuning', total=len(train_dataloader), disable=(not accelerator.is_local_main_process)):
                    optimizer.zero_grad()
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    # lr_scheduler.step()
                    if accelerator.is_local_main_process:
                        total_loss.append(loss.detach().float().cpu())
        args.logger.info(f'Total global training loss is: {np.mean(total_loss)}', main_process_only=True)
        return accelerator.unwrap_model(model)

    def lora_tuning_rank(self, model, ldr_train, args, client_index, client_real_id, round, hete_group_id):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

        no_weight_lora = []

        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(str(nd) in n for nd in no_weight_lora)]
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(str(nd) in n for nd in no_weight_lora)],
                    'lr': 0.0
                }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.local_lr)
        # # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, ldr_train)
        total_loss = []
        for t_au in range(self.args.tau):
            with accelerator.accumulate(model):
                for step, batch in tqdm(enumerate(train_dataloader), desc='Local Training Client '+str(client_index)+' Tau: '+str(t_au), total=len(train_dataloader), disable=(not accelerator.is_local_main_process)):
                    optimizer.zero_grad()

                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    # lr_scheduler.step()
                    if accelerator.is_local_main_process:
                        total_loss.append(loss.detach().float().cpu())
        args.logger.info(f'Total local training loss is: {np.mean(total_loss)}', main_process_only=True)

        # optimizer.zero_grad()
        return accelerator.unwrap_model(model).state_dict(), np.mean(total_loss), no_weight_lora

    def local_sgd(self, net, ldr_train, topk_model=None):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.local_lr)
        epoch_loss = []
        net.train()
        for _ in range(self.args.tau):
            for _, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
        w_new = copy.deepcopy(net.state_dict())
        return w_new, sum(epoch_loss) / len(epoch_loss)
    
    def local_sgd_mome(self, net, ldr_train, topk_model=None):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.local_lr, momentum=self.args.local_momentum)
        epoch_loss = []
        net.train()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        net, optimizer, ldr_train = accelerator.prepare(net, optimizer, ldr_train)
        for _ in range(self.args.tau):
            if self.args.dataset == 'femnist':
                for step, batch in tqdm(enumerate(ldr_train), desc='Local Training Client ', disable=(not accelerator.is_local_main_process)):
                    images, labels = batch['pixel_values'].to(self.args.device), batch['labels'].to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item())
            else:
                for _, (images, labels) in tqdm(enumerate(ldr_train), desc='Local Training Client ', disable=(not accelerator.is_local_main_process)):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item())
        w_new = copy.deepcopy(net.state_dict())
        return w_new, sum(epoch_loss) / len(epoch_loss)
    
    def local_Adamw_PEFT(self, net, ldr_train, topk_model=None):
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.local_lr)
        elif self.args.optimizer == 'sgdm':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.local_lr, momentum=self.args.local_momentum)
        elif self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(net.parameters(), lr=self.args.local_lr)
        epoch_loss = []
        net.train()
        for _ in range(self.args.tau):
            for _, batch in enumerate(ldr_train):
                # images, labels = images.to(self.args.device), labels.to(self.args.device)
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                net.zero_grad()
                # log_probs = net(images)
                # loss = self.loss_func(log_probs, labels)
                outputs = net(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.detach().float())
                # torch.cuda.empty_cache()
        w_new = copy.deepcopy(net.state_dict())
        return w_new, sum(epoch_loss) / len(epoch_loss)

    def local_sgd_adamw_ffm(self, net, ldr_train, topk_model=None):
        opt_model = net
        decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.local_lr, betas=(self.args.adam_beta1, self.args.adam_beta2))
        epoch_loss = []
        net.train()
        for _ in range(self.args.tau):
            for _, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
        w_new = copy.deepcopy(net.state_dict())
        return w_new, sum(epoch_loss) / len(epoch_loss)


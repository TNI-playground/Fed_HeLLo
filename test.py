import torch
from torch import nn
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
import evaluate
from datasets import load_metric
from sklearn.metrics import f1_score
import numpy as np

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
from dataclasses import dataclass

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    if 'label' in examples[0]:
        labels = torch.tensor([example["label"] for example in examples])
    else:
        labels = torch.tensor([example["fine_label"] for example in examples])
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

def test_vit(model, dataset, args, t):
    # Eval for each round
    metric = evaluate.load("accuracy")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    model.eval()
    eval_dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=args.batch_size)
    eval_dataloader, model = accelerator.prepare(eval_dataloader, model)

    for step, batch in tqdm(enumerate(eval_dataloader), desc='Evaulating Round: '+str(t), total=len(eval_dataloader), disable=(not accelerator.is_local_main_process)):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    cur_acc = eval_metric['accuracy']
    model = accelerator.unwrap_model(model)
    args.logger.info(eval_metric, main_process_only=True)

    return cur_acc, None

def test_ledgar(model, dataset, args, t):
    # Eval for each round
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    model.eval()
    eval_dataloader = DataLoader(dataset, collate_fn=args.data_collator, batch_size=args.batch_size)

    eval_dataloader, model = accelerator.prepare(eval_dataloader, model)

    predictions_list = []
    references_list = []
    for step, batch in tqdm(enumerate(eval_dataloader), desc='Evaulating Round: '+str(t), total=len(eval_dataloader), disable=(not accelerator.is_local_main_process)):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
        predictions_list.append(predictions.cpu())
        references_list.append(references.cpu())

    cur_macro_f1 = f1_score(y_true=np.concatenate(references_list, axis=0), y_pred=np.concatenate(predictions_list, axis=0), average='macro', zero_division=0)
    cur_micro_f1 = f1_score(y_true=np.concatenate(references_list, axis=0), y_pred=np.concatenate(predictions_list, axis=0), average='micro', zero_division=0)
    model = accelerator.unwrap_model(model)
    args.logger.info({'macro_f1': cur_macro_f1, 'micro_f1': cur_micro_f1}, main_process_only=True)

    return cur_macro_f1, cur_micro_f1, None

def test(net_g, dataset, args):
    net_g = copy.deepcopy(net_g).to(args.device)
    loss_func = nn.CrossEntropyLoss()
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    if args.dataset == 'femnist':
        data_loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=args.batch_size)
    else:
        data_loader = DataLoader(dataset, batch_size=args.test_batch_size)

    # attack init
    attack_flag = False
    if hasattr(args, 'attack'):
        if args.attack != 'None':
            attack_flag = True
    if attack_flag:
        attack_ratio = args.num_attackers / args.num_selected_users
        poison_data_iteration = int(attack_ratio * len(data_loader))

    for index, batch in enumerate(data_loader):
        if args.dataset == 'femnist':
            data, target = batch['pixel_values'], batch['labels']
        else:
            data, target = batch
        ################## <<< Attack Point 3: trigger for all data during testing
        if attack_flag:
            if index < poison_data_iteration:
                if args.attack == 'dba':
                    trigger_cnt = index % args.trigger_num
                    data, target = dba_poison(data, target, args, trigger_cnt, evaluation=True)
                elif args.attack == 'edge':
                    data, target = edge_poison(data, target, args, evaluation=True)
                
        data, target = data.to(args.device), target.to(args.device)
        # print(data[0], target[0])
        
        log_probs = net_g(data)

        test_loss += loss_func(log_probs, target).item()
        # get the index of the max log-probability
        # print(log_probs[0])
        # print(torch.max(log_probs[0], -2))
        # exit()
        if args.dataset == 'shakespeare':
            _, predicted = torch.max(log_probs, -2)
            # correct += predicted[:,-1].eq(target[:,-1]).sum()
            correct += predicted.eq(target).sum()/target.shape[1]
            # print(predicted[:,-1].eq(target[:,-1]))
        else:
            _, predicted = torch.max(log_probs, -1)
            correct += predicted.eq(target).sum()

    test_loss /= len(dataset)
    # print(correct, len(datatest))
    accuracy = 100.00 * correct.item() / len(dataset)
    return accuracy, test_loss

def extract_results(tokenized_data, args):
    target = tokenized_data['labels'].clone()
    target[target==-100] = 0
    untokenized_target = args.tokenizer.batch_decode(target)
    target_list = []
    for item in untokenized_target:
        item = item.replace('<unk>', '').replace('<pad>', '')
        target_list.append(item)
    return target_list
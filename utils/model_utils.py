from model.cnn import CNNFmnist, CNNFEmnist, CNNSvhn, CNNCifar
from model.mlp import MLP
from model.recurrent import CharLSTM, RNN_FedShakespeare
from model.vgg import vgg19_bn
from model.resnet import ResNet9FashionMNIST, ResNet18, ReducedResNet18, \
                         ResNet34, ResNet50, ResNet101, ResNet152, \
                         CIFARResNet20, SVHNResNet20
# from model.bert import Bert
from transformers import AutoModelForCausalLM, BloomForSequenceClassification, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
# from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
# from peft import PeftModel, PeftConfig
from peft import LoraConfig, get_peft_model

import torch
import torch.nn as nn
import copy

from transformers import AutoModelForImageClassification, AutoModelForSequenceClassification, AutoModelForMultipleChoice
from peft import LoraConfig, get_peft_model

################################### model setup ########################################
def model_setup(args):
    if args.model == 'bert-base-uncased':
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_classes)
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none"
        )
        net_glob = get_peft_model(model, config)
        net_glob.to(args.device)
    elif args.model == 'google/vit-base-patch16-224-in21k':
        model = AutoModelForImageClassification.from_pretrained(
            args.model,
            label2id=args.label2id,
            id2label=args.id2label,
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )

        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        net_glob = get_peft_model(model, config)
        net_glob.to(args.device)
    else:
        exit('Error: unrecognized model')

    global_model = copy.deepcopy(net_glob.state_dict())
    return args, net_glob, global_model, model_dim(global_model)

def model_dim(model):
    '''
    compute model dimension
    '''
    flat = [torch.flatten(model[k]) for k in model.keys()]
    s = 0
    for p in flat: 
        s += p.shape[0]
    return s


def model_clip(model, clip):
    '''
    clip model update
    '''
    model_norm=[]
    for k in model.keys():
        if 'num_batches_tracked' in k or 'running_mean' in k or 'running_var' in k:
            continue
        model_norm.append(torch.norm(model[k]))
        
    total_norm = torch.norm(torch.stack(model_norm))
    clip_coef = clip / (total_norm + 1e-8)
    if clip_coef < 1:
        for k in model.keys():
            if 'num_batches_tracked' in k or 'running_mean' in k or 'running_var' in k:
                continue
            model[k] = model[k] * clip_coef
    return model, total_norm

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

def get_trainable_values(net,mydevice=None):
    ' return trainable parameter values as a vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    paramlist=list(trainable) 
    N=0
    for params in paramlist:
        N+=params.numel()
    if mydevice:
        X=torch.empty(N,dtype=torch.float).to(mydevice)
    else:
        X=torch.empty(N,dtype=torch.float)
    X.fill_(0.0)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            X[offset:offset+numel].copy_(params.data.view_as(X[offset:offset+numel].data))
        offset+=numel

    return X

def put_trainable_values(net,X):
    ' replace trainable parameter values by the given vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    paramlist=list(trainable)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset+numel].data.view_as(params.data))
        offset+=numel

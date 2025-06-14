import torch, random, argparse, os
from algorithms.engine.ffm_fedavg_depthfl import ffm_fedavg_depthfl
from algorithms.engine.ffm_fedavg_depthffm import ffm_fedavg_depthffm
from algorithms.engine.ffm_fedavg_depthffm_fim import ffm_fedavg_depthffm_fim
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from utils.log_utils import set_log_path

from mmengine.config import Config
import logging
from datetime import datetime
import sys
import numpy as np
import copy


os.environ['NCCL_P2P_DISABLE']='1'

def merge_config(config, args):
    for arg in vars(args):
        setattr(config, arg, getattr(args, arg))
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',
                        type=list,
                        default=0,
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="seed")
    parser.add_argument('--repeat', type=int, default=1, help='repeat index')
    parser.add_argument('--vis_label_diversity', type=int, default=0, help='vis_label_diversity')
    parser.add_argument('--freeze_datasplit', type=int, default=0, help='freeze to save dict_users.pik or not')
    # Modify following configure to test with different settings 
    parser.add_argument('--config_name', type=str,
                        default='experiments/cifar100_vit_lora/6_6_6/image_cifar100_vit_fedavg_depthfl-6_6_6_iid.yaml',
                        help='method configuration')

    sys.setrecursionlimit(10000)

    meta_args = parser.parse_args()
    config_path = os.path.join('config/', meta_args.config_name)
    config = Config.fromfile(config_path)
    meta_args = merge_config(config, meta_args)

    # meta_args.device = torch.device('cuda:{}'.format(meta_args.gpu) if torch.cuda.is_available() and meta_args.gpu != -1 else 'cpu')
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    meta_args.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    meta_args.device = meta_args.accelerator.device
    logging.basicConfig(level=logging.INFO)
    meta_args.logger = get_logger(__name__)
    meta_args.log_path = set_log_path(meta_args)
    file_name = os.path.join(meta_args.log_path, "exp_log.txt")
    meta_args.model_save_path = os.path.join(meta_args.log_path, 'checkpoints')
    
    if meta_args.accelerator.is_local_main_process:
        f = open(file_name, 'a+')  # open file in append mode
        f.write(meta_args.log_path+'\n')
        f.close()
        file_handler = logging.FileHandler(file_name)
        meta_args.logger.logger.addHandler(file_handler)

    # for reproducibility
    score_box = []
    poisoned_ratio_box = []
    for r in range(meta_args.repeat):
        args = copy.deepcopy(meta_args)
        
        if meta_args.accelerator.is_local_main_process:
            args.logger.info('############ Case '+ str(r) + ' ############', main_process_only=True)
        torch.manual_seed(args.seed+r)
        # torch.cuda.manual_seed(args.seed+args.repeat) # avoid
        np.random.seed(args.seed+r)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        if args.method == "ffm_fedavg":
            if args.model_heterogeneity == 'depthfl':
                best_result, metric_keys = ffm_fedavg_depthfl(args)
            elif args.model_heterogeneity == 'depthffm':
                best_result, metric_keys = ffm_fedavg_depthffm(args)     
            elif args.model_heterogeneity == 'depthffm_fim':
                best_result, metric_keys = ffm_fedavg_depthffm_fim(args)     
        score_box = [[] for _ in range(len(metric_keys))]
        for index, (key, value) in enumerate(metric_keys.items()):
            if value == 1:
                score_box[index].append(best_result[index])
    
    for index, (key, value) in enumerate(metric_keys.items()):
        if value == 1:
            args.logger.info('repeated '+ key +' scores: ' + str(score_box[index]), main_process_only=True)
            avg_score = np.average(score_box[index])
            args.logger.info('avg of the '+ key +' scores ' + str(avg_score), main_process_only=True)

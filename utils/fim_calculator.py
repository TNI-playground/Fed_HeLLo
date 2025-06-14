import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import re
import sys
import time
from torch import Tensor
from typing import Dict, Optional
from sklearn.cluster import KMeans
import numpy as np

def move_dict_to_device(dict_of_tensors, device):
    """
    Move a dictionary of tensors to the specified device.
    
    Args:
        dict_of_tensors (dict): Dictionary containing tensors.
        device (str or torch.device): Device to move tensors to, e.g., 'cuda:0', 'cpu'.
    
    Returns:
        dict: Dictionary containing tensors moved to the specified device.
    """
    # Iterate over the items in the dictionary
    for key, tensor in dict_of_tensors.items():
        # Move each tensor to the specified device
        dict_of_tensors[key] = tensor.to(device)
    
    return dict_of_tensors

class FIMCalculator:
    def __init__(self, args, model, test_data):
        self.model_name = 'test_model_name'
        self.model = model
        self.test_data = test_data
        self.device = args.accelerator.device
        self.model.to(self.device)
        self.num_samples = len(self.test_data) 

    def compute_fim(self, empirical=True, verbose=True, every_n=None):
        # ipdb.set_trace()
        all_fims = self.fim_diag(self.model, 
                                 self.test_data, 
                                 samples_no=self.num_samples, 
                                 empirical=empirical, 
                                 device=self.device, 
                                 verbose=verbose, 
                                 every_n=every_n)
        
        fim_diag_by_layer = self.aggregate_fisher_information(all_fims, self.model_name)
        return fim_diag_by_layer

    @staticmethod
    def fim_diag(model: Module,
                 data_loader: DataLoader,
                 samples_no: int = None,
                 empirical: bool = False,
                 device: torch.device = None,
                 verbose: bool = False,
                 every_n: int = None) -> Dict[int, Dict[str, Tensor]]:
        model.eval()
        fim = {}
        for name, param in model.named_parameters():
            if param.requires_grad and 'classifier' not in name:
                fim[name] = torch.zeros_like(param)
        
        seen_no = 0
        last = 0
        tic = time.time()

        all_fims = dict({})
        while samples_no is None or seen_no < samples_no:
            data_iterator = iter(data_loader)
            try:
                batch = next(data_iterator)
            except StopIteration:
                if samples_no is None:
                    break
                data_iterator = iter(data_loader)
                batch = next(data_iterator)
            move_dict_to_device(batch, device)
            logits = model(**batch).logits

            if empirical:
                outdx = batch['labels'].unsqueeze(1)
            else:
                outdx = Categorical(logits=logits).sample().unsqueeze(1).detach()
            outdx = outdx.to(torch.int64) # added to fix stsb error
            samples = logits.gather(1, outdx)

            if 'pixel_values' in batch:
                idx, batch_size = 0, batch['pixel_values'].size(0)
            elif 'input_ids' in batch:
                idx, batch_size = 0, batch['input_ids'].size(0)

            while idx < batch_size and (samples_no is None or seen_no < samples_no):
                model.zero_grad()
                torch.autograd.backward(samples[idx], retain_graph=True)
                for name, param in model.named_parameters():
                    if param.requires_grad and 'classifier' not in name:
                        fim[name] += (param.grad * param.grad)
                        fim[name].detach_()
                seen_no += 1
                idx += 1

                if verbose and seen_no % 100 == 0:
                    toc = time.time()
                    fps = float(seen_no - last) / (toc - tic)
                    tic, last = toc, seen_no
                    sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.4f} samples/s.")

                if every_n and seen_no % every_n == 0:
                    all_fims[seen_no] = {n: f.clone().div_(seen_no).detach_()
                                        for (n, f) in fim.items()}

        if verbose:
            if seen_no > last:
                toc = time.time()
                fps = float(seen_no - last) / (toc - tic)
            sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.5f} samples/s.\n")

        for name, grad2 in fim.items():
            grad2 /= float(seen_no)

        all_fims[seen_no] = fim

        return all_fims

    @staticmethod
    def aggregate_fisher_information(all_fims, model_name):
        latest_fim_diag = all_fims[max(all_fims.keys())]
        fim_diag_by_layer = {}
        
        for param_name, param_fim_diag in latest_fim_diag.items():
            layer_name = '-'.join(param_name.split('.')[:6])
            if layer_name not in fim_diag_by_layer:
                fim_diag_by_layer[layer_name] = 0.0

            fim_diag_by_layer[layer_name] += torch.norm(param_fim_diag, p='fro').item()
        return fim_diag_by_layer

    @staticmethod
    def bottom_k_layers(input_dict, k):
        sorted_items = sorted(input_dict.items(), key=lambda x: x[1])
        keys = [int(re.findall(r'\d+', item[0])[0]) for item in sorted_items[:k]]

        kmeans = KMeans(n_clusters=3)
        kmeans.fit(np.array([x[1] for x in input_dict.items()]).reshape(-1,1))
        cluster_labels = kmeans.labels_
        return keys, cluster_labels

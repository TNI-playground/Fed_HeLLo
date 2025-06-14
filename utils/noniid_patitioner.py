# Reference to https://github.com/taokz/FeDepth
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import copy
from torch.utils.data import Dataset, DataLoader
import torch
from collections import Counter

class _Sampler(object):
    def __init__(self, arr):
        self.arr = copy.deepcopy(arr)

    def next(self):
        raise NotImplementedError()

class shuffle_sampler(_Sampler):
    def __init__(self, arr, rng=None):
        super().__init__(arr)
        if rng is None:
            rng = np.random
        rng.shuffle(self.arr)
        self._idx = 0
        self._max_idx = len(self.arr)

    def next(self):
        if self._idx >= self._max_idx:
            np.random.shuffle(self.arr)
            self._idx = 0
        v = self.arr[self._idx]
        self._idx += 1
        return v

class Partitioner(object):
    """Class for partition a sequence into multiple shares (or users).

    Args:
        rng (np.random.RandomState): random state.
        partition_mode (str): 'dir' for Dirichlet distribution or 'uni' for uniform.
        max_n_sample_per_share (int): max number of samples per share.
        min_n_sample_per_share (int): min number of samples per share.
        max_n_sample (int): max number of samples
        verbose (bool): verbosity
    """
    def __init__(self, rng=None, partition_mode="dir",
                 max_n_sample_per_share=-1,
                 min_n_sample_per_share=2,
                 max_n_sample=-1,
                 verbose=True,
                 dir_par_beta=1
                 ):
        assert max_n_sample_per_share < 0 or max_n_sample_per_share > min_n_sample_per_share, \
            f"max ({max_n_sample_per_share}) > min ({min_n_sample_per_share})"
        self.rng = rng if rng else np.random
        self.partition_mode = partition_mode
        self.max_n_sample_per_share = max_n_sample_per_share
        self.min_n_sample_per_share = min_n_sample_per_share
        self.max_n_sample = max_n_sample
        self.verbose = verbose
        self.dir_par_beta = dir_par_beta

    def __call__(self, n_sample, n_share, log=print):
        """Partition a sequence of `n_sample` into `n_share` shares.
        Returns:
            partition: A list of num of samples for each share.
        """
        assert n_share > 0, f"cannot split into {n_share} share"
        if self.verbose:
            log(f"  {n_sample} smp => {n_share} shards by {self.partition_mode} distribution")
        if self.max_n_sample > 0:
            n_sample = min((n_sample, self.max_n_sample))
        if self.max_n_sample_per_share > 0:
            n_sample = min((n_sample, n_share * self.max_n_sample_per_share))

        if n_sample < self.min_n_sample_per_share * n_share:
            raise ValueError(f"Not enough samples. Require {self.min_n_sample_per_share} samples"
                             f" per share at least for {n_share} shares. But only {n_sample} is"
                             f" available totally.")
        n_sample -= self.min_n_sample_per_share * n_share
        if self.partition_mode == "dir":
            partition = (self.rng.dirichlet(n_share * [self.dir_par_beta]) * n_sample).astype(int)
        elif self.partition_mode == "uni":
            partition = int(n_sample // n_share) * np.ones(n_share, dtype='int')
        else:
            raise ValueError(f"Invalid partition_mode: {self.partition_mode}")

        # uniformly add residual to as many users as possible.
        for i in self.rng.choice(n_share, n_sample - np.sum(partition)):
            partition[i] += 1
            # partition[-1] += n_sample - np.sum(partition)  # add residual
        assert sum(partition) == n_sample, f"{sum(partition)} != {n_sample}"
        partition = partition + self.min_n_sample_per_share
        n_sample += self.min_n_sample_per_share * n_share
        # partition = np.minimum(partition, max_n_sample_per_share)
        partition = partition.tolist()

        assert sum(partition) == n_sample, f"{sum(partition)} != {n_sample}"
        assert len(partition) == n_share, f"{len(partition)} != {n_share}"
        return partition

class ClassWisePartitioner(Partitioner):
    """Partition a list of labels by class. Classes will be shuffled and assigned to users
    sequentially.

    Args:
        n_class_per_share (int): number of classes per share (user).
        rng (np.random.RandomState): random state.
        partition_mode (str): 'dir' for Dirichlet distribution or 'uni' for uniform.
        max_n_sample_per_share (int): max number of samples per share.
        min_n_sample_per_share (int): min number of samples per share.
        max_n_sample (int): max number of samples
        verbose (bool): verbosity
    """
    def __init__(self, n_class_per_share=2, dir_par_beta=1, n_domain_per_share=6, **kwargs):
        super(ClassWisePartitioner, self).__init__(**kwargs)
        self.n_class_per_share = n_class_per_share
        self.dir_par_beta = dir_par_beta
        self.n_domain_per_share = n_domain_per_share
        self._aux_partitioner = Partitioner(dir_par_beta=dir_par_beta,**kwargs)

    def __call__(self, labels, n_user, log=print, user_ids_by_class=None,
                 return_user_ids_by_class=False, consistent_class=False, domains=None):
        """Partition a list of labels into `n_user` shares.
        Returns:
            partition: A list of users, where each user include a list of sample indexes.
        """
        if domains:
            # reorganize labels by class
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()
            idx_by_class = defaultdict(list)
            idx_by_domain = defaultdict(list)
            if len(labels) > 1e5:
                labels_iter = tqdm(labels, leave=False, desc='sort labels')
            else:
                labels_iter = labels
                domain_iter = domains
            for i, label in enumerate(labels_iter):
                idx_by_class[label].append(i)
            if domains:
                for i, domain in enumerate(domain_iter):
                    idx_by_domain[domain].append(i)

            n_class = len(idx_by_class)
            if domains:
                n_domain = len(idx_by_domain)
            # assert n_user * self.n_class_per_share > n_class, f"Cannot split {n_class} classes into " \
                                                            #   f"{n_user} users when each user only " \
                                                            #   f"has {self.n_class_per_share} classes."

            # assign classes to each user.
            if user_ids_by_class is None:
                user_ids_by_class = defaultdict(list)
                label_sampler = shuffle_sampler(list(range(n_class)),
                                                self.rng if consistent_class else None)
                for s in range(n_user):
                    s_classes = [label_sampler.next() for _ in range(self.n_class_per_share)]
                    for c in s_classes:
                        user_ids_by_class[c].append(s)

            # assign sample indexes to clients
            idx_by_user = [[] for _ in range(n_user)]
            if n_class > 100 or len(labels) > 1e5:
                idx_by_class_iter = tqdm(idx_by_class, leave=True, desc='split cls')
                log = lambda log_s: idx_by_class_iter.set_postfix_str(log_s[:10])  # tqdm.write
            else:
                idx_by_class_iter = idx_by_class
            for c in idx_by_class_iter:
                l = len(idx_by_class[c])
                log(f" class-{c} => {len(user_ids_by_class[c])} shares")
                initial_domains = np.arange(n_domain)
                np.random.shuffle(initial_domains)
                extra_assignments = np.random.choice(n_domain, len(user_ids_by_class[c]) - n_domain, replace=True)
                domain_assignments = np.concatenate((initial_domains, extra_assignments))
                counter_domain_samples = Counter(domain_assignments)
                l_per_domain = l / n_domain
                in_class_domain_ids = np.array(domain_iter)[idx_by_class[c]]
                in_class_domain_id_dict = {}
                for i in range(n_domain):
                    in_class_domain_id_dict[i] = np.array(idx_by_class[c])[np.where(in_class_domain_ids == i)[0]]
                
                for i_user, i_domain in zip(user_ids_by_class[c], domain_assignments):
                    domain_sample_dividen = int(l_per_domain / counter_domain_samples[i_domain])
                    selected_ids = np.random.choice(in_class_domain_id_dict[i_domain],
                                                    size=domain_sample_dividen if domain_sample_dividen <= len(in_class_domain_id_dict[i_domain]) else len(in_class_domain_id_dict[i_domain]),
                                                    replace=False)
                    idx_by_user[i_user].extend(selected_ids)
                    in_class_domain_id_dict[i_domain] = np.array(list(set(in_class_domain_id_dict[i_domain]) - set(selected_ids)))

            if return_user_ids_by_class:
                return idx_by_user, user_ids_by_class
            else:
                return idx_by_user
        else:
            # reorganize labels by class
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()
            idx_by_class = defaultdict(list)
            if len(labels) > 1e5:
                labels_iter = tqdm(labels, leave=False, desc='sort labels')
            else:
                labels_iter = labels
            for i, label in enumerate(labels_iter):
                idx_by_class[label].append(i)

            n_class = len(idx_by_class)
            assert self.n_domain_per_share > 1, "Only support 1 domain per share"
            # assert n_user * self.n_class_per_share > n_class, f"Cannot split {n_class} classes into " \
                                                            #   f"{n_user} users when each user only " \
                                                            #   f"has {self.n_class_per_share} classes."

            # assign classes to each user.
            if user_ids_by_class is None:
                user_ids_by_class = defaultdict(list)
                label_sampler = shuffle_sampler(list(range(n_class)),
                                                self.rng if consistent_class else None)
                for s in range(n_user):
                    s_classes = [label_sampler.next() for _ in range(self.n_class_per_share)]
                    for c in s_classes:
                        user_ids_by_class[c].append(s)

            # assign sample indexes to clients
            idx_by_user = [[] for _ in range(n_user)]
            if n_class > 100 or len(labels) > 1e5:
                idx_by_class_iter = tqdm(idx_by_class, leave=True, desc='split cls')
                log = lambda log_s: idx_by_class_iter.set_postfix_str(log_s[:10])  # tqdm.write
            else:
                idx_by_class_iter = idx_by_class
            for c in idx_by_class_iter:
                l = len(idx_by_class[c])
                log(f" class-{c} => {len(user_ids_by_class[c])} shares")
                l_by_user = self._aux_partitioner(l, len(user_ids_by_class[c]), log=log) # num of sample for each client id in this categories
                base_idx = 0
                for i_user, tl in zip(user_ids_by_class[c], l_by_user): # i_user: client_id, tl: num of sample
                    idx_by_user[i_user].extend(idx_by_class[c][base_idx:base_idx+tl])
                    base_idx += tl
            if return_user_ids_by_class:
                return idx_by_user, user_ids_by_class
            else:
                return idx_by_user
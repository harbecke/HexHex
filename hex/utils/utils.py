#!/usr/bin/env python3
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _one_pass(iters):
    for it in iters:
        try:
            yield next(it)
        except StopIteration:
            pass

def zip_list_of_lists_first_dim_reversed(*iterables):
    iters = [reversed(it) for it in iterables]
    output_list = []
    while True:
        iter_list = list(_one_pass(iters))
        output_list.extend(list(iter_list))
        if iter_list==[]:
            return output_list

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

def all_unique(x):
    """
    Returns whether all elements in the list are unique,
    i.e. if no element appears twice or more often.
    """
    seen = set()
    return not any(i in seen or seen.add(i) for i in x)

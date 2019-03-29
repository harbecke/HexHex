#!/usr/bin/env python3
import torch
import argparse

from hex.creation.create_model import create_model

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

def load_model(model_file):
    print("=== loading model ===")
    checkpoint = torch.load(model_file, map_location=device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--board_size', type=int, default=checkpoint['board_size'])
    parser.add_argument('--model_type', type=str, default=checkpoint['model_type'])
    parser.add_argument('--layer_type', type=str, default=checkpoint['layer_type'])
    parser.add_argument('--layers', type=int, default=checkpoint['layers'])
    parser.add_argument('--intermediate_channels', type=int, default=checkpoint['intermediate_channels'])

    model = create_model(parser.parse_args())
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

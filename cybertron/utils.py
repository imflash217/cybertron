"""
The utility module
"""

from inspect import isfunction
from functools import partial
import math
import torch
import torch.nn.functional as F


def exists(value) -> bool:
    """
    Checks if the value exists or not
    """
    return value is not None


def default(value, default):
    """
    If the "value" does not exists then the default is returned
    """
    if exists(value):
        return value
    return default() if isfunction(default) else default


def cast_tuple(value, depth) -> tuple:
    """
    Casts a value into tuple into the required depth
    """
    return value if isinstance(value, tuple) else (value,) * depth


def max_neg_value(tensor):
    """Returns the maximum negative value that can be represented by the data type of the tensor"""
    return -torch.finfo(tensor.dtype).max


## helpers => classes


class always:
    """A callable class that always returns the init value when called"""

    def __init__(self, value):
        self.value = value

    def __call__(self, *args, **kwargs):
        return self.value


class equals:
    """A callable class that checks if the input is equal to the init value"""

    def __init__(self, value):
        self.value = value

    def __call__(self, input, *args, **kwargs):
        return input == self.value


class not_equals:
    """A callable class that checks if the input is not-equal to the init value"""

    def __init__(self, value):
        self.value = value

    def __call__(self, input, *args, **kwargs):
        return input != self.value


## helpers => keyword arguments


def pick_and_pop(keys, d: dict) -> dict:
    """
    pick all (key, val) pair from dictionary 'd' for all keys in 'keys'
    Then, return a new dict with those picked (key, val) pair
    """
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))


def string_begins_with(prefix: str, string: str) -> bool:
    """
    Checks whether the "string" begins with the "prefix"
    """
    return string.startswith(prefix)


def group_dict_by_keys(condition, d: dict):
    """
    Groups the given dict "d" into two parts whether the keys satisfy the condition or not
    """
    return_values = [dict(), dict()]
    for key in d.keys():
        match = bool(condition(key))  ## does the keys satisfy condition?
        idx = int(not match)  ## idx=0 if the keys staisfy the condition else 1
        return_values[idx][key] = d[key]
    return (*return_values,)


def group_by_key_prefix(prefix, d: dict):
    return group_dict_by_keys(partial(string_begins_with, prefix), d)


def group_by_prefix_and_trim(prefix, d: dict):
    kwargs_with_prefix, kwargs = group_by_key_prefix(prefix, d)
    trimmer = lambda x: (x[0][len(prefix) :], x[1])
    kwargs_without_prefix = dict(map(trimmer, kwargs_with_prefix.items()))
    return kwargs_without_prefix, kwargs


## nucleus


def top_p(logits, thres=0.9):
    """Top Performing"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.sumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0
    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def top_k(logits, thres=0.9):
    """returns top-k elements which satisfies > 'thres'"""
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, idx = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    return probs.scatter(1, idx, val)


def top_a(logits, min_p_pow=2.0, min_p_ratio=0.02):
    probs = F.softmax(logits, dim=-1)
    limit = torch.pow(torch.max(probs), min_p_pow) * min_p_ratio
    logits[probs < limit] = float("-inf")
    logits[probs >= limit] = 1.0
    return logits

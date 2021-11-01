"""
PyTorch Implementation of Transformers architecture
"""


from inspect import isfunction
import torch
import torch.nn as nn
import torch.nn.functional as F

## constants
DEFAULT_DIM_HEAD = 64
INTERMEDIATES = namedtuple("INTERMEDIATES", ["pre_softmax_attn", "post_softmax_attn"])
LAYER_INTERMEDIATES = namedtuple("INTERMEDIATES", ["hiddens", "attn_intermediates"])

## helpers => functions


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
    return value if isinstance(value, tuple) else (val,) * depth


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


## helpers => initialization


def init_zero_(layer: nn.layer):
    """inplace intializes the weights and biases of the layer to zero"""
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)


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

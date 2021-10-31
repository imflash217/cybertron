"""
PyTorch Implementation of Transformers architecture
"""


from inspect import isfunction
import torch
import torch.nn.functional as F


## constants
DEFAULT_DIM_HEAD = 64
INTERMEDIATES = namedtuple("INTERMEDIATES", ["pre_softmax_attn", "post_softmax_attn"])
LAYER_INTERMEDIATES = namedtuple("INTERMEDIATES", ["hiddens", "attn_intermediates"])

## helpers => functions


def exists(value):
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


def cast_tuple(value, depth):
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


"""
PyTorch Implementation of Transformers architecture
"""


from inspect import isfunction
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15
from einops import rearrange, reduce, repeat

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


def init_zero_(layer):
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


## activation functions


class ReluSquared(nn.Module):
    def forward(self, input):
        return F.relu(input) ** 2


## NORMS


class Scale(nn.Module):
    def __init__(self, value, func):
        super().__init__()
        self.value = value
        self.func = func

    def forward(self, x, **kwargs):
        x, *rest = self.func(x, **kwargs)
        return (x * self.value, *rest)


class ReZero(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x, **kwargs):
        x, *rest = self.func(x, **kwargs)
        return (x * self.g, *rest)


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        norm *= self.scale
        return x / norm.clamp(min=self.eps) * self.g


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        norm *= self.scale
        return x / norm.clamp(min=self.eps) * self.g


## token shifting


def shift(t, amount, mask=None):
    if amount == 0:
        return t
    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.0)
    ## here, we are shifting in -2 axis, by padding by -ve amount and +ve amount
    ## -ve amount of adding essentially cuts down into the axis
    ## while +ve padding amount inflates that axis
    ## hence, keeping the size same but effectively shifting
    pad = (0, 0, -amount, amount)
    return F.pad(t, pad, value=0.0)


## feedforward


class GLU(nn.Module):
    """Applies Gated Linear Unit with custom activation function"""

    def __init__(self, in_dim, out_dim, act):
        super().__init__()
        self.act = act
        self.proj = nn.Linear(in_dim, out_dim ** 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(chunks=2, dim=-1)
        return x * self.act(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        glu=False,
        relu_squared=False,
        post_act_ln=False,
        dropout=0.0,
        zero_init_output=False,
    ):
        super().__init__()
        dim_hidden = int(dim * mult)
        dim_out = default(dim_out, dim)
        act = ReluSquared() if relu_squared else nn.GELU()

        if glu:
            project_in = GLU(dim, dim_hidden, act)
        else:
            project_in = nn.Sequential(nn.Linear(dim, dim_hidden), act)

        self.net = nn.Sequential(
            project_in,
            nn.LayerNorm(dim_hidden) if post_act_ln else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_out),
        )

        ## init last linear layer to 0
        if zero_init_output:
            init_zero_(self.net[-1])

    def forward(self, x):
        return self.net(x)


## ATTENTION


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=DEFAULT_DIM_HEAD,
        heads=8,
        causal=False,
        mask=None,
        talking_heads=False,
        head_scale=False,
        collab_heads=False,
        collab_compression=0.3,
        sparse_topk=None,
        use_entmax15=False,
        num_mem_kv=0,
        dropout=0.0,
        attn_on_attn=False,
        gate_values=False,
        zero_init_output=False,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.head = heads
        self.causal = causal
        self.mask = mask

        dim_qk = dim_v = dim_head * heads

        ## collaborative heads
        self.collab_heads = collab_heads
        if self.collab_heads:
            dim_qk = int(collab_compression * dim_qk)
            self.collab_mixing = nn.Parameter(torch.randn(heads, dim_qk))

        self.to_q = nn.Linear(dim, dim_qk, bias=False)
        self.to_k = nn.Linear(dim, dim_qk, bias=False)
        self.to_v = nn.Linear(dim, dim_v, bias=False)

        self.dropout = nn.Dropout(dropout)

        ## add GLU gating for aggregated values (from "alphafold2" paper)
        self.to_v_gate = None
        if self.gate_values:
            self.to_v_gate = nn.Linear(dim, dim_v)
            nn.init.constant_(self.to_v_gate.weight, 0.0)
            nn.init.constant_(self.to_v_gate.bias, 1.0)

        ## talking heads
        self.talking_heads = talking_heads
        if self.talking_heads:
            self.pre_softmax_proj = nn.Parameter(torch.randn(heads, heads))
            self.post_softmax_proj = nn.Parameter(torch.randn(heads, heads))

        ## head scaling
        self.head_scale = head_scale
        if self.head_scale:
            self.head_scale_params = nn.Parameter(torch.ones(1, heads, 1, 1))

        ## explicit "topk" sparse attention
        self.sparse_topk = sparse_topk

        ## entmax
        self.attn_fn = entmax15 if use_entmax15 else F.softmax

        ## add memory key/values
        self.num_mem_kv = num_mem_kv
        if self.num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))

        ## attention on attention
        self.attn_on_attn = attn_on_attn
        if self.attn_on_attn:
            self.to_out = nn.Sequential(nn.Linear(dim_v, dim * 2), nn.GLU())
        else:
            self.to_out = nn.Linear(dim_v, dim)

        ## init output projection to 0
        if zero_init_output:
            init_zero_(self.to_out)

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        attn_mask=None,
        rel_pos=None,
        sinusoidal_emb=None,
        rotary_pos_emb=None,
        prev_attn=None,
        mem=None,
    ):
        b, n, _ = *x.shape
        h = self.heads
        talking_heads = self.talking_heads
        collab_heads = self.collab_heads
        head_scale = self.head_scale
        device = x.device
        has_context = exists(context)

        kv_input = default(context, x)
        q_input = x
        k_input = kv_input
        v_input = kv_input

        if exists(mem):
            k_input = torch.cat((mem, k_input), dim=-2)
            v_input = torch.cat((mem, v_input), dim=-2)

        if exists(sinusoidal_emb):
            ## in SHORTFORMER, the query would start at a position offset
            ## depending on the past cached memory
            offset = k_input.shape[-2] - q_input.shape[-2]
            q_input = q_input + sinusoidal_emb(q_input, offset=offset)
            k_input = k_input + sinusoidal_emb(k_input)

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        if not collab_heads:
            q, k, v = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v)
            )
        else:
            q = torch.einsum("b i d, h d -> b h i d", q, self.collab_mixing)
            k = rearrange(k, "b n d -> b () n d")
            v = rearrange(v, "b n (h d) -> b h n d", h=h)

        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: torch.ones((b, n), device=device).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(
                k_mask, lambda: torch.ones((b, k.shape[-2]), device=device).bool()
            )
            q_mask = rearrange(q_mask, "b i -> b () i ()")
            k_mask = rearrange(k_mask, "b j -> b () () j")
            input_mask = q_mask * k_mask

        if self.num_mem_kv > 0:
            mem_k, mem_v = map(
                lambda t: repeat(t, "h n d -> b h n d", b=b), (self.mem_k, self.mem_v)
            )
            k = torch.cat((mem_k, k), dim=-2)
            v = torch.cat((mem_v, v), dim=-2)
            if exists(input_mask):
                input_mask = F.pad(input_mask, (self.num_mem_kv, 0), value=True)

        if collab_heads:
            k = k.expand(-1, h, -1, -1)

        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        mask_value = max_neg_value(dots)

        if exists(prev_attn):
            dots = dots + prev_attn

        pre_softmax_attn = dots.clone()

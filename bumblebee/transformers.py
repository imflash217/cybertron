"""
PyTorch Implementation of Transformers architecture
"""

import math
from functools import partial
from inspect import isfunction
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15
from einops import rearrange, reduce, repeat

## constants
DEFAULT_DIM_HEAD = 64
Intermediates = namedtuple("Intermediates", ["pre_softmax_attn", "post_softmax_attn"])
LayerIntermediates = namedtuple("Intermediates", ["hiddens", "attn_intermediates"])

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


## positional embeddings


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        n = torch.arange(x.shape[1], device=x.device)
        pos_emb = self.emb(n)
        pos_emb = rearrange(pos_emb, "n d -> () n d")
        return pos_emb * self.scale


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (1e4 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_dim=1, offset=0):
        t = (
            torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            + offset
        )
        sinusoid_input = torch.einsum("i, j -> i j", t, self.inv_freq)
        emb = torch.cat((sinusoid_input.sin(), sinusoid_input.cos()), dim=-1)
        return rearrange(emb, "n d -> () n d")


class RelativePositionBias(nn.Module):
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.heads = heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, causal=True, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        i, j = *qk_dots.shape[-2:]
        device = qk_dots.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arnage(j, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, "i j h -> () h i j")
        return qk_dots + (bias * self.scale)


class AlibiPositionalBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, "h -> () h () ()")
        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                0 : heads - closest_power_of_2
            ]
        )

    def forward(self, qk_dots):
        h, i, j = *qk_dots.shape[-3:]
        device = qk_dots.device
        if exists(self.bias) and self.bias.shape[-1] >= j:
            return qk_dots + self.bias[..., :j]
        bias = torch.arange(j, device=device)
        bias = rearrange(bias, "j -> () () () j")
        bias *= self.slopes
        bias = F.pad(bias, (0, 0, 0, 0, 0, h - bias.shape[1]))
        self.register_buffer("bias", bias, persistent=False)
        return qk_dots + self.bias


class LearnedAlibiPositionalBias(AlibiPositionalBias):
    def __init__(self, heads):
        super().__init__(heads)
        self.learned_logslopes = nn.Parameter(torch.log(self.slopes))

    def forward(self, qk_dots):
        h, i, j = *qk_dots.shape[-3:]
        device = qk_dots.device
        slopes = self.learned_logslopes.exp()
        slopes = F.pad(slopes, (0, 0, 0, 0, 0, h - slopes.shape[1]))
        if exists(self.bias) and self.bias.shape[-1] >= j:
            bias = self.bias[..., :j]
        else:
            bias = torch.arange(j, device=device)
            self.register_buffer("bias", bias, persistent=False)
        return qk_dots + (bias * slopes)


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

        if talking_heads:
            dots = torch.einsum(
                "b h i j, h k -> b k i j", dots, self.pre_softmax_proj
            ).contiguous()

        if exists(rel_pos):
            dots = rel_pos(dots)

        if exists(input_mask):
            dots.masked_fill_(~input_mask, mask_value)
            del input_mask

        if exists(attn_mask):
            assert (
                2 <= attn_mask.ndim <= 4
            ), "attn mask must have atleast 2 and atmost 4 dimensions"
            if attn_mask.ndim == 2:
                attn_mask = rearrange(attn_mask, "i j -> () () i j")
            elif attn_mask.ndim == 3:
                attn_mask = rearrange(attn_mask, "h i j -> () h i j")
            dots.masked_fill_(~attn_mask, mask_value)

        if self.causal:
            i, j = dots.shape[-2:]
            r = torch.arange(i, device=device)
            mask = rearrange(r, "i -> () () i ()") < rearrange(r, "j -> () () () j")
            mask = F.pad(mask, (j - i, 0), value=False)
            dots.masked_fill_(mask, mask_value)
            del mask

        if exists(self.sparse_topk) and self.sparse_topk < dots.shape[-1]:
            top, _ = dots.topk(self.sparse_topk, dim=-1)
            vk = top[..., -1].unsqueeze(-1).expand_as(dots)
            mask = dots < vk
            dots.masked_fill_(mask, mask_value)
            del mask

        attn = self.attn_fn(dots, dim=-1)
        post_softmax_attn = attn.clone()

        attn = self.dropout(attn)

        if talking_heads:
            attn = torch.einsum(
                "b h i j, h k -> b k i j", attn, self.post_softmax_proj
            ).contiguous()

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)

        if self.head_scale:
            out *= self.head_scale_params

        out = rearrange(out, "b h n d -> b n (h d)")

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            out *= gates.sigmoid()

        intermediates = Intermediates(
            pre_softmax_attn=pre_softmax_attn, post_softmax_attn=post_softmax_attn
        )
        return self.to_out(out), intermediates


class AttentionLayers(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads=8,
        causal=False,
        cross_attend=False,
        only_cross=False,
        use_scalenorm=False,
        use_rmsnorm=False,
        use_rezero=False,
        alibi_pos_bias=False,
        alibi_num_heads=None,
        alibi_learned=False,
        rel_pos_bias=False,
        rel_pos_num_buckets=32,
        rel_pos_max_distance=128,
        position_infused_attn=False,
        rotary_pos_emb=False,
        rotary_emb_dim=None,
        custom_layers=None,
        sandwich_coeff=None,
        par_ratio=None,
        residual_attn=False,
        cross_residual_attn=False,
        macaron=False,
        pre_norm=True,
        gate_residual=False,
        scale_residual=False,
        shift_tokens=0,
        sandwich_norm=False,
        zero_init_branch_output=False,
        **kwargs,
    ):
        super().__init__()
        ff_kwargs, kwargs = group_by_prefix_and_trim("ff_", kwargs)
        attn_kwargs, _ = group_by_prefix_and_trim("attn_", kwargs)

        dim_head = attn_kwargs.get("dim_head", DEFAULT_DIM_HEAD)

        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        self.has_pos_emb = position_infused_attn or rel_pos_bias or rotary_pos_emb
        self.pia_pos_emb = (
            FixedPositionalEmbedding(dim) if position_infused_attn else None
        )

        rotary_emb_dim = max(32, default(rotary_emb_dim, dim_head // 2))
        self.rotary_pos_emb = (
            RotaryEmbedding(rotary_emb_dim) if rotary_pos_emb else None
        )

        assert not (
            alibi_pos_bias and rel_pos_bias
        ), "you can only choose ALiBi positional bias or T5 relative positional bias; not BOTH"

        assert (
            rel_pos_num_buckets <= rel_pos_max_distance
        ), "number of relative position buckets must be less than the relative position max distance"

        if rel_pos_bias:
            self.rel_pos = RelativePositionBias(
                scale=dim_head ** 0.5,
                causal=causal,
                heads=heads,
                num_buckets=rel_pos_num_buckets,
                max_distance=rel_pos_max_distance,
            )
        elif alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert (
                alibi_num_heads <= heads
            ), "number of ALiBi heads must be <= total numbe rof heads"
            assert causal, "ALiBi does not work with non-autoregressive mode just yet"
            alibi_pos_klass = (
                LearnedAlibiPositionalBias if alibi_learned else AlibiPositionalBias
            )
            self.rel_pos = alibi_pos_klass(heads=alibi_num_heads)
        else:
            self.rel_pos = None

        assert not (
            not pre_norm and sandwich_norm
        ), "sandwichNorm cannot be used when preNorm is not used"
        self.pre_norm = pre_norm
        self.sandwich_norm = sandwich_norm

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        self.cross_attend = cross_attend

        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        norm_class = RMSNorm if use_rmsnorm else norm_class
        norm_fn = partial(norm_class, dim)
        norm_fn = nn.Identity if use_rezero else norm_fn

        branch_fn = ReZero if use_rezero else None

        if cross_attend and not only_cross:
            ## "attention", "cross-attention", "feedforward"
            default_block = ("a", "c", "f")
        elif cross_attend and only_cross:
            ## "cross-attention" and "feedforward"
            default_block = ("c", "f")
        else:
            ## vanilla setting: "attention" and "feedforward"
            default_block = ("a", "f")

        if macaron:
            ## adds a "feedforward" block before as per the paper.
            default_block = "f" + default_block

        ## zero init
        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, "zero_init_output": True}
            ff_kwargs = {**ff_kwargs, "zero_init_output": True}

        ## calculate layer block order
        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 <= par_ratio <= par_depth, "par_ratio out of range"
            default_block = tuple(filter(not_equals("f"), default_block))
            par_attn = par_depth // par_ratio

            ## 2/3 attention layer cutoff as suggested by PAR paper
            depth_cut = par_depth * (2 // 3)
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert (
                len(default_block) <= par_width
            ), "default block is too large for par_ratio"
            par_block = default_block + ("f",) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ("f",) * (par_depth - len(par_head))
        elif exists(sandwich_coeff):
            assert (
                0 < sandwich_coeff <= depth
            ), "sandwich coefficient should be less than depth"
            layer_types = (
                ("a",) * sandwich_coeff
                + default_block * (depth - sandwich_coeff)
                + ("f",) * sandwich_coeff
            )
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals("a"), layer_types)))

        ## calculate token shifting
        shift_tokens = cast_tuple(shift_tokens, len(layer_types))

        ## iterate and construct layers
        for layer_type, layer_shift_tokens in zip(self.layer_types, shift_tokens):
            if layer_type == "a":
                layer = Attention(dim, heads=heads, causal=causal, **attn_kwargs)
            elif layer_type == "c":
                layer = Attention(dim, heads=heads, **attn_kwargs)
            elif layer_type == "f":
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f"invalid layer type: {layer_type}")

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens if not causal else 0
                layer = ShiftTokens(range(shift_range_lower, shift_range_upper), layer)

            if isinstance(layer, Attention) and exists(branch_fn):
                layer = branch_fn(layer)

            residual_fn = GRUGating if gate_residual else Residual
            residual = residual_fn(dim, scale_residual=scale_residual)

            if sandwich_norm:
                norm = nn.ModuleList([norm_fn(), norm_fn()])
            else:
                norm = norm_fn()

            self.layers.append(nn.ModuleList([norm, layer, residual]))

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        attn_mask=None,
        mems=None,
        return_hiddens=False,
    ):
        assert not (
            self.cross_attend ^ exists(context)
        ), "context must be passed in if cross_attend is set to True"

        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            max_rotary_emb_length = max(
                list(map(lambda m: (m.shape[1] if exists(m) else 0) + x.shape[1], mems))
            )
            rotary_pos_emb = self.rotary_pos_emb(max_rotary_emb_length, x.device)

        for idx, (layer_type, (norm, block, residual_fn)) in enumerate(
            zip(self.layer_types, self.layers)
        ):
            is_last = idx == (len(self.layers) - 1)

            if layer_type == "a":
                hiddens.append(x)
                layer_mem = mems.pop(0) if mems else None

            residual = x

            if self.sandwich_norm:
                norm, postnorm = norm
            if self.pre_norm:
                x = norm(x)

            if layer_type == "a":
                out, inter = block(
                    x,
                    mask=mask,
                    attn_mask=attn_mask,
                    sinusoidal_emb=self.pia_pos_emb,
                    rel_pos=self.rel_pos,
                    rotary_pos_emb=rotary_pos_emb,
                    prev_attn=prev_attn,
                    mem=layer_mem,
                )
            elif layer_type == "c":
                out, inter = block(
                    x,
                    context=context,
                    mask=mask,
                    context_mask=context_mask,
                    prev_attn=prev_cross_attn,
                )
            elif layer_type == "f":
                out = block(x)

            if self.sandwich_norm:
                out = postnorm(out)

            x = residual_fn(out, residual)

            if layer_type in ("a", "c"):
                intermediates.append(inter)

            if layer_type == "a" and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == "c" and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if not self.pre_norm and not is_last:
                x = norm(x)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hideens=hiddens, attn_intermediates=intermediates
            )
            return x, intermediates
        return x

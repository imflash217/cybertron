"""
Autoregressive Wrapper
"""


import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from entmax import entmax_bisect

import utils

## entmax

ENTMAX_ALPHA = 1.3
entmax = entmax_bisect


class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, ignore_index=-100, pad_value=0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.net = net
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def generate(
        self,
        start_tokens,
        seq_len,
        eos_token=None,
        temperature=1.0,
        filter_logits_fn=utils.top_k,
        filter_thres=0.9,
        min_p_pow=2.0,
        min_p_ratio=0.02,
        **kwargs
    ):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)  ## why not use `start_tokens.ndim` ?

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwargs.pop("mask", None)

        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len]
            mask = mask[:, -self.max_seq_len :]
            logits = self.net(x, mask=mask, **kwargs)[:, -1, :]

            if filter_logits_fn in {utils.top_k, utils.top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
            elif filter_logits_fn is utils.top_a:
                filtered_logits = filter_logits_fn(
                    logits, min_p_pow=min_p_pow, min_p_ratio=min_p_ratio
                )
                probs = F.softmax(filtered_logits / temperature, dim=-1)
            elif filter_logits_fn is entmax:
                probs = filter_logits_fn(
                    logits / temperature, alpha=ENTMAX_ALPHA, dim=-1
                )
            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if utils.exists(eos_token):
                is_eos_tokens = out == eos_token
                if is_eos_tokens.any(dim=-1).all():
                    ## mask out everything after EOS tokens
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                    out = out.masked_fill(mask, self.pad_value)
                    break

        out = out[:, t:]
        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def forward(self, x, **kwargs):
        xi = x[:, :-1]
        x0 = x[:, 1:]

        ## if the use supply a "mask" that is off only by one from the source sequence
        ## resolve it for them
        mask = kwargs.get("mask", None)
        if mask is not None and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs["mask"] = mask

        out = self.net(xi, **kwargs)
        loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index=self.ignore_index)
        return loss

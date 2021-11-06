"""Full Encoder Decoder"""

import torch
from transformers import Cybertron

## Step-1: Creating a model instance

model = Cybertron(
    dim=512,
    enc_num_tokens=256,
    enc_depth=6,
    enc_heads=8,
    enc_max_seq_len=1024,
    dec_num_tokens=256,
    dec_depth=6,
    dec_heads=8,
    dec_max_seq_len=1024,
    tie_token_emb=True,  ## tie embeddings of encoder & decoder
)

## Step-2: Generate the data and labels
src = torch.randint(0, 256, (1, 1024))
src_mask = torch.ones_like(src).bool()

tgt = torch.randint(0, 256, (1, 1024))
tgt_mask = torch.ones_like(tgt).bool()

## Step-3: Forward Pass (here we are showing a single forward/backward pass)
## loss.shape=(1,1024, 512)
loss = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
loss.backward()

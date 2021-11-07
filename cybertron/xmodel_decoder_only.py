"""Decoder Only (GPT-like)"""

import tqdm
import wandb
import torch

from transformers import TransformerWrapper, Decoder
from autoregressive_wrapper import AutoregressiveWrapper

wandb.init(project="Cybertron_GPT", entity="imflash217")

## Hyperparameters
NUM_BATCHES = int(1e5)
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
GENERATE_EVERY = 100
NUM_TOKENS = 20000
SEQ_LEN = 1024

device = "cuda" if torch.cuda.is_available() else "cpu"

## helpers


def cycle():
    while True:
        x = torch.randint(2, 256, (1, SEQ_LEN)).long().to(device)
        yield x


## Creating a model instance

model = TransformerWrapper(
    num_tokens=NUM_TOKENS,
    max_seq_len=SEQ_LEN,
    attn_layers=Decoder(dim=512, depth=12, heads=8),
)

model = AutoregressiveWrapper(model).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"model.parameters() = {num_params}")
## optimizer
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

## training
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10, desc="training"):
    model.train()
    x = next(cycle())

    ## loss.shape = (1, 1024, 20000)
    loss = model(x)
    wandb.log({"loss": loss.item()})

    loss.backward()

    optim.step()
    optim.zero_grad()

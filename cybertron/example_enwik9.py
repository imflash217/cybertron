"""
enwik9 dataset analysis
"""

import random
import tqdm
import gzip
import numpy as np
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import TransformerWrapper, Decoder
from autoregressive_wrapper import AutoregressiveWrapper

## wandb setup

wandb.init(project="Cybertron_ENWIK9", entity="imflash217")

## hyperparameters

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_TOKENS = 256
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
SEQ_LEN = 1024
GENERATE_LENGTH = 1024
GRADIENT_ACCUMULATE_EVERY = 4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500

## helpers


def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, device):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.device = device

    def __getitem__(self, index):
        rand_start_idx = torch.randint(0, self.data.shape[0] - self.seq_len - 1, (1,))
        start = rand_start_idx
        end = rand_start_idx + self.seq_len + 1
        full_seq = self.data[start:end].long()
        return full_seq.to(self.device)

    def __len__(self):
        return self.data.shape[0] // self.seq_len


## instantiate GPT-like "decoder-only" model
model = TransformerWrapper(
    num_tokens=NUM_TOKENS,
    max_seq_len=SEQ_LEN,
    attn_layers=Decoder(dim=512, depth=6, heads=8),
)

model = AutoregressiveWrapper(model).to(device)

## prepare enwik9 data

with gzip.open("../data/enwik8.gz") as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    train_X, val_X = np.split(X, [int(90e6)])
    data_train = torch.from_numpy(train_X)
    data_val = torch.from_numpy(val_X)

## creating dataloaders for train & validation sets
train_ds = TextSamplerDataset(data_train, SEQ_LEN, device)
valid_ds = TextSamplerDataset(data_val, SEQ_LEN, device)
train_dl = cycle(DataLoader(train_ds, batch_size=BATCH_SIZE))
valid_dl = cycle(DataLoader(valid_ds, batch_size=BATCH_SIZE))

## optimizer
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

## training

for i in tqdm.tqdm(range(NUM_BATCHES), desc="training", mininterval=10.0):
    model.train()

    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_dl))
        loss.backward()

    wandb.log({"training_loss": loss.item()})

    ## gradient clipping (inplace)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

    optim.step()
    optim.zero_grad()

    ## validation
    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(valid_dl))
            wandb.log({"validation_loss": loss.item()})

    ## text generation (for sample testing)
    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(valid_ds)[:-1]
        prime = decode_tokens(inp)

        print("***" * 20)
        print(f"prime = {prime}")
        print("---" * 20)

        sample = model.generate(inp, GENERATE_LENGTH)
        output_str = decode_tokens(sample)

        print("---" * 20)
        print(f"output_str = {output_str}")
        print("***" * 20)

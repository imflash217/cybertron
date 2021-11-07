"""Full Encoder Decoder"""

import tqdm
import wandb
import torch

from transformers import Cybertron

wandb.init(project="Cybertron", entity="imflash217")
wandb_table = wandb.Table(columns=["batch", "loss", "incorrects", "text"])

## Hyperparameters
NUM_BATCHES = int(1e5)
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
GENERATE_EVERY = 100
NUM_TOKENS = 16 + 2
ENC_SEQ_LEN = 32
DEC_SEQ_LEN = 64 + 1

device = "cuda" if torch.cuda.is_available() else "cpu"

## helpers


def cycle():
    while True:
        prefix = torch.ones((BATCH_SIZE, 1)).long().to(device)
        src = torch.randint(2, NUM_TOKENS, (BATCH_SIZE, ENC_SEQ_LEN)).long().to(device)
        tgt = torch.cat((prefix, src, src), dim=1)
        src_mask = torch.ones(BATCH_SIZE, src.shape[1]).bool().to(device)
        tgt_mask = torch.ones(BATCH_SIZE, tgt.shape[1]).bool().to(device)
        yield (src, tgt, src_mask, tgt_mask)


## Creating a model instance

model = Cybertron(
    dim=512,
    tie_token_emb=True,  ## tie embeddingsof encoder & decoeer
    return_tgt_loss=True,
    enc_num_tokens=NUM_TOKENS,
    enc_depth=3,
    enc_heads=8,
    enc_max_seq_len=ENC_SEQ_LEN,
    dec_num_tokens=NUM_TOKENS,
    dec_depth=3,
    dec_heads=8,
    dec_max_seq_len=DEC_SEQ_LEN,
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"model.parameters() = {num_params}")
## optimizer
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

## training
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10, desc="training"):
    model.train()
    src, tgt, src_mask, tgt_mask = next(cycle())

    ## loss.shape = (10,2024,512)
    loss = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
    wandb.log({"loss": loss.item()})

    loss.backward()

    ## print("loss = ", loss.item())

    optim.step()
    optim.zero_grad()

    if i != 0 and i % GENERATE_EVERY == 0:
        model.eval()
        src, _, src_mask, _ = next(cycle())
        src = src[:1]
        src_mask = src_mask[:1]
        start_tokens = (torch.ones((1, 1)) * 1).long().to(device)

        sample = model.generate(src, start_tokens, ENC_SEQ_LEN, src_mask=src_mask)
        incorrects = (src != sample).abs().sum()

        wandb.log({"incorrects": incorrects})
        wandb_table.add_data(i, loss.item(), incorrects, sample)

        # print(f"inputs: {src}\npredicted: {sample}\nincorrects # {incorrects}")

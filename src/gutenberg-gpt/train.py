import torch
from torch import nn
from torch.nn import functional as F
import os
import numpy as np
from tqdm import tqdm
from gutenberg_tokenizer import GutenbergTokenizer
from paths import TOKENIZER_PATH, BIN_DATA_DIRECTORY, DATA_DIRECTORY, MODEL_PATH
from config import MODEL_CONFIG, TRAINING_CONFIG, GENERATION_CONFIG
from model import LanguageModel

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GutenbergTokenizer().load(str(TOKENIZER_PATH))
vocab_size = tokenizer.VOCABULARY_SIZE


train_bin = f"{BIN_DATA_DIRECTORY}/train.bin"
val_bin = f"{BIN_DATA_DIRECTORY}/val.bin"


max_iters = TRAINING_CONFIG["max_iters"]
eval_interval = TRAINING_CONFIG["eval_interval"]
eval_iters = TRAINING_CONFIG["eval_iters"]

batch_size = TRAINING_CONFIG["batch_size"]
learning_rate = TRAINING_CONFIG["learning_rate"]
train_split = TRAINING_CONFIG["train_split"]

context_length = MODEL_CONFIG["context_length"]
n_embed = MODEL_CONFIG["n_embed"]
n_layer = MODEL_CONFIG["n_layer"]
n_head = MODEL_CONFIG["n_head"]
dropout_prob = MODEL_CONFIG["dropout_prob"]

prompt = GENERATION_CONFIG["prompt"]
max_new_tokens = GENERATION_CONFIG["max_new_tokens"]

data_files = [
    f"{DATA_DIRECTORY}/{file_name}" for file_name in os.listdir(DATA_DIRECTORY)
]

n_train = int(len(data_files) * train_split)
train_files = data_files[:n_train]
val_files = data_files[n_train:]


def build_bin(files, out_path):
    with open(out_path, "wb") as f_out:
        for path in tqdm(files):
            with open(path, encoding="utf-8") as f:
                ids = tokenizer.encode(f.read())
            np.array(ids, dtype=np.uint16).tofile(f_out)


@torch.no_grad()
def estimate_loss(train_files, val_files, eval_iters=200):
    model.eval()
    loss_mean = {}
    for split in ("train", "val"):
        if split == "train":
            files = train_files
        else:
            files = val_files
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batches(
                files, batch_size=batch_size, context_length=context_length
            )
            _, loss = model(X, Y)
            losses[k] = loss
        loss_mean[split] = losses.mean()
    model.train()
    return loss_mean


def get_batches(bin_path, batch_size=4, context_length=256):
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - context_length - 1, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + context_length].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + context_length].astype(np.int64))
            for i in ix
        ]
    )
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


if __name__ == "__main__":
    if not os.path.exists(train_bin):
        build_bin(train_files, train_bin)
    if not os.path.exists(val_bin):
        build_bin(val_files, val_bin)

    model = LanguageModel(
        vocab_size,
        n_head=n_head,
        n_layer=n_layer,
        context_length=context_length,
        n_embed=n_embed,
        dropout_prob=dropout_prob,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    print(tokenizer.decode(model.generate(idx, max_new_tokens)[0].tolist()))

    model.train()
    min_loss = float("+inf")
    for steps in range(max_iters):
        xb, yb = xb, yb = get_batches(
            train_bin, batch_size=batch_size, context_length=context_length
        )
        if steps % eval_interval == 0:
            losses = estimate_loss(
                train_files=train_bin, val_files=val_bin, eval_iters=eval_iters
            )
            print(
                f"Batch : {steps} | Train loss : {losses['train']} | Val loss : {losses['val']}"
            )
            if losses["val"] < min_loss:
                min_loss = losses["val"]
                print("Saving the current model...")
                model.eval()
                print(tokenizer.decode(model.generate(idx, max_new_tokens)[0].tolist()))
                model.train()
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "context_length": context_length,
                        "n_embed": n_embed,
                        "n_layer": n_layer,
                        "n_head": n_head,
                        "dropout_prob": dropout_prob,
                    },
                    MODEL_PATH,
                )

        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

<p align="center">
  <img src="assets/banner.png" alt="gutenberg-gpt" width="100%">
</p>

# gutenberg-gpt

A decoder-only transformer, written from scratch in PyTorch, trained on French public-domain books from Project Gutenberg.

I built this to stop treating attention as a black box. Every piece of the model is hand-rolled here: the attention head, the multi-head wrapper, the feed-forward block, the residual stream, the training loop. The one thing I did not write myself is the BPE trainer, which comes from Hugging Face `tokenizers`. Most walkthroughs of this kind stop at a single Shakespeare file, so I pointed mine at a few thousand French novels instead and trained the tokenizer on the same texts.

## What's in here

```
.
├── config.json                    model and training hyperparameters
├── .env.example                   optional path overrides
├── requirements.txt
├── assets/                        the banner at the top of this page
├── artifacts/
│   ├── gutenberg-fr-tokenizer.json   the trained tokenizer
│   └── gutenberg-gpt.pt           the checkpoint, written by train.py
├── data/
│   ├── books_clean/               the corpus, one plain-text book per file
│   ├── books/                     empty, where the raw books sat before cleaning
│   └── bin/                       tokenized train.bin / val.bin
└── src/gutenberg-gpt/
    ├── model.py                   the transformer
    ├── gutenberg_tokenizer.py     byte-level BPE tokenizer
    ├── train.py                   data prep and training loop
    ├── inference.py               interactive sampling from a checkpoint
    ├── config.py                  reads config.json
    └── paths.py                   resolves every path, honours .env
```

The corpus and the trained tokenizer are committed, so a clone gives you everything you need to start training. The two things you build yourself are the tokenized `.bin` files and the checkpoint, both gitignored. Fair warning: the books alone are around 750 MB, so the clone is not a quick one.

## How it works

**Tokenizer.** A byte-level BPE with a vocabulary of 16384 and a single special token, `<|endoftext|>`. `encode` returns plain token ids, and `train.py` is what appends the special token, once per book, while it builds the `.bin` files. Byte-level matters more than it sounds for French: accented characters, ligatures and the old typography scattered through Gutenberg texts never produce an unknown token, they just cost a few extra bytes. Training runs on a random 5% sample of the training files, which is plenty for BPE and a lot faster than reading the whole 750 MB.

**Data.** Books get tokenized once into two flat `uint16` files, `train.bin` and `val.bin`. Training reads them back with `np.memmap` and slices random windows out of them, so nothing large ever sits in memory and the split stays at the book level rather than the sentence level.

**Model.** Standard GPT-style stack: token embeddings plus learned position embeddings, then N pre-norm blocks, each one causal self-attention followed by a feed-forward layer, both wrapped in residual connections. The causal mask is a lower-triangular buffer registered on each head, so a token can only ever attend to what came before it. Sampling uses temperature and top-k filtering, crops the context window to the last `context_length` tokens on every step, and can stop early when it draws the end-of-text token.

## Setup

```bash
git clone <this repo>
cd gutenberg-gpt

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Built and tested on Python 3.14. Training picks up CUDA automatically when it's available and falls back to CPU otherwise, which works but is slow enough that you'll want a GPU for anything past a smoke test.

## The corpus

`data/books_clean/` holds 1880 French novels, one plain `.txt` file per book, roughly 750 MB of text with the Project Gutenberg header and footer already stripped out. I downloaded and cleaned them with my [Project Gutenberg Reader](https://github.com/quentin-lauret/project-gutenberg-reader).

If you'd rather train on your own texts, leave the folder alone and point `DATA_DIRECTORY` at yours in a `.env` file. Same goes for every other path in the project, see `.env.example`.

## Train the tokenizer

`artifacts/gutenberg-fr-tokenizer.json` ships with the repo, so you only need this step if you change the vocabulary or switch to a different corpus.

```bash
python src/gutenberg-gpt/gutenberg_tokenizer.py
```

Both the training script and the inference script load the tokenizer on import, so the file has to be there before you run either one. If you do retrain it, delete `data/bin/train.bin` and `data/bin/val.bin` by hand, otherwise training keeps reading token ids that no longer mean anything.

## Train the model

```bash
python src/gutenberg-gpt/train.py
```

The first run tokenizes every book into `data/bin/train.bin` and `data/bin/val.bin`. That happens once, and the script skips it if the files already exist. Then it trains, printing train and validation loss every `eval_interval` steps and saving a checkpoint whenever validation loss improves. It also prints a short sample at each save, and one before the very first step, straight from the untrained model, which is the fastest way to see whether the thing is actually learning French or just learning where the spaces go.

Checkpoints store the architecture next to the weights, so a saved model stays loadable even if you change `config.json` afterwards.

## Generate

```bash
python src/gutenberg-gpt/inference.py
```

This loads the checkpoint and drops you into a small interactive prompt. Type a sentence, press Enter, and the model writes the next 100 tokens right underneath. Each round starts again from everything on screen, your words and the model's, so you steer the story by adding a line whenever the continuation drifts. Enter sends, and an empty line quits, along with Ctrl-C and Ctrl-D. There is no way to ask for another round without typing something, so keep a word in reserve when you only want more text.

Generation can also stop short, when the model draws the end-of-text token and decides the document is over. Temperature and top-k sit in the `inference` function at the top of the file.

Two flags, both optional:

```bash
python src/gutenberg-gpt/inference.py -nt 300 -o artifacts/output.txt
```

`-nt / --nbtokens` sets how many tokens each round generates, 100 by default. `-o / --output` saves the text, rewriting the file after every round with everything so far. Without it nothing is written to disk.

## Configuration

Everything tunable sits in `config.json`.

| key | current | what it does |
| --- | --- | --- |
| `model.context_length` | 128 | how many tokens the model can look back |
| `model.n_embed` | 192 | width of the residual stream |
| `model.n_layer` | 12 | number of transformer blocks |
| `model.n_head` | 12 | attention heads per block, must divide `n_embed` |
| `model.dropout_prob` | 0.0 | dropout in attention and the feed-forward |
| `training.max_iters` | 100000 | total training steps |
| `training.eval_interval` | 1000 | steps between loss evaluations |
| `training.eval_iters` | 50 | batches averaged per evaluation |
| `training.batch_size` | 16 | sequences per step |
| `training.learning_rate` | 3e-4 | AdamW learning rate |
| `training.train_split` | 0.9 | share of book files used for training |
| `generation.prompt` | `"Rappelez vous, "` | prompt for the samples printed during training |
| `generation.max_new_tokens` | 100 | length of those samples |

Paths are separate, and all optional. Copy `.env.example` to `.env` if you want to override where the config, the corpus, the `.bin` files, the tokenizer or the checkpoint live. A relative value there is resolved from the directory you launch the script in, not from the repo root, so use an absolute path unless you always start from the root.

## Credits

Andrej Karpathy's [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) was a very helpful resource to build this project.

The texts come from [Project Gutenberg](https://www.gutenberg.org/).

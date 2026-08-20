import torch
from model import LanguageModel
from gutenberg_tokenizer import GutenbergTokenizer
from paths import MODEL_PATH, TOKENIZER_PATH

torch.manual_seed(1337)

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GutenbergTokenizer().load(str(TOKENIZER_PATH))
vocabulary_size = tokenizer.VOCABULARY_SIZE


checkpoint = torch.load(MODEL_PATH, map_location=device)

model = LanguageModel(
    vocab_size=vocabulary_size,
    n_head=checkpoint["n_head"],
    n_layer=checkpoint["n_layer"],
    context_length=checkpoint["context_length"],
    n_embed=checkpoint["n_embed"],
    dropout_prob=checkpoint["dropout_prob"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])

# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

model.eval()


sentence = """L'histoire que je vais vous raconter est celle d'un vieil homme, il avait en"""

idx = torch.tensor([tokenizer.encode(sentence)], dtype=torch.long, device=device)


print(tokenizer.decode(model.generate(idx, 500)[0].tolist()))

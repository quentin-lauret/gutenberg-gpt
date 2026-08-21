import torch
from model import LanguageModel
from gutenberg_tokenizer import GutenbergTokenizer
from paths import MODEL_PATH, TOKENIZER_PATH, ARTIFACTS_DIRECTORY

#torch.manual_seed(1337)

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


sentence = """A force d'aller en avant, il parvint au point où le brouillard de la fusillade devenait transparent. Si bien que les tirailleurs de la ligne rangés et à l'affût derrière leur levée de pavés, et les tirailleurs de la banlieue massés à l'angle de la rue, se montrèrent soudainement quelque chose qui remuait dans la fumée.Au moment où Gavroche débarrassait de ses cartouches un sergent gisant près d'une borne, une balle frappa le cadavre.- Fichtre ! dit Gavroche. Voilà qu'on me tue mes morts. Une deuxième balle fit étinceler le pavé à côté de lui. Une troisième renversa son panier. Gavroche regarda, et vit que cela venait de la banlieue."""

idx = torch.tensor([tokenizer.encode(sentence)], dtype=torch.long, device=device)

print("Parameters :", sum(p.numel() for p in model.parameters()))

text = tokenizer.decode(model.generate(idx, 100, EOT=tokenizer.EOT, temperature=1, top_k=10)[0].tolist())
with open(ARTIFACTS_DIRECTORY/"output.txt", "w") as file:
    file.write(text)

import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, head_size, context_length=256, n_embed=384, dropout_prob=0.2):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout_prob)
        self.register_buffer(
            "tril", torch.tril(torch.ones((context_length, context_length)))
        )

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)  # x @ W_k | (B, T, C) @ (C, 16) -> (B, T, 16)
        q = self.query(x)  # x @ W_q | (B, T, C) @ (C, 16) -> (B, T, 16)
        v = self.value(x)  # x @ W_v | (B, T, C) @ (C, 16) -> (B, T, 16)
        # (B, T, 16) @ (B, 16, T)
        wei = q @ k.transpose(-2, -1) * 1 / (self.head_size ** (1 / 2))  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = nn.functional.softmax(wei, dim=-1)

        wei = self.dropout(wei)

        out = wei @ v  # (B, T, T) @ (B, T, 16) -> (B, T, 16)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, context_length=256, n_embed=384, dropout_prob=0.2):
        super().__init__()
        assert n_embed % num_head == 0, "n_embed must be divisible by num_head"
        head_size = n_embed // num_head
        self.heads = nn.ModuleList(
            [
                Head(head_size, context_length, n_embed, dropout_prob)
                for i in range(num_head)
            ]
        )
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], -1)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout_prob=0.2):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        return self.nn(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head, context_length=256, dropout_prob=0.2):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            n_head,
            context_length=context_length,
            n_embed=n_embed,
            dropout_prob=dropout_prob,
        )
        self.ffn = FeedForward(n_embed=n_embed, dropout_prob=dropout_prob)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))  # (B, T, embed_size)
        x = x + self.ffn(self.layer_norm2(x))  # (B, T, embed_size)
        return x


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_head,
        n_layer,
        context_length=256,
        n_embed=384,
        dropout_prob=0.2,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(context_length, n_embed)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embed,
                    n_head=n_head,
                    context_length=context_length,
                    dropout_prob=dropout_prob,
                )
                for _ in range(n_layer)
            ]
        )
        self.layer_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        # idx : (B, T)
        # targets : (B, T)
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C) where C is embed_size
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape

        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, EOT=None, top_k=10, temperature=1):
        self.eval()
        i = 0
        while i < max_new_tokens and (EOT is None or idx[-1, -1] != EOT):
            i += 1
            
            idx_cropped = idx[:, -self.context_length :]

            logits, _ = self(idx_cropped)  # (B, T, C)

            logits = logits[:, -1, :].float() / temperature # (B, C)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
      

            probs = F.softmax(logits, dim=-1)  # (B, C)

            idx_next = torch.multinomial(
                probs, num_samples=1
            )

            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            #print(f"Token number {i} : {idx[-1, -1]}")
        return idx

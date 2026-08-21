import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    """
    A single causal self-attention head.

    Keeps its own key, query and value projections and its own causal mask, so
    a head can be read on its own without untangling a fused qkv projection.

    Parameters
    ----------
    head_size : int
        Width of the key, query and value projections.
    context_length : int, optional
        Longest sequence the head can attend over, which is the size of the
        registered causal mask.
    n_embed : int, optional
        Width of the incoming embeddings.
    dropout_prob : float, optional
        Dropout applied to the attention weights.
    """

    def __init__(
        self,
        head_size: int,
        context_length: int = 256,
        n_embed: int = 384,
        dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout_prob)
        self.register_buffer(
            "tril", torch.tril(torch.ones((context_length, context_length)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Attend over the sequence.

        Parameters
        ----------
        x : torch.Tensor
            Embeddings of shape ``(B, T, n_embed)``.

        Returns
        -------
        torch.Tensor
            Attention output of shape ``(B, T, head_size)``. Positions can only
            look at themselves and at what comes before them.
        """
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
    """
    Several attention heads run side by side, concatenated then projected.

    The embedding is split evenly across the heads, so each one works on
    ``n_embed // num_head`` dimensions and the concatenation is back to
    ``n_embed``.

    Parameters
    ----------
    num_head : int
        Number of heads. Must divide `n_embed`.
    context_length : int, optional
        Longest sequence the heads can attend over.
    n_embed : int, optional
        Width of the incoming embeddings, and of the output projection.
    dropout_prob : float, optional
        Dropout applied inside each head and after the output projection.

    Raises
    ------
    AssertionError
        If `n_embed` is not divisible by `num_head`.
    """

    def __init__(
        self,
        num_head: int,
        context_length: int = 256,
        n_embed: int = 384,
        dropout_prob: float = 0.2,
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run every head and merge their outputs.

        Parameters
        ----------
        x : torch.Tensor
            Embeddings of shape ``(B, T, n_embed)``.

        Returns
        -------
        torch.Tensor
            Projected attention output of shape ``(B, T, n_embed)``.
        """
        out = torch.cat([head(x) for head in self.heads], -1)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    """
    The position-wise MLP of a transformer block, with a 4x inner width.

    Parameters
    ----------
    n_embed : int
        Width of the input and of the output. The hidden layer is four times
        wider.
    dropout_prob : float, optional
        Dropout applied to the output.
    """

    def __init__(self, n_embed: int, dropout_prob: float = 0.2) -> None:
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the MLP to each position on its own.

        Parameters
        ----------
        x : torch.Tensor
            Embeddings of shape ``(B, T, n_embed)``.

        Returns
        -------
        torch.Tensor
            Same shape as `x`.
        """
        return self.nn(x)


class Block(nn.Module):
    """
    A pre-norm transformer block: attention then feed-forward, both residual.

    Parameters
    ----------
    n_embed : int
        Width of the embeddings flowing through the block.
    n_head : int
        Number of attention heads. Must divide `n_embed`.
    context_length : int, optional
        Longest sequence the attention can attend over.
    dropout_prob : float, optional
        Dropout used in both sublayers.
    """

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        context_length: int = 256,
        dropout_prob: float = 0.2,
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Take the input through both sublayers.

        Parameters
        ----------
        x : torch.Tensor
            Embeddings of shape ``(B, T, n_embed)``.

        Returns
        -------
        torch.Tensor
            Same shape as `x`. Each sublayer normalises its input, then adds its
            result back to the residual stream.
        """
        x = x + self.self_attention(self.layer_norm1(x))  # (B, T, embed_size)
        x = x + self.ffn(self.layer_norm2(x))  # (B, T, embed_size)
        return x


class LanguageModel(nn.Module):
    """
    The decoder-only GPT: token and position embeddings, n_layer blocks, a head.

    Every hyperparameter is passed in, so this module never reads config.json
    and a checkpoint can rebuild the exact architecture it was trained with.

    Parameters
    ----------
    vocab_size : int
        Size of the tokenizer vocabulary, which is both the number of token
        embeddings and the width of the output logits.
    n_head : int
        Number of attention heads per block. Must divide `n_embed`.
    n_layer : int
        Number of transformer blocks stacked on top of each other.
    context_length : int, optional
        Longest sequence the model can see, which is the number of learned
        position embeddings.
    n_embed : int, optional
        Width of the residual stream.
    dropout_prob : float, optional
        Dropout used throughout the blocks.

    Attributes
    ----------
    context_length : int
        Kept on the module because `generate` needs it to crop the context.
    """

    def __init__(
        self,
        vocab_size: int,
        n_head: int,
        n_layer: int,
        context_length: int = 256,
        n_embed: int = 384,
        dropout_prob: float = 0.2,
    ) -> None:
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

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Score a batch of token ids, and compute the loss when targets are given.

        Parameters
        ----------
        idx : torch.Tensor
            Token ids of shape ``(B, T)``, with ``T <= context_length``.
        targets : torch.Tensor, optional
            Token ids of shape ``(B, T)``, `idx` shifted one position to the
            left. Leave it out to score without a loss.

        Returns
        -------
        logits : torch.Tensor
            ``(B, T, vocab_size)`` without targets, flattened to
            ``(B * T, vocab_size)`` with them, which is the shape cross entropy
            wants.
        loss : torch.Tensor or None
            Mean cross entropy over the batch, or None when `targets` is None.
        """
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
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        EOT: int | None = None,
        top_k: int | None = 10,
        temperature: float = 1,
    ) -> torch.Tensor:
        """
        Sample tokens one at a time and append them to the context.

        Only the last `context_length` tokens are fed back in, since the
        position embeddings go no further. Switches the model to eval mode and
        leaves it there.

        Parameters
        ----------
        idx : torch.Tensor
            Prompt token ids of shape ``(B, T)``.
        max_new_tokens : int
            How many tokens to sample at most.
        EOT : int, optional
            End-of-text id. Sampling stops as soon as it comes up, if one is
            passed.
        top_k : int, optional
            Keep the k most likely tokens at each step and drop the rest. None
            samples from the whole vocabulary.
        temperature : float, optional
            Below 1 sharpens the distribution, above 1 flattens it.

        Returns
        -------
        torch.Tensor
            Token ids of shape ``(B, T + n)``, with the prompt still at the
            front and ``n <= max_new_tokens``.
        """
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

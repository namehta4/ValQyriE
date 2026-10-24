"""Decoder-only transformer (GPT) used as the GQE circuit generator. CPU-only PyTorch."""
from __future__ import annotations

import torch
from torch import nn


class GPT(nn.Module):
    """Minimal causal transformer over operator-pool tokens.

    Token ``vocab_size`` is the begin-of-sequence token.  The output head returns one
    logit w_j per pool token; following GPT-QE, tokens are sampled with probability
    proportional to exp(-beta * w_j), so *low* logits mean *preferred* operators.
    """

    def __init__(self, vocab_size: int, max_len: int, n_layer: int = 2, n_head: int = 4,
                 n_embd: int = 64, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.bos = vocab_size
        self.tok_emb = nn.Embedding(vocab_size + 1, n_embd)
        self.pos_emb = nn.Embedding(max_len + 1, n_embd)
        layer = nn.TransformerEncoderLayer(n_embd, n_head, dim_feedforward=4 * n_embd, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        pos = torch.arange(t, device=idx.device)
        h = self.tok_emb(idx) + self.pos_emb(pos)[None]
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=idx.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.head(self.ln_f(h))

    @torch.no_grad()
    def generate(self, batch: int, length: int, beta: float, generator: torch.Generator) -> torch.Tensor:
        seq = torch.full((batch, 1), self.bos, dtype=torch.long)
        for _ in range(length):
            logits = self(seq)[:, -1, :]
            probs = torch.softmax(-beta * logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
            seq = torch.cat([seq, nxt], dim=1)
        return seq[:, 1:]

    def sequence_logits(self, seqs: torch.Tensor) -> torch.Tensor:
        """Logits w of the chosen tokens, shape (batch, length), with gradients."""
        bos = torch.full((seqs.shape[0], 1), self.bos, dtype=torch.long)
        logits = self(torch.cat([bos, seqs[:, :-1]], dim=1))
        return logits.gather(-1, seqs.unsqueeze(-1)).squeeze(-1)

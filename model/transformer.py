"""
Амика · трансформър (архитектура от типа GPT): v1 върху знаци, v2 върху токени от BPE.
Моделът получава редица от токени и за всяка позиция предсказва следващия токен.
Всичко е написано върху PyTorch от нулата: embedding, внимание, MLP, нормализация.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int        # брой токени: различните знаци при v1, 8192 при v2
    block_size: int = 256  # колко токена назад вижда моделът (контекстен прозорец)
    n_layer: int = 4       # брой блокове един след друг
    n_head: int = 4        # брой глави на вниманието във всеки блок
    n_embd: int = 128      # ширина на модела: дължина на вектора за всеки знак
    dropout: float = 0.1   # случайно заглушаване на част от сигналите при обучение, срещу наизустяване
    tie_weights: bool = False   # v2: изходният слой ползва таблицата на вграждането, пести vocab × n_embd параметри


class CausalSelfAttention(nn.Module):
    """Вниманието: всяка позиция събира информация от предишните, претеглена по важност."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # от всеки вектор: заявка, ключ и стойност
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        # маска: позиция i вижда само позиции до i включително, бъдещето е скрито
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape                                  # партида, дължина, ширина
        q, k, v = self.qkv(x).split(C, dim=2)
        # разделяме ширината на n_head глави; всяка глава гледа различен аспект
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, глави, T, C/глави)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # колко всяка позиция „гледа“ всяка предишна: скаларно произведение заявка·ключ
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)                       # тегла, които по ред се сумират до 1
        att = self.dropout(att)
        y = att @ v                                        # претеглена сума на стойностите
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # събираме главите обратно
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    """Малка мрежа за всяка позиция поотделно: тук моделът „мисли“ върху събраното от вниманието."""

    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Един блок: внимание, после MLP, всяко с остатъчна връзка (x + ...) и нормализация преди него."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Amica(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # вектор за всеки знак
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # вектор за всяка позиция
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)   # от вектор към logits за всеки знак
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight   # една и съща таблица чете токените и ги предсказва
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=idx.device)
        x = self.dropout(self.tok_emb(idx) + self.pos_emb(pos))   # знак + позиция
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))                   # (B, T, vocab)
        loss = None
        if targets is not None:
            # cross-entropy: както в биграма, -log на вероятността за верния следващ знак
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        # същият цикъл като при биграма: гледаш последните знаци, теглиш следващия, повтаряш
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]       # само последните block_size знака
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature       # предсказанието за следващия знак
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
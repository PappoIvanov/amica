"""
Амика v2 · обвивка около Amica за работа с текст през BPE токенизатора.
Същият интерфейс като CharTransformer и BigramCounts (load и generate), затова app/chat.py
не знае кой модел е вътре. Текстът става номера на токени с tokenizer/bpe.py и обратно.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))        # за да намери transformer.py
sys.path.insert(0, str(ROOT / "tokenizer"))    # за да намери bpe.py
from transformer import Amica, Config
from bpe import BPE


class TokenTransformer:
    def __init__(self, model, bpe, temperature=0.8):
        self.model = model
        self.bpe = bpe
        self.doc = bpe.special["<|doc|>"]      # служебният токен за край (и начало) на документ
        self.temperature = temperature         # под 1: по-предпазливи проби; над 1: по-смели и по-хаотични

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu")
        model = Amica(Config(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()                           # изключва dropout: при разговор нищо не се заглушава
        bpe = BPE.load(ROOT / "tokenizer" / ckpt["tokenizer"])   # името е записано при обучението
        return cls(model, bpe)

    def generate(self, prefix, n_tokens):
        # документът започва след <|doc|>, както в обучението; крайният интервал на префикса пада,
        # защото при BPE думите носят интервала си отпред (" кон"), а самотен интервал е рядък токен
        ids = [self.doc] + self.bpe.encode(prefix.rstrip(" "))
        idx = torch.tensor([ids], dtype=torch.long)
        out = self.model.generate(idx, n_tokens, temperature=self.temperature)[0].tolist()
        new = out[len(ids):]                   # само новите токени
        if self.doc in new:
            new = new[:new.index(self.doc)]    # моделът е сложил край на документа: спираме там
        continuation = self.bpe.decode(new)
        if prefix.endswith(" ") and continuation.startswith(" "):
            continuation = continuation[1:]    # интервалът вече го има в префикса
        return prefix + continuation

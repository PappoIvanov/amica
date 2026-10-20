"""
Амика v1 * обвивка около Amica за работа с текст.
Amica приема и връща тензори с номера на знаци; приложението работи с низове.
Класът превежда между двете и има същия интерфейс като BigramCounts
(load и generate), затова app/chat.py не знае кой модел е вътре.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))   # за да намери transformer.py
from transformer import Amica, Config


class CharTransformer:
    def __init__(self, model, chars, temperature=0.8):
        self.model = model
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.temperature = temperature   # под 1: по-предпазливи проби; над 1: по-смели и по-хаотични

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu")
        model = Amica(Config(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()   # изключва dropout: при разговор нищо не се заглушава
        return cls(model, ckpt["chars"])

    def generate(self, prefix, n_chars):
        # знаци, които не са били в корпуса (емоджи, редки букви), се пропускат
        ids = [self.stoi[ch] for ch in prefix if ch in self.stoi]
        if not ids:
            ids = [self.stoi["\n"]]
        idx = torch.tensor([ids], dtype=torch.long)
        out = self.model.generate(idx, n_chars, temperature=self.temperature)[0].tolist()
        new = out[len(ids):]   # само новите знаци; връщаме оригиналния префикс плюс тях
        return prefix + "".join(self.chars[i] for i in new)
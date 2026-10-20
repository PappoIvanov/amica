"""
Амика v0 * невронен биграм.
Същата таблица V x V като при броенето, но сега тя е матрица от тегла W,
която учим с градиентно спускане, както в train_toy.py. Целта: същият loss
като при броенето, но по път, който работи и за много по-големи модели.
Пускане от папката на проекта: py learn/02_bigram/train_bigram_nn.py
"""
import sys
import random
from pathlib import Path

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(42)
random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
text = (ROOT / "data" / "raw" / "prikazki.txt").read_text(encoding="utf-8")
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
V = len(chars)

# 1. Данните: всяка двойка съседни знаци е един пример. xs = знакът, ys = следващият.
xs = torch.tensor([stoi[a] for a in text[:-1]])
ys = torch.tensor([stoi[b] for b in text[1:]])
print(f"examples: {len(xs)}     dictionary: {V}")

# 2. "Мрежата": един слой без bias. Входът е one-hot вектор с дължина V
#    (единица на мястото на знака, нули другаде), изходът са V числа (logits),
#    по едно за всеки възможен следващ знак. Така W[a] е редът за знака a, 
#    точно както редът в таблицата с броячите, само че започва от случайни числа.
W = torch.randn((V, V), requires_grad=True)
x_onehot = F.one_hot(xs, num_classes=V).float()

lr = 500.0   # голяма стъпка, защото градиентът е усреднен по десетки хиляди примера
for step in range(200):
    # forward
    logits = x_onehot @ W                                   # (брой примери, V)
    counts = logits.exp()                                   # "броячи": положителни и учени
    probs = counts / counts.sum(1, keepdim=True)            # всеки ред става вероятности (softmax)
    # loss: средно -log на вероятността за верния следващ знак (cross-entropy),
    # плюс малко наказание за големи тегла, същата роля като +1 при броенето
    loss = -probs[torch.arange(len(xs)), ys].log().mean() + 0.01 * (W ** 2).mean()

    # backward и стъпките: същите три реда както в train_toy
    W.grad = None
    loss.backward()
    W.data -= lr * W.grad

    if step % 20 == 0 or step == 199:
        print(f"step: {step:3d}   loss = {loss.item():.4f}")

# 3. Генериране: както при броенето, но вероятностите идват от W
probs = (W.exp() / W.exp().sum(1, keepdim=True)).detach()


def generate(prefix, n_chars):
    out = prefix
    ch = prefix[-1] if prefix and prefix[-1] in stoi else "\n"
    for _ in range(n_chars):
        i = torch.multinomial(probs[stoi[ch]], num_samples=1).item()
        ch = itos[i]
        out += ch
    return out

for _ in range(2):
    print("---")
    print(generate("\n", 200))

torch.save({"chars": chars, "W": W.detach()}, ROOT / "checkpoints" / "amica-v0-bigram-nn.pt")
print("saved: amica-v0-bigram-nn.pt")
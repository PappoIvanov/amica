"""
Първо обучение: мрежа 3 -> 4 -> 4 -> 1 върху 4 примера.
Пускане: py learn\01_micrograd\train_toy.py
"""
import sys
import random

from nn import MLP

sys.stdout.reconfigure(encoding="utf-8")
random.seed(42)   # едни и същи слуайни числа при всяко пускане

# 4 примера по 3 входа и желан изход за всеки: +1 или -1
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1])
print("parameters:", len(model.parameters()))

lr = 0.05   # скорост на учене: колко голяма стъпка правим по градиента

for step in range(50):
    # 1. forward: предсказания и грешка (средноквадратично отклонение)
    preds = [model(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys)) / len(xs)

    # 2. нулираме старите градиенти, иначе += в _backward ги трупа между стъпките
    for p in model.parameters():
        p.grad = 0.0

    # 3. backward: градиент на loss спрямо всеки параметър
    loss.backward()

    # 4. стъпка срещу градиента: минус, защото искаме loss да намалява
    for p in model.parameters():
        p.data -= lr * p.grad

    if step % 10 == 0 or step == 49:
        print(f"step {step:2d}   loss = {loss.data:.4f}")

preds = [model(x) for x in xs]
print("predictions:", [round(p.data, 3) for p in preds])
print("targeted:   ", ys)
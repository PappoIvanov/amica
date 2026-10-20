"""
Обучение на Амика v0: символичен биграм върху data/raw/prikazki.txt.
Пускане от папката на проекта: py learn/02_bigram/train_bigram.py
"""
import sys
import math
import random

from pathlib import Path

from bigram import BigramCounts

sys.stdout.reconfigure(encoding="utf-8")
random.seed(42)

ROOT = Path(__file__).resolve().parents[2]   # папката на проекта, две нива нагоре
text = (ROOT / "data" / "raw" / "prikazki.txt").read_text(encoding="utf-8")
print(f"text: {len(text)} signs")

model = BigramCounts()
model.fit(text)
V = len(model.chars)
print(f"dictionary: {V} different signs")
# случайно познаване = всеки знак с вероятност 1/V, тоест loss = ln(V)
print(f"loss on the text: {model.loss(text):.3f}       (casual acquaintace: {math.log(V):.3f})")

for _ in range(3):
    print("---")
    print(model.generate("\n", 200))

out = ROOT / "checkpoints" / "amica-v0-bigram.json"
model.save(out)
print(f"saved: {out.name}")
m2 = BigramCounts.load(out)
print(f"check after loading, loss: {m2.loss(text):.3f}")
"""
Амика * терминален чат.
Приложението не знае нищо за модела освен метода generate(prefix, n_chars).
v0 е биграм, v1 е символен трансформър, v2 е трансформър върху BPE токени; този файл остава същият.
Пускане от папката на проекта: py app/chat.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "learn" / "02_bigram"))   # за да намери bigram.py
sys.path.insert(0, str(ROOT / "model"))                  # за да намери char_model.py
from bigram import BigramCounts
from char_model import CharTransformer
from token_model import TokenTransformer

sys.stdout.reconfigure(encoding="utf-8")

# сменяем модел: py app/chat.py v0 (биграм), v1 (символен трансформър) или v2 (BPE трансформър, по подразбиране) или v3
MODELS = {
    "v0": (BigramCounts, ROOT / "checkpoints" / "amica-v0-bigram.json", "char bigram"),
    "v1": (CharTransformer, ROOT / "checkpoints" / "amica-v1-char.pt", "char transformer"),
    "v2": (TokenTransformer, ROOT / "checkpoints" / "amica-v2.pt", "BPE transformer"),
    "v3": (TokenTransformer, ROOT / "checkpoints" / "amica-v3.pt", "BPE transformer v3"),
}
version = sys.argv[1] if len(sys.argv) > 1 else "v3"
cls, path, label = MODELS[version]
model = cls.load(path)

print(f"Amica {version} ({label}). Write something; empty row for end.")

while True:
    try:
        user_text = input("you: ")
    except (EOFError, KeyboardInterrupt):
        break
    if not user_text.strip():
        break
    # моделът продължава твоя текст; показваме само продължението
    reply = model.generate(user_text + " ", 200)[len(user_text) + 1:]
    print("Amica:", reply)
print("Bye.")
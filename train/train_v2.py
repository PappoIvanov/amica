"""
Амика v2 · обучение на трансформъра върху токените от BPE.

Данните са шестте .bin файла от tokenizer/encode_corpus.py (uint16, по група и дял). Четат се с
np.memmap, тоест без да се зареждат целите в паметта. Всяка партида се сглобява от трите групи по
теглата WEIGHTS, а загубата се мери за всяка група поотделно върху val файловете.
Пускане от папката на проекта: py train/train_v2.py
Настройки от средата, както при v1 (на Kaggle: TOKENS=/kaggle/input/datasets/pavel77ivanov/amica-bg/tokens):
  TOKENS папката с .bin файловете, RUN име на пускането, MAX_STEPS, EVAL_EVERY, EVAL_BATCHES,
  BLOCK_SIZE, BATCH_SIZE, N_LAYER, N_HEAD, N_EMBD, W_LIT, W_BG_WIKI, W_EN.
На процесора пълният размер прави около 20 секунди на стъпка; за проба (около 5 минути):
  $env:MAX_STEPS=10; $env:EVAL_EVERY=5; $env:EVAL_BATCHES=2
"""
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tokenizer"))
from transformer import Amica, Config
from bpe import BPE

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(42)
rng = np.random.default_rng(42)

# ---- настройки ----
TOKENS = Path(os.environ.get("TOKENS", ROOT / "data" / "clean" / "tokens"))
RUN = os.environ.get("RUN", time.strftime("%Y%m%d-%H%M"))
CHECKPOINT = ROOT / "checkpoints" / f"amica-v2-{RUN}.pt"
LOG = CHECKPOINT.with_suffix(".csv")     # кривата на loss, до checkpoint-а
GROUPS = ["lit", "bg_wiki", "en"]
WEIGHTS = {"lit": float(os.environ.get("W_LIT", 0.1)),         # дял от партидата за всяка група;
           "bg_wiki": float(os.environ.get("W_BG_WIKI", 0.6)), # литературата е 2.7% от корпуса,
           "en": float(os.environ.get("W_EN", 0.3))}           # тук се вдига на 10%
VOCAB_SIZE = 8192     # 8185 заети номера плюс 7 запазени; кръгло число за таблицата
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", 512))     # контекстен прозорец в токени
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))      # колко откъса наведнъж
N_LAYER = int(os.environ.get("N_LAYER", 6))
N_HEAD = int(os.environ.get("N_HEAD", 6))
N_EMBD = int(os.environ.get("N_EMBD", 384))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 2000))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 100))
EVAL_BATCHES = int(os.environ.get("EVAL_BATCHES", 10))
LR_MAX, LR_MIN, WARMUP = 1e-3, 1e-4, 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- данни ----
bpe = BPE.load(ROOT / "tokenizer" / "amica_bpe.json")
train_data = {g: np.memmap(TOKENS / f"{g}_train.bin", dtype=np.uint16, mode="r") for g in GROUPS}
val_data = {g: np.memmap(TOKENS / f"{g}_val.bin", dtype=np.uint16, mode="r") for g in GROUPS}
probs = np.array([WEIGHTS[g] for g in GROUPS])
probs = probs / probs.sum()              # теглата стават вероятности, сумата е 1
for g, p in zip(GROUPS, probs):
    print(f"{g:8} train {len(train_data[g]):>11,} tokens   val {len(val_data[g]):>9,} tokens   share {p:.2f}")
print(f"tokens per step: {BATCH_SIZE * BLOCK_SIZE:,}   device: {DEVICE}")


def windows(source, n):
    # n случайни откъса от един масив: x е откъсът, y е същият откъс, изместен с един токен напред
    ix = rng.integers(len(source) - BLOCK_SIZE - 1, size=n)
    chunks = torch.stack([torch.from_numpy(source[i:i + BLOCK_SIZE + 1].astype(np.int64)) for i in ix])
    x, y = chunks[:, :-1].contiguous(), chunks[:, 1:].contiguous()   # плътни копия: view(-1) в модела го иска
    return x.to(DEVICE), y.to(DEVICE)


def get_batch():
    # партида от трите групи: колко откъса от всяка се тегли по вероятностите, средно 10/60/30
    counts = rng.multinomial(BATCH_SIZE, probs)
    parts = [windows(train_data[g], n) for g, n in zip(GROUPS, counts) if n > 0]
    return torch.cat([x for x, _ in parts]), torch.cat([y for _, y in parts])


@torch.no_grad()
def estimate_loss(model):
    # средна грешка: train върху смесени партиди, val за всяка група поотделно
    model.eval()
    out = {"train": sum(model(*get_batch())[1].item() for _ in range(EVAL_BATCHES)) / EVAL_BATCHES}
    for g in GROUPS:
        losses = [model(*windows(val_data[g], BATCH_SIZE))[1].item() for _ in range(EVAL_BATCHES)]
        out[f"val_{g}"] = sum(losses) / EVAL_BATCHES
    model.train()
    return out


def lr_at(step):
    # загрявка: линейно от 0 до LR_MAX; после косинусово спадане до LR_MIN
    if step < WARMUP:
        return LR_MAX * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))


# ---- модел ----
cfg = Config(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE, n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
             tie_weights=True)
model = Amica(cfg).to(DEVICE)
print(f"parameters: {model.count_parameters():,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=0.1)
LOG.write_text("step,train,val_lit,val_bg_wiki,val_en,lr,seconds\n")

# ---- цикъл: същите четири стъпки като при v1 ----
t0 = time.time()
for step in range(MAX_STEPS):
    for group in optimizer.param_groups:
        group["lr"] = lr_at(step)
    if step % EVAL_EVERY == 0 or step == MAX_STEPS - 1:
        losses = estimate_loss(model)
        print(f"step {step:5d}  train {losses['train']:.3f}  val lit {losses['val_lit']:.3f}  "
              f"bg_wiki {losses['val_bg_wiki']:.3f}  en {losses['val_en']:.3f}  "
              f"lr {lr_at(step):.2e}  {time.time() - t0:.0f}s")
        with LOG.open("a") as f:
            f.write(f"{step},{losses['train']:.4f},{losses['val_lit']:.4f},{losses['val_bg_wiki']:.4f},"
                    f"{losses['val_en']:.4f},{lr_at(step):.2e},{time.time() - t0:.0f}\n")
    x, y = get_batch()
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # ограничава стъпката при резки градиенти
    optimizer.step()

# ---- запис и проба ----
torch.save({"config": cfg.__dict__, "tokenizer": "amica_bpe.json", "state_dict": model.state_dict()}, CHECKPOINT)
print(f"saved: {CHECKPOINT.name}")
model.eval()
start = torch.tensor([[bpe.special["<|doc|>"]]], dtype=torch.long, device=DEVICE)   # начало на нов документ
sample = model.generate(start, 200, temperature=0.8)[0].tolist()
print("---")
print(bpe.decode(sample))

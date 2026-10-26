"""
Амика v3 · обучение на трансформъра върху корпуса от 2026-10-26 (287M токена в четири групи).

Същият цикъл като train_v2.py с четири промени, всяка отбелязана с „v3:“ в кода:
  1. четири групи данни (lit, bg_wiki, en, law) с тегла в партидата и val за всяка;
  2. смесена точност: умноженията са във float16 под torch.autocast, а GradScaler пази градиентите
     от занулявяне (T4 няма bfloat16); 2 до 3 пъти по-бързо, същият резултат;
  3. вниманието през F.scaled_dot_product_attention (Config.fast_attention=True): същата математика,
     но готово ядро, което не държи матрицата T × T в паметта; при контекст 1024 това е разликата
     между „става“ и „не се побира“;
  4. междинен запис на всеки SAVE_EVERY стъпки и продължаване от него с RESUME, защото Kaggle спира
     сесията на 12 часа.
Данните са осемте .bin файла от tokenizer/encode_corpus.py (uint16, 548 MB), заредени изцяло в паметта.
Пускане от папката на проекта: py train/train_v3.py
Настройки от средата (на Kaggle: TOKENS=/kaggle/input/datasets/pavel77ivanov/amica-bg/tokens):
  TOKENS, RUN, MAX_STEPS, EVAL_EVERY, EVAL_BATCHES, SAVE_EVERY, RESUME (път до .pt, от който да продължи),
  BLOCK_SIZE, BATCH_SIZE, N_LAYER, N_HEAD, N_EMBD, LR_MAX, W_LIT, W_BG_WIKI, W_EN, W_LAW.
Проба на процесора (около 2 минути, само дали върви):
  $env:MAX_STEPS='3'; $env:EVAL_EVERY='2'; $env:EVAL_BATCHES='1'; $env:N_LAYER='2'; $env:N_EMBD='64'; $env:N_HEAD='2'; $env:BLOCK_SIZE='128'; $env:BATCH_SIZE='4'; py train/train_v3.py
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
CHECKPOINT = ROOT / "checkpoints" / f"amica-v3-{RUN}.pt"
LOG = CHECKPOINT.with_suffix(".csv")     # кривата на loss, до checkpoint-а
RESUME = os.environ.get("RESUME")        # v3: път до междинен запис, от който да продължи
GROUPS = ["lit", "bg_wiki", "en", "law"]                       # v3: четвъртата група са законите и OPUS bg
WEIGHTS = {"lit": float(os.environ.get("W_LIT", 0.08)),        # дял от партидата за всяка група;
           "bg_wiki": float(os.environ.get("W_BG_WIKI", 0.52)),  # литературата е 2.2% от токените, тук 8%;
           "en": float(os.environ.get("W_EN", 0.32)),           # законите и официалните текстове са 8%,
           "law": float(os.environ.get("W_LAW", 0.08))}         # колкото са, за да не оцветят езика
VOCAB_SIZE = 8192
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", 1024))    # v3: двойно по-дълъг прозорец
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))      # 16 × 1024 = същите 16k токена на стъпка като v2
N_LAYER = int(os.environ.get("N_LAYER", 10))
N_HEAD = int(os.environ.get("N_HEAD", 8))
N_EMBD = int(os.environ.get("N_EMBD", 512))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 30000))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 500))
EVAL_BATCHES = int(os.environ.get("EVAL_BATCHES", 10))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 5000))   # v3: междинен запис
LR_MAX = float(os.environ.get("LR_MAX", 1e-3))
LR_MIN, WARMUP = LR_MAX / 10, 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"                              # v3: смесена точност само на GPU

# ---- данни ----
bpe = BPE.load(ROOT / "tokenizer" / "amica_bpe.json")
train_data = {g: np.fromfile(TOKENS / f"{g}_train.bin", dtype=np.uint16) for g in GROUPS}   # целите в RAM
val_data = {g: np.fromfile(TOKENS / f"{g}_val.bin", dtype=np.uint16) for g in GROUPS}
probs = np.array([WEIGHTS[g] for g in GROUPS])
probs = probs / probs.sum()
tokens_per_step = BATCH_SIZE * BLOCK_SIZE
for g, p in zip(GROUPS, probs):
    passes = MAX_STEPS * tokens_per_step * p / len(train_data[g])   # колко пъти ще мине през групата
    print(f"{g:8} train {len(train_data[g]):>12,} tokens   val {len(val_data[g]):>10,}   share {p:.2f}   passes {passes:.1f}")
print(f"tokens per step: {tokens_per_step:,}   steps: {MAX_STEPS:,}   device: {DEVICE}   mixed precision: {USE_AMP}")


def windows(source, n):
    # n случайни откъса от един масив: x е откъсът, y е същият откъс, изместен с един токен напред
    ix = rng.integers(len(source) - BLOCK_SIZE - 1, size=n)
    chunks = torch.stack([torch.from_numpy(source[i:i + BLOCK_SIZE + 1].astype(np.int64)) for i in ix])
    x, y = chunks[:, :-1].contiguous(), chunks[:, 1:].contiguous()
    return x.to(DEVICE), y.to(DEVICE)


def get_batch():
    # партида от четирите групи: колко откъса от всяка се тегли по вероятностите
    counts = rng.multinomial(BATCH_SIZE, probs)
    parts = [windows(train_data[g], n) for g, n in zip(GROUPS, counts) if n > 0]
    return torch.cat([x for x, _ in parts]), torch.cat([y for _, y in parts])


def forward_loss(model, x, y):
    # v3: под autocast умноженията са float16, загубата се смята във float32
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
        _, loss = model(x, y)
    return loss


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {"train": sum(forward_loss(model, *get_batch()).item() for _ in range(EVAL_BATCHES)) / EVAL_BATCHES}
    for g in GROUPS:
        losses = [forward_loss(model, *windows(val_data[g], BATCH_SIZE)).item() for _ in range(EVAL_BATCHES)]
        out[f"val_{g}"] = sum(losses) / EVAL_BATCHES
    model.train()
    return out


def lr_at(step):
    if step < WARMUP:
        return LR_MAX * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))


def save(model, step):
    torch.save({"config": cfg.__dict__, "tokenizer": "amica_bpe.json", "state_dict": model.state_dict(), "step": step}, CHECKPOINT)
    print(f"saved: {CHECKPOINT.name} (step {step})")


# ---- модел ----
cfg = Config(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE, n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
             tie_weights=True, fast_attention=True)      # v3: готовото ядро за вниманието
model = Amica(cfg).to(DEVICE)
print(f"parameters: {model.count_parameters():,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=0.1)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)  # v3: мащабира загубата, за да не се занулят малките градиенти във float16
start_step = 0
if RESUME:
    ckpt = torch.load(RESUME, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    start_step = ckpt.get("step", 0) + 1
    print(f"resumed from {RESUME} at step {start_step} (optimizer starts fresh, the lr schedule continues)")
    with LOG.open("a") as f:
        f.write(f"# resumed from {RESUME} at step {start_step}\n")
else:
    LOG.write_text("step,train,val_lit,val_bg_wiki,val_en,val_law,lr,seconds\n")

# ---- цикъл ----
t0 = time.time()
for step in range(start_step, MAX_STEPS):
    for group in optimizer.param_groups:
        group["lr"] = lr_at(step)
    if step % EVAL_EVERY == 0 or step == MAX_STEPS - 1:
        losses = estimate_loss(model)
        print(f"step {step:5d}  train {losses['train']:.3f}  val lit {losses['val_lit']:.3f}  "
              f"bg_wiki {losses['val_bg_wiki']:.3f}  en {losses['val_en']:.3f}  law {losses['val_law']:.3f}  "
              f"lr {lr_at(step):.2e}  {time.time() - t0:.0f}s")
        with LOG.open("a") as f:
            f.write(f"{step},{losses['train']:.4f},{losses['val_lit']:.4f},{losses['val_bg_wiki']:.4f},"
                    f"{losses['val_en']:.4f},{losses['val_law']:.4f},{lr_at(step):.2e},{time.time() - t0:.0f}\n")
    x, y = get_batch()
    loss = forward_loss(model, x, y)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()                        # v3: мащабираната загуба назад
    scaler.unscale_(optimizer)                           # обратно в истински мащаб преди ограничаването
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)                               # пропуска стъпката, ако градиентите са прелели
    scaler.update()
    if SAVE_EVERY and step > start_step and step % SAVE_EVERY == 0:
        save(model, step)                                # v3: междинен запис

# ---- запис и проба ----
save(model, MAX_STEPS - 1)
model.eval()
start = torch.tensor([[bpe.special["<|doc|>"]]], dtype=torch.long, device=DEVICE)
with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
    sample = model.generate(start, 200, temperature=0.8)[0].tolist()
print("---")
print(bpe.decode(sample))

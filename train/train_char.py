"""
Амика v1 · обучение на символния трансформър върху корпуса.
Пускане от папката на проекта: py train/train_char.py
На процесора върви бавно: около 2 секунди на стъпка, тоест около час за 2000 стъпки.
За бързи проби: $env:MAX_STEPS=100 в конзолата преди пускане.
"""
import sys
import math
import time
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from transformer import Amica, Config

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(42)

# ---- настройки ----
CORPUS = Path(os.environ.get("CORPUS", ROOT / "data" / "clean" / "corpus_public.txt"))   # на Kaggle: $CORPUS
RUN = os.environ.get("RUN", time.strftime("%Y%m%d-%H%M"))   # име на пускането: подадено или дата и час
CHECKPOINT = ROOT / "checkpoints" / f"amica-v1-char-{RUN}.pt"
LOG = CHECKPOINT.with_suffix(".csv")     # кривата на loss, до checkpoint-а, за графика след това
LOG.write_text("step,train,val,lr,seconds\n")
BLOCK_SIZE = 256      # контекстен прозорец в знаци
BATCH_SIZE = 32       # колко откъса наведнъж
N_LAYER, N_HEAD, N_EMBD = 4, 4, 128
MAX_STEPS = int(os.environ.get("MAX_STEPS", 2000))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 100))
LR_MAX, LR_MIN, WARMUP = 1e-3, 1e-4, 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- данни ----
text = CORPUS.read_text(encoding="utf-8")
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
split = int(0.95 * len(data))
train_data, val_data = data[:split], data[split:]   # последните 5% моделът не вижда при обучение
print(f"corpus: {len(text)} chars, vocab: {len(chars)}, train: {len(train_data)}, val: {len(val_data)}, device: {DEVICE}")


def get_batch(source):
    # случайни откъси с дължина BLOCK_SIZE; целта е същият откъс, изместен с един знак напред
    ix = torch.randint(len(source) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([source[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([source[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model, batches=20):
    # средна грешка върху няколко партиди от тренировъчния и от скрития текст
    model.eval()
    out = {}
    for name, source in [("train", train_data), ("val", val_data)]:
        losses = [model(*get_batch(source))[1].item() for _ in range(batches)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def lr_at(step):
    # загрявка: линейно от 0 до LR_MAX; после косинусово спадане до LR_MIN
    if step < WARMUP:
        return LR_MAX * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))


# ---- модел ----
cfg = Config(vocab_size=len(chars), block_size=BLOCK_SIZE, n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD)
model = Amica(cfg).to(DEVICE)
print(f"parameters: {model.count_parameters():,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=0.1)

# ---- цикъл: същите четири стъпки като в train_toy ----
t0 = time.time()
for step in range(MAX_STEPS):
    for group in optimizer.param_groups:
        group["lr"] = lr_at(step)
    if step % EVAL_EVERY == 0 or step == MAX_STEPS - 1:
        losses = estimate_loss(model)
        print(f"step {step:5d}  train {losses['train']:.3f}  val {losses['val']:.3f}  lr {lr_at(step):.2e}  {time.time() - t0:.0f}s")
        with LOG.open("a") as f:
                        f.write(f"{step},{losses['train']:.4f},{losses['val']:.4f},{lr_at(step):.2e},{time.time() - t0:.0f}\n")
    x, y = get_batch(train_data)
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # ограничава стъпката при резки градиенти
    optimizer.step()

# ---- запис и проба ----
torch.save({"config": cfg.__dict__, "chars": chars, "state_dict": model.state_dict()}, CHECKPOINT)
print(f"saved: {CHECKPOINT.name}")
model.eval()
start = torch.tensor([[stoi["\n"]]], dtype=torch.long, device=DEVICE)
sample = model.generate(start, 500, temperature=0.8)[0].tolist()
print("---")
print("".join(itos[i] for i in sample))
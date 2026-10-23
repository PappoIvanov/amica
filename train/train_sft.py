"""
Амика · етап 5 · SFT: дообучаване на v2 върху разговорите (supervised fine-tuning).

Тръгва от checkpoints/amica-v2.pt и продължава обучението върху data/clean/dialogues.jsonl.
Всеки разговор става токени и маска през tokenizer/chat_format.py; загубата се смята само там,
където маската е 1, тоест върху репликите на Амика. Потребителят, системата и резултатите са
само контекст. Към всяка стъпка се добавя и партида от предобучението (replay), за да не забрави
Амика езика, докато учи формата; тежестта ѝ е REPLAY_WEIGHT.
Партидата се тегли по ситуации, не по записи: всеки разговор има равен шанс, а от разгънатите
шаблони се взема случайно копие. Иначе 10-те шаблонни разговора с по 30 копия заглушават другите 36.
Разговорите с номер, завършващ на 7, не се учат: те са проверката (val) върху непознати ситуации.
Накрая моделът отговаря на четири пробни въпроса, за да се види какво е научил.

Пускане от папката на проекта: py train/sft.py          (на процесора около 15 минути за 300 стъпки)
Настройки от средата: RUN, MAX_STEPS, BATCH_SIZE, LR, REPLAY_WEIGHT, BASE (checkpoint), TOKENS (папка .bin).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tokenizer"))
from transformer import Amica, Config
from chat_format import load_tokenizer, encode_conversation, prompt_ids

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(42)
rng = np.random.default_rng(42)

# ---- настройки ----
BASE = Path(os.environ.get("BASE", ROOT / "checkpoints" / "amica-v2.pt"))
DIALOGUES = Path(os.environ.get("DIALOGUES", ROOT / "data" / "clean" / "dialogues.jsonl"))
TOKENS = Path(os.environ.get("TOKENS", ROOT / "data" / "clean" / "tokens"))
RUN = os.environ.get("RUN", time.strftime("%Y%m%d-%H%M"))
CHECKPOINT = ROOT / "checkpoints" / f"amica-v2-sft-{RUN}.pt"
LOG = CHECKPOINT.with_suffix(".csv")
MAX_STEPS = int(os.environ.get("MAX_STEPS", 300))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))         # разговори на стъпка
LR = float(os.environ.get("LR", 1e-4))                    # 10 пъти под предобучението: само донастройка
REPLAY_WEIGHT = float(os.environ.get("REPLAY_WEIGHT", 0.5))
REPLAY_BATCH, REPLAY_BLOCK = 4, 256                       # откъси от предобучението на всяка стъпка
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 50))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- данни ----
bpe = load_tokenizer()
END, DOC = bpe.special["<|end|>"], bpe.special["<|doc|>"]
records = [json.loads(line) for line in open(DIALOGUES, encoding="utf-8")]


def number_of(record):
    return int(record["id"].split("/")[1])


train_records = [r for r in records if number_of(r) % 10 != 7]
val_records = [r for r in records if number_of(r) % 10 == 7]
groups = {}                                               # номер на разговора → неговите записи (копия)
for r in train_records:
    groups.setdefault(number_of(r), []).append(r)
numbers = list(groups)
print(f"dialogues: {len(train_records)} train records in {len(numbers)} conversations, "
      f"{len(val_records)} val (numbers ending in 7), device: {DEVICE}")

replay = {}
if TOKENS.exists():
    for name in ["lit", "bg_wiki", "en"]:
        replay[name] = np.fromfile(TOKENS / f"{name}_train.bin", dtype=np.uint16)
else:
    print(f"warning: {TOKENS} not found, training without replay")
REPLAY_PROBS = {"lit": 0.1, "bg_wiki": 0.6, "en": 0.3}


def sample_batch():
    # равен шанс за всяка ситуация; при шаблон се взема случайно копие
    picked = rng.choice(numbers, size=BATCH_SIZE)
    return [groups[n][rng.integers(len(groups[n]))] for n in picked]


def make_batch(batch_records, block_size):
    # разговорите стават токени и маска; по-късите се допълват отдясно с <|end|> и маска 0,
    # което не пречи, защото вниманието гледа само назад
    encoded = [encode_conversation(bpe, r["turns"]) for r in batch_records]
    encoded = [(ids[:block_size], mask[:block_size]) for ids, mask in encoded]
    width = max(len(ids) for ids, _ in encoded)
    x = torch.full((len(encoded), width), END, dtype=torch.long)
    m = torch.zeros((len(encoded), width), dtype=torch.float)
    for i, (ids, mask) in enumerate(encoded):
        x[i, :len(ids)] = torch.tensor(ids)
        m[i, :len(mask)] = torch.tensor(mask, dtype=torch.float)
    # целта е същият ред, изместен с един токен; маската се мести заедно с нея
    return x[:, :-1].to(DEVICE), x[:, 1:].to(DEVICE), m[:, 1:].to(DEVICE)


def masked_loss(model, x, y, m):
    logits, _ = model(x)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    return (ce * m.reshape(-1)).sum() / m.sum().clamp(min=1)


def replay_batch():
    # откъси от предобучението, както в train_v2.py, за да не се забрави езикът
    names = rng.choice(list(REPLAY_PROBS), size=REPLAY_BATCH, p=list(REPLAY_PROBS.values()))
    chunks = []
    for name in names:
        source = replay[name]
        i = rng.integers(len(source) - REPLAY_BLOCK - 1)
        chunks.append(torch.from_numpy(source[i:i + REPLAY_BLOCK + 1].astype(np.int64)))
    chunks = torch.stack(chunks)
    return chunks[:, :-1].contiguous().to(DEVICE), chunks[:, 1:].contiguous().to(DEVICE)


@torch.no_grad()
def val_loss(model, block_size):
    model.eval()
    losses = [masked_loss(model, *make_batch(val_records[i:i + BATCH_SIZE], block_size)).item()
              for i in range(0, len(val_records), BATCH_SIZE)]
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def reply(model, turns, max_tokens=60, temperature=0.7):
    # една реплика на Амика: от разговора досега до <|end|>; връща текста ѝ, с маркера, ако има
    model.eval()
    ids = prompt_ids(bpe, turns)
    out = []
    for _ in range(max_tokens):
        idx = torch.tensor([ids[-model.cfg.block_size:]], dtype=torch.long, device=DEVICE)
        logits, _ = model(idx)
        probs = F.softmax(logits[0, -1] / temperature, dim=-1)
        token = torch.multinomial(probs, 1).item()
        if token in (END, DOC):
            break
        ids.append(token)
        out.append(token)
    model.train()
    return bpe.decode(out)


# ---- модел ----
ckpt = torch.load(BASE, map_location="cpu")
cfg = Config(**ckpt["config"])
model = Amica(cfg).to(DEVICE)
model.load_state_dict(ckpt["state_dict"])
print(f"base: {BASE.name}, parameters: {model.count_parameters():,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
LOG.write_text("step,sft,replay,val,seconds\n")

# ---- цикъл ----
t0 = time.time()
for step in range(MAX_STEPS):
    loss_sft = masked_loss(model, *make_batch(sample_batch(), cfg.block_size))
    loss = loss_sft
    loss_replay = torch.tensor(0.0)
    if replay:
        _, loss_replay = model(*replay_batch())
        loss = loss_sft + REPLAY_WEIGHT * loss_replay
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % EVAL_EVERY == 0 or step == MAX_STEPS - 1:
        v = val_loss(model, cfg.block_size)
        print(f"step {step:4d}  sft {loss_sft.item():.3f}  replay {loss_replay.item():.3f}  "
              f"val {v:.3f}  {time.time() - t0:.0f}s")
        with LOG.open("a") as f:
            f.write(f"{step},{loss_sft.item():.4f},{loss_replay.item():.4f},{v:.4f},{time.time() - t0:.0f}\n")

# ---- запис и проба ----
torch.save({"config": cfg.__dict__, "tokenizer": ckpt["tokenizer"], "state_dict": model.state_dict()}, CHECKPOINT)
print(f"saved: {CHECKPOINT.name}")
tests = [
    [{"role": "user", "text": "Коя си ти?"}],
    [{"role": "system", "text": "Инструменти: clock, date."}, {"role": "user", "text": "Колко е часът?"}],
    [{"role": "user", "text": "Колко е часът?"}],
    [{"role": "user", "text": "Ти коя си?"}],
]
print("---")
for turns in tests:
    print(f"user : {turns[-1]['text']}")
    print(f"amica: {reply(model, turns)!r}")

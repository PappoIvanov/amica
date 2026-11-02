"""
Амика v4 · обучение на трансформъра върху корпус v4 (2026-11-02: 1.94B токена в шест групи, речник 16 384).

Същият цикъл като train_v3.py с шест промени, всяка отбелязана с „v4:“ в кода:
  1. шест групи данни (bg_wiki, sten, law, lit, en, pairs) с тегла в партидата и val за всяка;
  2. речникът идва от json-а на токенизатора (16 384 заедно със служебните токени), не е число в кода,
     а името на токенизатора се записва в checkpoint-а, откъдето го четат engine.py и train_sft.py;
  3. моделът е около 100M параметъра: 12 слоя, 12 глави, ширина 768, контекст 1024;
  4. пълно продължаване: междинният запис (-resume.pt) пази и оптимизатора, скалера, генераторите на
     случайни числа, кривата на loss и изминалото време, така че четири сесии в Kaggle са едно обучение
     (v3 продължаваше само от теглата, с нов оптимизатор);
  5. ограничение по време (TIME_LIMIT): записва и спира сам, преди Kaggle да убие сесията на 12-ия час;
  6. записите се качват в Hugging Face (HF_REPO), откъдето следващата сесия ги сваля сама (RESUME=auto);
  плюс ACCUM: партидата минава на части (натрупване на градиентите), ако 16 × 1024 не се побере в паметта на T4.
Данните са дванадесетте .bin файла от tokenizer/encode_corpus.py (uint16, 3.9 GB), заредени изцяло в паметта.
Пускане от папката на проекта: py train/train_v4.py
Настройки от средата (в Kaggle: TOKENS=/tmp/tokens/v4, OUT=/kaggle/working, HF_REPO=p7ivanov/amica-bg,
RESUME=auto, TIME_LIMIT=39600, а токенът за Hugging Face е в променливата HF_TOKEN от Kaggle Secrets):
  TOKENS, TOKENIZER, OUT, RUN, MAX_STEPS, EVAL_EVERY, EVAL_BATCHES, SAVE_EVERY, TIME_LIMIT (секунди, 0 = без),
  RESUME (път до -resume.pt, „auto“ или празно), HF_REPO, BLOCK_SIZE, BATCH_SIZE, ACCUM, N_LAYER, N_HEAD, N_EMBD,
  DROPOUT, LR_MAX, WARMUP, W_BG_WIKI, W_STEN, W_LAW, W_LIT, W_EN, W_PAIRS.
Проба на процесора (около 2 минути, само дали върви; данните са 3.9 GB и се четат около минута):
  $env:MAX_STEPS='3'; $env:EVAL_EVERY='2'; $env:EVAL_BATCHES='1'; $env:N_LAYER='2'; $env:N_EMBD='64'; $env:N_HEAD='2'; $env:BLOCK_SIZE='128'; $env:BATCH_SIZE='4'; $env:RUN='proba'; py train/train_v4.py
Проба в Kaggle: MAX_STEPS=200 без HF_REPO, после планът за стъпките по измереното време на стъпка (плюс 25%).
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
from chat_format import load_tokenizer      # v4: токенизаторът идва със служебните токени, речникът е len(bpe.vocab)

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(42)
rng = np.random.default_rng(42)

# ---- настройки ----
TOKENS = Path(os.environ.get("TOKENS", ROOT / "data" / "clean" / "tokens" / "v4"))
TOKENIZER = ROOT / "tokenizer" / os.environ.get("TOKENIZER", "amica_bpe_16k.json")   # v4: новият речник
OUT = Path(os.environ.get("OUT", ROOT / "checkpoints"))                              # v4: в Kaggle /kaggle/working
RUN = os.environ.get("RUN", time.strftime("%Y%m%d-%H%M"))
CHECKPOINT = OUT / f"amica-v4-{RUN}.pt"               # теглата, за приложението (около 400 MB)
RESUME_FILE = OUT / f"amica-v4-{RUN}-resume.pt"       # v4: теглата + оптимизаторът + всичко за продължаване (около 1.2 GB)
LOG = CHECKPOINT.with_suffix(".csv")                  # кривата на loss, до checkpoint-а
RESUME = os.environ.get("RESUME", "")                 # v4: път до -resume.pt, „auto“ или празно (отначало)
HF_REPO = os.environ.get("HF_REPO", "")               # v4: хранилище в Hugging Face за записите; празно = без качване
TIME_LIMIT = float(os.environ.get("TIME_LIMIT", 0))   # v4: секунди за тази сесия, 0 = без ограничение
GROUPS = ["bg_wiki", "sten", "law", "lit", "en", "pairs"]        # v4: шестте групи на tokenizer/groups.py
WEIGHTS = {"bg_wiki": float(os.environ.get("W_BG_WIKI", 0.27)),   # дял от партидата за всяка група:
           "sten": float(os.environ.get("W_STEN", 0.22)),         # българският общо 0.55 (в токените е 18%),
           "law": float(os.environ.get("W_LAW", 0.03)),           # английският 0.39 (79%), двойките 0.06 (2%);
           "lit": float(os.environ.get("W_LIT", 0.03)),           # при 120 000 стъпки: 3.9 обиколки на bg_wiki,
           "en": float(os.environ.get("W_EN", 0.39)),             # 2.3 на sten, 3.1 на law, 5.6 на lit, 0.5 на en,
           "pairs": float(os.environ.get("W_PAIRS", 0.06))}       # 2.8 на pairs (виж EXPERIMENTS.md)
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", 1024))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))      # 16 × 1024 = 16 384 токена на стъпка, както при v3
ACCUM = int(os.environ.get("ACCUM", 1))                 # v4: на колко части минава партидата (1 = наведнъж)
N_LAYER = int(os.environ.get("N_LAYER", 12))            # v4: 12 слоя × 768 са около 98M параметъра
N_HEAD = int(os.environ.get("N_HEAD", 12))
N_EMBD = int(os.environ.get("N_EMBD", 768))
DROPOUT = float(os.environ.get("DROPOUT", 0.1))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 120000))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 1000))
EVAL_BATCHES = int(os.environ.get("EVAL_BATCHES", 10))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 10000))   # междинен запис и качване, около 3 часа при 1 s на стъпка
LR_MAX = float(os.environ.get("LR_MAX", 6e-4))          # v4: по-голям модел, по-ниска безопасна скорост (v3: 1e-3)
LR_MIN, WARMUP = LR_MAX / 10, int(os.environ.get("WARMUP", 1000))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"                              # смесена точност само на GPU
assert BATCH_SIZE % ACCUM == 0, "BATCH_SIZE must be divisible by ACCUM"

# ---- данни ----
bpe = load_tokenizer(TOKENIZER)
VOCAB_SIZE = len(bpe.vocab)                             # v4: 16 384 = байтове + сливания + 8 служебни токена
train_data = {g: np.fromfile(TOKENS / f"{g}_train.bin", dtype=np.uint16) for g in GROUPS}   # целите в RAM (3.9 GB)
val_data = {g: np.fromfile(TOKENS / f"{g}_val.bin", dtype=np.uint16) for g in GROUPS}
probs = np.array([WEIGHTS[g] for g in GROUPS])
probs = probs / probs.sum()
tokens_per_step = BATCH_SIZE * BLOCK_SIZE
for g, p in zip(GROUPS, probs):
    passes = MAX_STEPS * tokens_per_step * p / len(train_data[g])   # колко пъти ще мине през групата
    print(f"{g:8} train {len(train_data[g]):>14,} tokens   val {len(val_data[g]):>10,}   share {p:.2f}   passes {passes:.1f}")
print(f"tokenizer: {TOKENIZER.name}, {VOCAB_SIZE} tokens   tokens per step: {tokens_per_step:,}   "
      f"steps: {MAX_STEPS:,}   device: {DEVICE}   mixed precision: {USE_AMP}")


def windows(source, n):
    # n случайни откъса от един масив: x е откъсът, y е същият откъс, изместен с един токен напред
    ix = rng.integers(len(source) - BLOCK_SIZE - 1, size=n)
    chunks = torch.stack([torch.from_numpy(source[i:i + BLOCK_SIZE + 1].astype(np.int64)) for i in ix])
    x, y = chunks[:, :-1].contiguous(), chunks[:, 1:].contiguous()
    return x.to(DEVICE), y.to(DEVICE)


def get_batch(n=BATCH_SIZE):
    # партида от шестте групи: колко откъса от всяка се тегли по вероятностите
    counts = rng.multinomial(n, probs)
    parts = [windows(train_data[g], k) for g, k in zip(GROUPS, counts) if k > 0]
    return torch.cat([x for x, _ in parts]), torch.cat([y for _, y in parts])


def forward_loss(model, x, y):
    # под autocast умноженията са float16, загубата се смята във float32
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


def write_log():
    # v4: кривата се пренаписва цялата от списъка curve, който пътува с междинния запис между сесиите
    LOG.write_text("step,train," + ",".join(f"val_{g}" for g in GROUPS) + ",lr,seconds\n" + "".join(curve))


def hf_upload(path):
    # v4: качва файла в хранилището за данни (папка checkpoints/); при грешка само предупреждава,
    # обучението продължава, а локалното копие остава в OUT (в Kaggle: изходът на notebook-а)
    if not HF_REPO:
        return
    try:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=f"checkpoints/{path.name}",
                            repo_id=HF_REPO, repo_type="dataset")
        print(f"uploaded: {path.name} -> {HF_REPO}/checkpoints/")
    except Exception as e:
        print(f"upload failed for {path.name}: {e}")


def hf_download(name):
    # v4: сваля междинния запис на предишната сесия; None, ако още няма такъв (първата сесия)
    from huggingface_hub import HfApi, hf_hub_download
    if not HfApi().file_exists(HF_REPO, f"checkpoints/{name}", repo_type="dataset"):
        return None
    return Path(hf_hub_download(HF_REPO, f"checkpoints/{name}", repo_type="dataset"))


def find_resume():
    # v4: откъде да продължи: даден път; „auto“ = локалният запис, иначе Hugging Face, иначе отначало
    if not RESUME:
        return None
    if RESUME != "auto":
        return Path(RESUME)
    if RESUME_FILE.exists():
        return RESUME_FILE
    return hf_download(RESUME_FILE.name) if HF_REPO else None


def save(step):
    # v4: два файла: теглата за приложението и пълният запис за продължаване; step е последната завършена стъпка
    elapsed = elapsed_before + time.time() - t0
    state = {"config": cfg.__dict__, "tokenizer": TOKENIZER.name, "state_dict": model.state_dict(), "step": step}
    torch.save(state, CHECKPOINT)
    torch.save({**state, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if DEVICE == "cuda" else None,
                "curve": curve, "elapsed": elapsed, "max_steps": MAX_STEPS}, RESUME_FILE)
    write_log()
    print(f"saved: {CHECKPOINT.name} and {RESUME_FILE.name} (step {step}, {elapsed / 3600:.2f} h in total)")
    for path in (CHECKPOINT, RESUME_FILE, LOG):
        hf_upload(path)


# ---- модел ----
cfg = Config(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE, n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
             dropout=DROPOUT, tie_weights=True, fast_attention=True)
model = Amica(cfg).to(DEVICE)
print(f"parameters: {model.count_parameters():,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=0.1)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)  # мащабира загубата, за да не се занулят малките градиенти във float16
OUT.mkdir(parents=True, exist_ok=True)
start_step, elapsed_before, curve = 0, 0.0, []
resume_path = find_resume()
if resume_path:
    # v4: всичко се връща както е било в момента на записа; weights_only=False, защото файлът е наш и носи
    # състоянието на генераторите (речници и тензори), не само тегла
    ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    start_step = ckpt["step"] + 1
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt["scaler"]:
            scaler.load_state_dict(ckpt["scaler"])
        rng.bit_generator.state = ckpt["rng"]
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        if DEVICE == "cuda" and ckpt["cuda_rng"] is not None:
            torch.cuda.set_rng_state(ckpt["cuda_rng"].cpu())
        elapsed_before, curve = ckpt["elapsed"], ckpt["curve"]
        if ckpt["max_steps"] != MAX_STEPS:
            print(f"warning: MAX_STEPS was {ckpt['max_steps']:,}, now {MAX_STEPS:,}: the lr schedule changes")
        print(f"resumed from {resume_path} at step {start_step:,} ({elapsed_before / 3600:.2f} h done before)")
    else:
        print(f"resumed weights only from {resume_path} at step {start_step:,} (optimizer starts fresh)")
else:
    print("starting from scratch")

# ---- цикъл ----
t0 = time.time()
step = start_step
while step < MAX_STEPS:
    if TIME_LIMIT and time.time() - t0 > TIME_LIMIT:
        print(f"time limit of {TIME_LIMIT / 3600:.1f} h reached before step {step:,}: saving and stopping")   # v4
        break
    for group in optimizer.param_groups:
        group["lr"] = lr_at(step)
    if step % EVAL_EVERY == 0 or step == MAX_STEPS - 1:
        losses = estimate_loss(model)
        seconds = elapsed_before + time.time() - t0
        print(f"step {step:6d}  train {losses['train']:.3f}  "
              + "  ".join(f"{g} {losses['val_' + g]:.3f}" for g in GROUPS)
              + f"  lr {lr_at(step):.2e}  {seconds:.0f}s")
        curve.append(f"{step},{losses['train']:.4f}," + ",".join(f"{losses['val_' + g]:.4f}" for g in GROUPS)
                     + f",{lr_at(step):.2e},{seconds:.0f}\n")
        write_log()
    optimizer.zero_grad(set_to_none=True)
    for _ in range(ACCUM):                                   # v4: партидата на ACCUM части, градиентите се събират
        x, y = get_batch(BATCH_SIZE // ACCUM)
        loss = forward_loss(model, x, y) / ACCUM
        scaler.scale(loss).backward()                        # мащабираната загуба назад
    scaler.unscale_(optimizer)                               # обратно в истински мащаб преди ограничаването
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)                                   # пропуска стъпката, ако градиентите са прелели
    scaler.update()
    step += 1
    if SAVE_EVERY and step % SAVE_EVERY == 0 and step < MAX_STEPS:
        save(step - 1)                                       # междинен запис: последната завършена стъпка

# ---- запис и проба ----
save(step - 1)
session = time.time() - t0
print(f"this session: {step - start_step:,} steps in {session:.0f}s = {session / max(1, step - start_step):.2f} s per step")
if step < MAX_STEPS:
    print(f"not finished: {step:,} of {MAX_STEPS:,} steps; run again with RESUME=auto to continue")   # v4
    sys.exit(0)
model.eval()
start = torch.tensor([[bpe.special["<|doc|>"]]], dtype=torch.long, device=DEVICE)
with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
    sample = model.generate(start, 200, temperature=0.8)[0].tolist()
print("---")
print(bpe.decode(sample))

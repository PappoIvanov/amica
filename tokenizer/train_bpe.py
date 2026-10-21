"""
Амика · етап 4 · обучение на токенизатора върху извадка от корпуса.

Извадка от 30M знака: 70% български (литература плюс статии от Уикипедия) и 30% английски,
избрани на случаен принцип с фиксирано зърно, за да може пускането да се повтори.
Речникът е 8192 токена, от които последните 8 номера остават свободни за служебни токени
(край на документ, реплика на потребителя, реплика на Амика и други, част 5).
Накрая токенизаторът се проверява върху невиждан текст: знаци на токен за всеки език.

Резултат: tokenizer/amica_bpe.json (сливанията, влиза в git)
          tokenizer/amica_bpe.vocab.txt (речникът, подреден по номер, за четене)
Пускане от папката на проекта: py tokenizer/train_bpe.py      (около 10 минути)
"""
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # за да намери bpe.py
from bpe import BPE

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path(os.environ.get("CORPUS", ROOT / "data" / "clean" / "corpus_public.jsonl"))
OUT = ROOT / "tokenizer" / "amica_bpe.json"
VOCAB_TXT = ROOT / "tokenizer" / "amica_bpe.vocab.txt"

VOCAB_SIZE = 8192
N_SPECIAL = 8                          # запазени номера в края на речника
SEED = 42
BUDGET = {"lit": 7_500_000, "bg_wiki": 13_500_000, "en": 9_000_000}   # знаци за обучение по група
EVAL_CHARS = 1_000_000                 # знаци на група за проверка върху невиждан текст
BG_WIKI_SOURCE, EN_SOURCE = 33, 34     # номерата на двете Уикипедии в literature.txt

SAMPLES = [
    "Дойде пролет и насипа цвят, нали? Yes, in 1878!",
    "Амика е езиков модел, обучен от нулата на български.",
    "The quick brown fox jumps over the lazy dog.",
]


def group_of(doc):
    if doc["source"] == EN_SOURCE:
        return "en"
    if doc["source"] == BG_WIKI_SOURCE:
        return "bg_wiki"
    return "lit"


def read_groups():
    # всички документи, разделени по група; пазим само текста
    groups = {"lit": [], "bg_wiki": [], "en": []}
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            groups[group_of(doc)].append(doc["text"])
    return groups


def take(texts, n_chars):
    # взема цели документи от началото на списъка, докато събере n_chars знака;
    # последният се реже, за да е извадката точно толкова; връща текста и колко документа са взети
    parts, used, i = [], 0, 0
    while used < n_chars and i < len(texts):
        part = texts[i][:n_chars - used]
        parts.append(part)
        used += len(part)
        i += 1
    return "\n\n".join(parts), i


def chars_per_token(bpe, text):
    return len(text) / len(bpe.encode(text))


def token_kind(token_bytes):
    # вид на токена за статистиката на речника
    try:
        s = token_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return "partial"                      # байтове, които не са цяла буква
    body = s.lstrip(" ")
    if body == "" or body.isspace():
        return "space"
    if body.isdigit():
        return "number"
    if all(("Ѐ" <= ch <= "ӿ") for ch in body):
        return "cyrillic"
    if all(ch.isascii() and ch.isalpha() for ch in body):
        return "latin"
    if body.isalpha():
        return "other letters"
    return "punct"


if __name__ == "__main__":
    t0 = time.time()
    groups = read_groups()
    rng = random.Random(SEED)
    train_parts, eval_parts = [], {}
    for name, texts in groups.items():
        rng.shuffle(texts)
        train_text, used = take(texts, BUDGET[name])
        eval_text, _ = take(texts[used:], EVAL_CHARS)      # документи, които не са в обучението
        train_parts.append(train_text)
        eval_parts[name] = eval_text
        print(f"{name:8} {len(texts):>6,} docs, train {len(train_text):>12,} chars from {used:,} docs, "
              f"eval {len(eval_text):,} chars")
    text = "\n\n".join(train_parts)
    print(f"sample: {len(text):,} chars, read in {time.time() - t0:.0f}s\n")

    bpe = BPE()
    bpe.train(text, vocab_size=VOCAB_SIZE - N_SPECIAL)
    print(f"trained in {time.time() - t0:.0f}s\n")

    bpe.save(OUT)
    with open(VOCAB_TXT, "w", encoding="utf-8") as f:
        for i, token in enumerate(bpe.vocab):
            f.write(f"{i}\t{token.decode('utf-8', errors='replace')!r}\n")
    print(f"saved {OUT.name} and {VOCAB_TXT.name}")

    kinds = {}
    for token in bpe.vocab[256:]:
        kinds[token_kind(token)] = kinds.get(token_kind(token), 0) + 1
    print("learned tokens by kind:")
    for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14} {n:5}")
    whole_words = sum(1 for t in bpe.vocab[256:] if t.startswith(b" ") and len(t) > 1)
    print(f"  tokens starting with a space (whole words): {whole_words}\n")

    print("chars per token on unseen text:")
    for name, eval_text in eval_parts.items():
        print(f"  {name:8} {chars_per_token(bpe, eval_text):.2f}")
    print()
    for s in SAMPLES:
        ids = bpe.encode(s)
        print(f"{len(s):3} chars -> {len(ids):3} tokens: {'|'.join(bpe.decode([i]) for i in ids)}")

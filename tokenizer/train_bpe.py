"""
Амика · обучение на токенизатора върху извадка от корпуса (v2 на 2026-10-21, v4 от 2026-11-02).

Извадката е по шестте групи от groups.py с бюджет в знаци за всяка (за v4 общо 60M: 60% български,
35% английски, 5% двойки bg-en), избрана на случаен принцип с фиксирано зърно, за да може пускането да се повтори.
Корпусът v4 е 8 GB и не се събира в паметта, затова минаваме два пъти през него:
  проход 1 брои знаците по група, като чете само полетата source и chars от началото на всеки ред;
  проход 2 взима всеки документ с вероятност p = OVERSAMPLE × бюджет / знаци на групата, така че всеки
  източник влиза пропорционално на знаците си (книгите от Gutenberg според дължината си, не според броя си),
  а в паметта остават само около два пъти бюджета; после извадката се разбърква и се реже до бюджета.
Речникът е VOCAB_SIZE токена, от които последните N_SPECIAL номера остават свободни за служебните токени
(край на документ, реплики, инструменти); chat_format.py ги добавя и ги чете от json-а.
Накрая токенизаторът се проверява върху невиждан текст: знаци на токен за всяка група.

Резултат: tokenizer/<OUT>.json (сливанията, влиза в git) и tokenizer/<OUT>.vocab.txt (речникът, подреден по номер)
          amica_bpe.json (8192, v2 и v3) остава непипнат; v4 пише amica_bpe_16k.json.
Пускане от папката на проекта: py tokenizer/train_bpe.py
          (v4: около 2 минути на проход и 20 до 40 минути обучение; v2 беше 30M знака, речник 8192, 277 s)
          бърза проба: $env:SCALE = '0.001'; $env:VOCAB_SIZE = '512'; $env:OUT = 'proba'; py tokenizer/train_bpe.py
"""
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # за да намери bpe.py и groups.py
from bpe import BPE
from groups import GROUP_NAMES, source_groups

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path(os.environ.get("CORPUS", ROOT / "data" / "clean" / "corpus_public.jsonl"))
OUT_NAME = os.environ.get("OUT", "amica_bpe_16k")
OUT = ROOT / "tokenizer" / f"{OUT_NAME}.json"
VOCAB_TXT = ROOT / "tokenizer" / f"{OUT_NAME}.vocab.txt"

VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", "16384"))
N_SPECIAL = 8                          # запазени номера в края на речника
SEED = 42
SCALE = float(os.environ.get("SCALE", "1"))   # умножава бюджетите; под 1 е за бърза проба
BUDGET = {                             # знаци за обучение по група; общо 60M
    "bg_wiki": 14_000_000,
    "sten": 10_000_000,
    "law": 6_000_000,
    "lit": 6_000_000,
    "en": 21_000_000,
    "pairs": 3_000_000,
}
BUDGET = {name: int(chars * SCALE) for name, chars in BUDGET.items()}
OVERSAMPLE = 2.0                       # проход 2 събира толкова пъти бюджета, после се реже до него
EVAL_CHARS = int(1_000_000 * SCALE)    # знаци на група за проверка върху невиждан текст
# полетата преди "text" са кратки, затова source и chars се четат с регулярен израз от началото на реда:
# така един проход през 8 GB е около 2 минути вместо 4 с json.loads на всеки ред
HEAD = re.compile(r'"source": (\d+), "license": "[^"]*", "chars": (\d+), "text":')

SAMPLES = [
    "Дойде пролет и насипа цвят, нали? Yes, in 1878!",
    "Амика е езиков модел, обучен от нулата на български.",
    "The quick brown fox jumps over the lazy dog.",
    "Добър ден. ||| Good afternoon.",
]


def head_of(line):
    """(източник, знаци) на един ред от jsonl, без да се разглобява целият документ."""
    m = HEAD.search(line, 0, 3000)
    if m:
        return int(m.group(1)), int(m.group(2))
    doc = json.loads(line)                                   # рядък случай: много дълго заглавие
    return doc["source"], doc["chars"]


def count_groups(groups):
    """Проход 1: документи и знаци за всяка група в целия корпус."""
    docs = {name: 0 for name in GROUP_NAMES}
    chars = {name: 0 for name in GROUP_NAMES}
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            source, n = head_of(line)
            name = groups.get(source, "lit")
            docs[name] += 1
            chars[name] += n
    return docs, chars


def sample_documents(groups, chars):
    """Проход 2: всеки документ влиза с вероятност p = OVERSAMPLE × бюджет / знаци на групата."""
    rng = random.Random(SEED)
    p = {name: min(1.0, OVERSAMPLE * BUDGET[name] / chars[name]) if chars[name] else 0.0 for name in GROUP_NAMES}
    texts = {name: [] for name in GROUP_NAMES}
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            source, _ = head_of(line)
            name = groups.get(source, "lit")
            if rng.random() < p[name]:                       # жребият се тегли за всеки ред, за да е повторимо
                texts[name].append(json.loads(line)["text"])
    return texts, p


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


def main():
    assert set(BUDGET) == set(GROUP_NAMES), "BUDGET must name exactly the groups from groups.py"
    t0 = time.time()
    groups = source_groups()
    n_docs, n_chars = count_groups(groups)
    print(f"pass 1: {sum(n_docs.values()):,} docs, {sum(n_chars.values()):,} chars, {time.time() - t0:.0f}s")
    texts, p = sample_documents(groups, n_chars)
    print(f"pass 2: {sum(len(t) for t in texts.values()):,} docs sampled, {time.time() - t0:.0f}s\n")

    rng = random.Random(SEED)
    train_parts, eval_parts = [], {}
    print(f"{'group':8} {'docs':>9} {'chars':>15} {'p':>7} {'sampled':>8} {'train':>12} {'eval':>10}")
    for name in GROUP_NAMES:
        rng.shuffle(texts[name])
        train_text, used = take(texts[name], BUDGET[name])
        eval_text, _ = take(texts[name][used:], EVAL_CHARS)      # документи, които не са в обучението
        if len(train_text) < BUDGET[name] or len(eval_text) < EVAL_CHARS:
            print(f"Warning: group {name} is short of text: train {len(train_text):,}, eval {len(eval_text):,} chars")
        train_parts.append(train_text)
        eval_parts[name] = eval_text
        print(f"{name:8} {n_docs[name]:>9,} {n_chars[name]:>15,} {p[name]:>7.4f} {len(texts[name]):>8,} "
              f"{len(train_text):>12,} {len(eval_text):>10,}")
    text = "\n\n".join(train_parts)
    del texts                                                  # извадката над бюджета вече не трябва
    print(f"sample: {len(text):,} chars, ready in {time.time() - t0:.0f}s\n")

    bpe = BPE()
    bpe.train(text, vocab_size=VOCAB_SIZE - N_SPECIAL)
    print(f"trained in {time.time() - t0:.0f}s\n")

    bpe.save(OUT)
    with open(VOCAB_TXT, "w", encoding="utf-8") as f:
        for i, token in enumerate(bpe.vocab):
            f.write(f"{i}\t{token.decode('utf-8', errors='replace')!r}\n")
    print(f"saved {OUT.name} and {VOCAB_TXT.name}: {len(bpe.vocab)} tokens, {N_SPECIAL} numbers left for special tokens")

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
        if eval_text:
            print(f"  {name:8} {chars_per_token(bpe, eval_text):.2f}")
    print()
    for s in SAMPLES:
        ids = bpe.encode(s)
        print(f"{len(s):3} chars -> {len(ids):3} tokens: {'|'.join(bpe.decode([i]) for i in ids)}")


if __name__ == "__main__":
    main()

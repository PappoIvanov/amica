"""
Амика · етап 4 · кодиране на целия корпус в номера на токени за Kaggle.

Чете data/clean/corpus_public.jsonl, кодира всеки документ с tokenizer/amica_bpe.json и записва
номерата като uint16 (2 байта на токен, речникът е под 65 536) в data/clean/tokens/.
Документите са в четири групи, всяка в свой файл, за да може обучението да ги смесва с избрани тегла и да
мери загубата за всяка поотделно: lit (българска литература от Читанка), bg_wiki (Българска Уикипедия),
en (Simple English и English Wikipedia, книгите от Project Gutenberg) и law (законите и Конституцията).
Групата се познава по адреса на източника в data/raw/literature.txt, не по твърди номера (от 2026-10-26).
Част от документите отиват за проверка (val): цели документи, 1% от Уикипедия и 3% от литературата,
която е малка и иначе би дала твърде малко текст за стабилна загуба.
След всеки документ стои служебният токен <|doc|>, за да знае моделът къде свършва един текст.

Резултат: data/clean/tokens/{lit,bg_wiki,en,law}_{train,val}.bin   (не влизат в git, качват се в Kaggle)
          четене в Kaggle: np.fromfile(path, dtype=np.uint16)
Пускане от папката на проекта: py tokenizer/encode_corpus.py      (v2: 4 минути за 310M знака; v3 около 3 пъти повече)
"""
import json
import os
import random
import re
import sys
import time
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # за да намери bpe.py
from bpe import BPE

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path(os.environ.get("CORPUS", ROOT / "data" / "clean" / "corpus_public.jsonl"))
TOKENIZER = ROOT / "tokenizer" / "amica_bpe.json"
OUT_DIR = ROOT / "data" / "clean" / "tokens"
SEED = 42
VAL_SHARE = {"lit": 0.03, "bg_wiki": 0.01, "en": 0.01, "law": 0.03}   # дял документи за проверка по група
LITERATURE = ROOT / "data" / "raw" / "literature.txt"


def source_groups():
    """{номер на източник: група} по адреса в literature.txt: Уикипедия bg → bg_wiki, Уикипедия en и Gutenberg → en,
    parliament.bg и Уикиизточник → law, OPUS: английската страна → en, Tatoeba → lit, останалите → law; всичко друго (Читанка) → lit."""
    groups = {}
    for line in LITERATURE.read_text(encoding="utf-8-sig").splitlines():
        m = re.match(r"^(\d+)\.\s*(.+)$", line.strip())
        if not m:
            continue
        parts = [p.strip() for p in m.group(2).split(" - ")]
        url = parts[2] if len(parts) >= 3 else ""
        if "20231101.bg" in url:
            group = "bg_wiki"
        elif "wikimedia/wikipedia" in url or "gutenberg.org" in url or url.endswith("#en"):
            group = "en"                          # английските страни на OPUS са с адрес, завършващ на #en
        elif "OPUS-Tatoeba" in url:
            group = "lit"                         # битови изречения на български, не официален регистър
        elif "OPUS-" in url:
            group = "law"                         # Europarl, EU Bookshop, DGT на български: официален регистър
        elif "parliament.bg" in url or "wikisource.org" in url:
            group = "law"
        else:
            group = "lit"
        groups[int(m.group(1))] = group
    return groups


GROUPS = source_groups()


def group_of(doc):
    return GROUPS.get(doc["source"], "lit")


if __name__ == "__main__":
    t0 = time.time()
    bpe = BPE.load(TOKENIZER)
    if "<|doc|>" not in bpe.special:
        bpe.add_special("<|doc|>")
        bpe.save(TOKENIZER)                # токенизаторът вече носи и служебния токен
    doc_token = bpe.special["<|doc|>"]
    print(f"tokenizer: {len(bpe.vocab)} tokens, <|doc|> = {doc_token}")

    rng = random.Random(SEED)
    streams = {}                           # (група, train/val) → array от номера, 2 байта на токен
    stats = {}                             # (група, train/val) → [документи, знаци, токени]
    with open(CORPUS, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            doc = json.loads(line)
            group = group_of(doc)
            split = "val" if rng.random() < VAL_SHARE[group] else "train"
            key = (group, split)
            ids = bpe.encode(doc["text"])
            ids.append(doc_token)
            streams.setdefault(key, array("H")).extend(ids)
            s = stats.setdefault(key, [0, 0, 0])
            s[0] += 1
            s[1] += len(doc["text"])
            s[2] += len(ids)
            if n % 10_000 == 0:
                print(f"  {n:,} docs, {time.time() - t0:.0f}s")

    OUT_DIR.mkdir(exist_ok=True)
    total_tokens = 0
    for (group, split), ids in sorted(streams.items()):
        path = OUT_DIR / f"{group}_{split}.bin"
        with open(path, "wb") as f:
            ids.tofile(f)
        docs, chars, tokens = stats[(group, split)]
        total_tokens += tokens
        print(f"{path.name:18} {docs:>7,} docs {chars:>13,} chars {tokens:>12,} tokens  "
              f"{chars / tokens:.2f} chars/token  {path.stat().st_size / 1e6:.0f} MB")
    print(f"total: {total_tokens:,} tokens, done in {time.time() - t0:.0f}s")

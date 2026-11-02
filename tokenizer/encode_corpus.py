"""
Амика · кодиране на целия корпус в номера на токени за Kaggle (v2 на 2026-10-21, v4 от 2026-11-02).

Чете data/clean/corpus_public.jsonl, кодира всеки документ с токенизатора и записва номерата като uint16
(2 байта на токен, речникът е под 65 536) в data/clean/tokens/<VERSION>/.
Документите са в шестте групи на groups.py, всяка в свой файл, за да може обучението да ги смесва с избрани
тежести и да мери загубата за всяка поотделно: bg_wiki, sten, law, lit, en, pairs.
Част от документите отиват за проверка (val): цели документи, дял по група (VAL_SHARE); двойките bg-en във
val са и отделените двойки за мярката на превода (chrF) след обучението.
След всеки документ стои служебният токен <|doc|>, за да знае моделът къде свършва един текст.
Корпусът v4 е 7B знака, затова нищо не се държи в паметта: токените на всеки документ се записват веднага
в отворения файл на групата, а кешът на bpe.encode (парче → токени) се изпразва на всеки CACHE_RESET документа,
защото расте с всяка нова дума и при 7B знака би стигнал няколко GB.

Резултат: data/clean/tokens/<VERSION>/{bg_wiki,sten,law,lit,en,pairs}_{train,val}.bin и _stats.txt с числата
          (не влизат в git; качване в Hugging Face:
           py -m huggingface_hub.cli.hf upload p7ivanov/amica-bg data/clean/tokens/v4 v4 --repo-type dataset)
          четене в Kaggle: np.fromfile(path, dtype=np.uint16)
Пускане от папката на проекта: py tokenizer/encode_corpus.py      (v3: 464 s за 900M знака; v4: около час за 7B)
          настройки от средата: TOKENIZER (amica_bpe_16k.json), VERSION (v4), CORPUS, LIMIT
          бърза проба: $env:LIMIT = '20000'; $env:VERSION = 'proba'; py tokenizer/encode_corpus.py
"""
import json
import os
import random
import sys
import time
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # за да намери chat_format.py, bpe.py и groups.py
from chat_format import load_tokenizer
from groups import GROUP_NAMES, source_groups

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path(os.environ.get("CORPUS", ROOT / "data" / "clean" / "corpus_public.jsonl"))
TOKENIZER = ROOT / "tokenizer" / os.environ.get("TOKENIZER", "amica_bpe_16k.json")
VERSION = os.environ.get("VERSION", "v4")
OUT_DIR = ROOT / "data" / "clean" / "tokens" / VERSION
LIMIT = int(os.environ.get("LIMIT", "0"))         # над 0: само първите LIMIT документа, за проба
SEED = 42
VAL_SHARE = {"bg_wiki": 0.01, "sten": 0.01, "law": 0.03, "lit": 0.03, "en": 0.005, "pairs": 0.03}
CACHE_RESET = 50_000                               # документи между две изпразвания на кеша на encode
PROGRESS = 20_000                                  # документи между два реда на екрана


def main():
    t0 = time.time()
    bpe = load_tokenizer(TOKENIZER)                # добавя осемте служебни токена, ако липсват, и ги записва
    doc_token = bpe.special["<|doc|>"]
    assert len(bpe.vocab) <= 65536, "uint16 holds at most 65 536 token ids"
    print(f"tokenizer: {TOKENIZER.name}, {len(bpe.vocab)} tokens, <|doc|> = {doc_token}")
    groups = source_groups()
    rng = random.Random(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files, stats = {}, {}
    for name in GROUP_NAMES:
        for split in ("train", "val"):
            files[name, split] = open(OUT_DIR / f"{name}_{split}.bin", "wb")
            stats[name, split] = [0, 0, 0]         # документи, знаци, токени
    n = 0
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            if LIMIT and n >= LIMIT:
                break
            n += 1
            doc = json.loads(line)
            name = groups.get(doc["source"], "lit")
            split = "val" if rng.random() < VAL_SHARE[name] else "train"
            ids = bpe.encode(doc["text"])
            ids.append(doc_token)
            array("H", ids).tofile(files[name, split])
            s = stats[name, split]
            s[0] += 1
            s[1] += len(doc["text"])
            s[2] += len(ids)
            if n % CACHE_RESET == 0:
                bpe.cache.clear()
            if n % PROGRESS == 0:
                tokens = sum(s[2] for s in stats.values())
                print(f"  {n:,} docs, {tokens:,} tokens, {time.time() - t0:.0f}s", flush=True)
    for fh in files.values():
        fh.close()

    lines = [f"tokenizer: {TOKENIZER.name}, {len(bpe.vocab)} tokens, <|doc|> = {doc_token}; corpus: {CORPUS.name}; "
             f"date: {time.strftime('%Y-%m-%d')}",
             f"{'file':18} {'docs':>9} {'chars':>15} {'tokens':>14} {'chars/token':>11} {'MB':>7}"]
    total_tokens = 0
    for name in GROUP_NAMES:
        for split in ("train", "val"):
            docs, chars, tokens = stats[name, split]
            total_tokens += tokens
            path = OUT_DIR / f"{name}_{split}.bin"
            ratio = chars / tokens if tokens else 0.0
            lines.append(f"{path.name:18} {docs:>9,} {chars:>15,} {tokens:>14,} {ratio:>11.2f} {path.stat().st_size / 1e6:>7.0f}")
    lines.append(f"total: {n:,} docs, {total_tokens:,} tokens, done in {time.time() - t0:.0f}s")
    print("\n" + "\n".join(lines))
    (OUT_DIR / "_stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

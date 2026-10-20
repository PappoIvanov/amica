"""
Амика · етап 4 · внасяне на Българската Уикипедия в конвейера.
Чете data/raw/_wikipedia/*.parquet (извадката на Уикимедия от 2023-11-01, лиценз CC BY-SA 4.0,
свалена на ръка от huggingface.co/datasets/wikimedia/wikipedia), чисти всяка статия до свързан
текст и записва data/raw/wikipedia_NN.txt в нашия формат: "# Заглавие (N.)" + абзаци, на части
по 20 000 статии, за да не преяжда normalize.py. Добавя един ред в literature.txt за целия източник.
Пускане от папката на проекта: py data/import_wikipedia.py
Настройки от средата: WIKI (bg = Българска Уикипедия, en = Simple English Wikipedia), MAX_CHARS (таван),
MIN_CHARS (1000), SEED (42).
"""
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
# два езика с еднакъв конвейер: папка с parquet, подпапка в набора на Уикимедия, име за literature.txt,
# представка на изходните файлове, таван, азбука на езика, раздели в края на статията и заглавия за прескачане
WIKI = os.environ.get("WIKI", "bg")
SETTINGS = {
    "bg": dict(inbox="_wikipedia", subset="20231101.bg", title="Българска Уикипедия", prefix="wikipedia_",
               max_chars=200e6, script=r"[\u0400-\u04FF]",
               tail={"източници", "бележки", "литература", "външни препратки", "вижте също", "библиография",
                     "използвана литература", "източници и бележки", "препратки", "галерия", "допълнителна литература"},
               skip_title=r"^(Списък на |Списък с |Категория:|Шаблон:|Портал:)|\(пояснение\)$"),
    "en": dict(inbox="_wikipedia_en", subset="20231101.simple", title="Simple English Wikipedia", prefix="wikipedia_en_",
               max_chars=100e6, script=r"[A-Za-z]",
               tail={"references", "notes", "footnotes", "sources", "other websites", "related pages", "see also",
                     "external links", "further reading", "bibliography", "gallery"},
               skip_title=r"^(List of |Lists of |Category:|Template:|Portal:)|\(disambiguation\)$"),
}[WIKI]
INBOX = RAW / SETTINGS["inbox"]
LITERATURE = RAW / "literature.txt"
URL = f"https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/{SETTINGS['subset']}"
TITLE = SETTINGS["title"]
PREFIX = SETTINGS["prefix"]
STATUS = "свободно, CC BY-SA 4.0, извадка на Уикимедия от 2023-11-01"
MAX_CHARS = int(float(os.environ.get("MAX_CHARS", SETTINGS["max_chars"])))   # таван за целия източник
MIN_CHARS = int(os.environ.get("MIN_CHARS", 1000))           # по-късите статии са мънички и падат
SEED = int(os.environ.get("SEED", 42))                       # за повторим случаен подбор
PER_FILE = 20000                                             # статии в един суров файл
TAIL = SETTINGS["tail"]                                      # от първия такъв ред нататък всичко пада
SKIP_TITLE = re.compile(SETTINGS["skip_title"])
SCRIPT = re.compile(SETTINGS["script"])                      # буквите на езика: кирилица или латиница
LETTER = re.compile(r"[^\W\d_]")       # буква от коя да е азбука
END = ".!?…“”\")"                      # знаци, с които завършва нормален абзац


def clean_article(text):
    """Връща абзаците на статията без заглавия на раздели, списъци, таблици и опашката с източници."""
    paragraphs = []
    for line in text.split("\n"):
        line = " ".join(line.split()).lstrip("#").strip()
        if not line:
            continue
        if line.lower().rstrip(":.") in TAIL:
            break                                    # оттук нататък са препратки и източници
        if len(line) < 50 and line[-1] not in END:
            continue                                 # заглавие на раздел или елемент от списък
        letters = LETTER.findall(line)
        if not letters or len(SCRIPT.findall(line)) < 0.6 * len(letters):
            continue                                 # таблица, формула, списък на латиница
        paragraphs.append(line)
    return paragraphs


def source_number():
    # номерът на Уикипедия в literature.txt; ако го няма, добавя ред и връща новия номер
    text = LITERATURE.read_text(encoding="utf-8-sig")
    max_number = 0
    for line in text.splitlines():
        m = re.match(r"^(\d+)\.\s*(.+?)\s+-\s+(.+?)\s+-\s+(\S+)\s+-\s+(.+)$", line.strip())
        if m:
            max_number = max(max_number, int(m.group(1)))
            if m.group(4) == URL:
                return int(m.group(1))
    number = max_number + 1
    with open(LITERATURE, "a", encoding="utf-8") as f:
        if text and not text.endswith("\n"):
            f.write("\n")
        f.write(f"{number}. Уикипедия - {TITLE} - {URL} - {STATUS}\n")
    print(f"added to literature.txt as source {number}")
    return number


def main():
    files = sorted(INBOX.glob("*.parquet"))
    if not files:
        print(f"no .parquet files in {INBOX}")
        return
    number = source_number()
    stats = Counter()
    articles = []                                    # (заглавие, абзаци, брой знаци)
    for path in files:
        table = pq.read_table(path, columns=["title", "text"])
        for title, text in zip(table.column("title").to_pylist(), table.column("text").to_pylist()):
            stats["read"] += 1
            title = " ".join(title.split())
            if SKIP_TITLE.search(title):
                stats["skipped: list or disambiguation"] += 1
                continue
            paragraphs = clean_article(text)
            chars = sum(len(p) for p in paragraphs)
            if chars < MIN_CHARS:
                stats["skipped: shorter than MIN_CHARS after cleaning"] += 1
                continue
            articles.append((title, paragraphs, chars))
        print(f"{path.name}: {stats['read']:,} articles read so far, {len(articles):,} usable")

    # случаен подбор до тавана MAX_CHARS, повторим заради SEED; после по азбучен ред
    random.Random(SEED).shuffle(articles)
    kept, total = [], 0
    for article in articles:
        if total + article[2] <= MAX_CHARS:
            kept.append(article)
            total += article[2]
    kept.sort(key=lambda a: a[0])

    for old in RAW.glob(f"{PREFIX}[0-9]*.txt"):     # старите части от предишно пускане, само за този език
        old.unlink()
    for i in range(0, len(kept), PER_FILE):
        out = []
        for title, paragraphs, _ in kept[i:i + PER_FILE]:
            out.append(f"# {title} ({number}.)")
            out.append("")
            for p in paragraphs:
                out.append(p)
                out.append("")
        target = RAW / f"{PREFIX}{i // PER_FILE:02d}.txt"
        target.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
        print(f"wrote {target.name}: {len(kept[i:i + PER_FILE]):,} articles")

    usable = sum(a[2] for a in articles)
    print(f"\nread: {stats['read']:,}   usable: {len(articles):,} articles, {usable:,} chars")
    for k, v in stats.most_common():
        if k != "read":
            print(f"  {k}: {v:,}")
    print(f"kept: {len(kept):,} articles, {total:,} chars (MAX_CHARS={MAX_CHARS:,}, MIN_CHARS={MIN_CHARS})")


if __name__ == "__main__":
    main()
"""
Амика · данни за v4 · избор на английски книги от каталога на Project Gutenberg.

Чете data/raw/_gutenberg/pg_catalog.csv (свален с download_sources.py gutenberg_catalog) и допълва
data/raw/_gutenberg/_index.json, който import_gutenberg.py вече разбира: номер, автор, заглавие, година на
смъртта, адрес. Старите записи (68-те книги от DOWNLOAD_LIST_2.md) остават непроменени.
Правила за избор, за да е сигурен статусът „свободно“ и при публикуване:
  - Type = Text и Language = en (само английски, без двуезични издания);
  - всеки човек в колоната Authors (автори, преводачи, редактори) има година на смъртта до DEATH_BY (1955,
    както при Читанка: 70 години от 1 януари на следващата година); без година или „Anonymous“/„Various“ отпада;
  - без периодични издания (LoCC AP, Subjects с „Periodicals“) и без указатели (заглавие „Index of ...“);
  - едно заглавие на автор: при няколко издания остава най-малкият номер;
  - равномерна извадка от всички подходящи до LIMIT книги (4000 по подразбиране): всяка k-та по номер, за да
    има и класиката с малките номера, и по-късно качените книги и нехудожествени текстове.
Пускане от папката на проекта: py data/select_gutenberg.py            (LIMIT и DEATH_BY от средата)
"""
import csv
import json
import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
INBOX = ROOT / "data" / "raw" / "_gutenberg"
CATALOG = INBOX / "pg_catalog.csv"
INDEX = INBOX / "_index.json"
LIMIT = int(os.environ.get("LIMIT", 4000))
DEATH_BY = int(os.environ.get("DEATH_BY", 1955))

YEARS = re.compile(r"(\d{4})\??\s*-\s*(\d{4})?\??")       # „1812-1870“, „1856-“, „1812?-1870“
ROLE = re.compile(r"\s*\[[^\]]*\]")                         # „[Translator]“, „[Editor]“
BAD_NAMES = re.compile(r"^(anonymous|various|unknown)", re.I)


def persons(field):
    """Колоната Authors: списък (име в ред „Име Фамилия“, година на смъртта или None)."""
    out = []
    for person in field.split(";"):
        person = ROLE.sub("", person).strip()
        if not person:
            continue
        m = YEARS.search(person)
        died = int(m.group(2)) if m and m.group(2) else None
        name = YEARS.sub("", person).strip(" ,")
        parts = [p.strip() for p in name.split(",")]
        name = f"{parts[1]} {parts[0]}" if len(parts) >= 2 and parts[1] else parts[0]
        out.append((name, died))
    return out


def main():
    if not CATALOG.exists():
        print(f"missing {CATALOG}: run download_sources.py gutenberg_catalog first")
        return
    books = json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else []
    have = {b["number"] for b in books}
    seen = {(re.sub(r"\W+", " ", b["title"].lower()).strip(), b["author"].lower()) for b in books}
    counts = {"rows": 0, "text_en": 0, "rights_ok": 0, "kept": 0}
    chosen = []
    with open(CATALOG, encoding="utf-8", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["Text#"]))
    for row in rows:
        counts["rows"] += 1
        if row["Type"] != "Text" or row["Language"] != "en":
            continue
        counts["text_en"] += 1
        people = persons(row["Authors"])
        if not people or any(died is None or died > DEATH_BY for _, died in people):
            continue
        if any(BAD_NAMES.match(name) for name, _ in people):
            continue
        if row["LoCC"].startswith("AP") or "Periodicals" in row["Subjects"] or re.match(r"^Index of", row["Title"], re.I):
            continue
        counts["rights_ok"] += 1
        number = int(row["Text#"])
        title = row["Title"].split("\n")[0].strip()
        author = people[0][0]
        key = (re.sub(r"\W+", " ", title.lower()).strip(), author.lower())
        if number in have or key in seen:
            continue
        seen.add(key)
        chosen.append({"number": number, "author": author, "title": title,
                       "died": max(died for _, died in people), "url": f"https://www.gutenberg.org/ebooks/{number}"})
    eligible = len(chosen)
    if eligible > LIMIT:                                        # всяка k-та по номер, равномерно по целия каталог
        chosen = [chosen[round(i * eligible / LIMIT)] for i in range(LIMIT)]
    counts["kept"] = len(chosen)
    books += chosen
    INDEX.write_text(json.dumps(books, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"catalog rows {counts['rows']:,}, English texts {counts['text_en']:,}, rights ok {counts['rights_ok']:,}, "
          f"eligible new {eligible:,}, chosen {counts['kept']:,} (LIMIT={LIMIT}, DEATH_BY={DEATH_BY}); index now {len(books):,} books")
    if chosen:
        print(f"first new: {chosen[0]['number']} {chosen[0]['author']} - {chosen[0]['title'][:50]}")
        print(f"last new:  {chosen[-1]['number']} {chosen[-1]['author']} - {chosen[-1]['title'][:50]}")


if __name__ == "__main__":
    main()

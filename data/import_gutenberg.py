"""
Амика · данни за v3 · внасяне на английски книги от Project Gutenberg.

Чете data/raw/_gutenberg/pg<номер>.txt (чистият текст в UTF-8 от огледалото gutenberg.pglaf.org,
свалени по списъка в data/raw/DOWNLOAD_LIST_2.md, раздел 2) и data/raw/_gutenberg/_index.json
(номер, автор, заглавие, година на смъртта). За всяка книга записва data/raw/gutenberg_<номер>_<slug>.txt
в нашия суров формат: „# Заглавие — Автор (N.)“ + абзаци, и добавя ред в data/raw/literature.txt.
Лицензът на Project Gutenberg позволява свободно ползване на самите текстове (обществено достояние),
стига да паднат служебната глава и опашка и всяко споменаване на проекта: скриптът реже между редовете
„*** START OF ...“ и „*** END OF ...“ и маха абзаците, в които се споменава Project Gutenberg.
Редовете в книгите са нарочно къси (около 70 знака): абзацът е групата редове между два празни реда и се
събира в един ред, както при останалите ни източници.
Пускане от папката на проекта: py data/import_gutenberg.py
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INBOX = RAW / "_gutenberg"
LITERATURE = RAW / "literature.txt"

START = re.compile(r"^\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*\*\*\*\s*$", re.I)
END = re.compile(r"^\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*\*\*\*\s*$", re.I)
PROJECT = re.compile(r"project gutenberg|gutenberg\.org|gutenberg literary archive", re.I)
PRODUCED = re.compile(r"^(Produced by|E-text prepared by|Transcribed from|This etext was|Updated editions will)", re.I)
ITALIC = re.compile(r"_([^_\n]+?)_")                 # _is_ -> is


def slugify(text):
    text = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return text[:40]


def book_paragraphs(text):
    """Текстът между маркерите на Gutenberg, на абзаци по един ред."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start = next((i for i, line in enumerate(lines) if START.match(line.strip())), None)
    end = next((i for i, line in enumerate(lines) if END.match(line.strip())), None)
    if start is None or end is None or end <= start:
        return None
    body = lines[start + 1:end]
    paragraphs, current = [], []
    for line in body:
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    # служебни абзаци: кой е подготвил електронния текст, споменаване на проекта
    paragraphs = [p for p in paragraphs if not PROJECT.search(p) and not PRODUCED.match(p)]
    # надписи под илюстрации: „[Illustration: ...]“, понякога през няколко абзаца до затварящата скоба
    cleaned, inside = [], False
    for p in paragraphs:
        if inside:
            inside = not p.endswith("]")
            continue
        if p.startswith("[Illustration"):
            inside = not p.endswith("]")
            continue
        cleaned.append(ITALIC.sub(r"\1", p))          # _дума_ е курсив в Gutenberg: падат само подчертавките
    return cleaned


def read_literature():
    entries, max_number = {}, 0
    if LITERATURE.exists():
        for line in LITERATURE.read_text(encoding="utf-8-sig").splitlines():
            m = re.match(r"^(\d+)\.\s*(.+?)\s+-\s+(.+?)\s+-\s+(\S+)\s+-\s+(.+)$", line.strip())
            if m:
                max_number = max(max_number, int(m.group(1)))
                entries[m.group(4)] = int(m.group(1))
    return entries, max_number


def main():
    index_path = INBOX / "_index.json"
    if not index_path.exists():
        print(f"missing {index_path}")
        return
    books = json.loads(index_path.read_text(encoding="utf-8"))
    entries, number = read_literature()
    new_lines, total_chars, kept = [], 0, 0
    for book in books:
        path = INBOX / f"pg{book['number']}.txt"
        if not path.exists():
            print(f"  missing file: {path.name} ({book['title'][:40]})")
            continue
        paragraphs = book_paragraphs(path.read_text(encoding="utf-8-sig", errors="replace"))
        if not paragraphs:
            print(f"  skip {path.name}: no START/END markers found")
            continue
        url = book["url"]
        if url in entries:
            source = entries[url]
        else:
            number += 1
            source = entries[url] = number
            status = f"свободно, починал {book['died']}"
            new_lines.append(f"{number}. {book['author']} - {book['title']} - {url} - {status}")
        chars = sum(len(p) for p in paragraphs)
        title = re.sub(r"\s*\(.*?\)\s*$", "", book["title"])          # без бележката за преводача в заглавието
        target = RAW / f"gutenberg_{book['number']}_{slugify(title)}.txt"
        out = [f"# {title} — {book['author']} ({source}.)", ""]
        for p in paragraphs:
            out += [p, ""]
        target.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
        kept += 1
        total_chars += chars
        print(f"  {path.name:12s} -> {target.name:50s} {len(paragraphs):6d} paragraphs, {chars:9,} chars")
    print(f"\nkept {kept} of {len(books)} books, {total_chars:,} chars")
    if new_lines:
        existing = LITERATURE.read_text(encoding="utf-8-sig") if LITERATURE.exists() else ""
        with open(LITERATURE, "a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write("\n".join(new_lines) + "\n")
        print(f"added {len(new_lines)} lines to literature.txt (first: {new_lines[0][:80]})")


main()

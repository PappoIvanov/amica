"""
Амика · данни за v4 · внасяне на стенограмите от пленарните заседания на Народното събрание.

Чете data/raw/_stenograms/sten_<id>.json (отговорите на https://www.parliament.bg/api/v1/pl-sten/<id>, свалени
с download_sources.py stenograms). Текстът на заседанието е HTML от Word в полето Pl_Sten_body; тук става на
абзаци по същия начин като законите (html_to_lines от import_laws.py). Записва data/raw/stenograms_NN.txt на части
по PART заседания в нашия суров формат: „# Заглавие (N.)“ + абзаци, едно заседание = един документ, и добавя
един ред в data/raw/literature.txt за целия източник.
Филтри: заседания преди DATE_FROM (1945-01-01) отпадат, защото са със стария правопис, а също и всеки документ,
в който старите букви ѣ и ѫ са повече от OLD_LETTERS на 10 000 знака (сканирани текстове с остатъци); документи
под MIN_CHARS (2000 знака) отпадат. Стенограмите са говорим език (изказвания, реплики, обръщения), затова при
токенизирането за v4 отиват в отделна група, не при законите.
Права: стенограмите са официални документи на Народното събрание, публикувани за свободен достъп без условия;
статусът е „свободно“ с бележка, че не са нормативни актове по ЗАПСП, чл. 4, т. 1 (преценка на автора).
Пускане от папката на проекта: py data/import_stenograms.py        (DATE_FROM, MIN_CHARS, PART, OLD_LETTERS от средата)
"""
import html
import json
import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INBOX = RAW / "_stenograms"
LITERATURE = RAW / "literature.txt"
PREFIX = "stenograms_"
URL = "https://www.parliament.bg/bg/plenaryst"
DATE_FROM = os.environ.get("DATE_FROM", "1945-01-01")
MIN_CHARS = int(os.environ.get("MIN_CHARS", 2000))
PART = int(os.environ.get("PART", 200))
OLD_LETTERS = float(os.environ.get("OLD_LETTERS", 5))          # на 10 000 знака
OLD = re.compile(r"[ѣѫѭѢѪѬ]")


def html_to_lines(body):
    """HTML от Word -> списък от редове с текст; копие на същата функция в import_laws.py (той няма
    __main__ предпазител и не може да се внася като модул)."""
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)                 # коментарите на Word
    body = re.sub(r"<(style|script|xml)[^>]*>.*?</\1>", "", body, flags=re.S | re.I)
    body = re.sub(r"<\s*(br|/p|/div|/li|/tr|/h\d)[^>]*>", "\n", body, flags=re.I)
    body = re.sub(r"<\s*/td[^>]*>", " ", body, flags=re.I)             # клетките на една таблична колона
    body = re.sub(r"<[^>]+>", "", body)                                # всички останали тагове
    body = html.unescape(body).replace("\xa0", " ").replace(" ", " ")   # твърди и широки интервали
    body = body.replace("\x7f", "").replace("­", "")                     # боклук от Word: DEL и мек тире-пренос
    lines = []
    for line in body.split("\n"):
        line = re.sub(r"[ \t\r\f\v]+", " ", line).strip()
        if line:
            lines.append(line)
    return lines


def read_literature():
    entries, last = {}, 0
    if LITERATURE.exists():
        for line in LITERATURE.read_text(encoding="utf-8-sig").splitlines():
            m = re.match(r"^(\d+)\.\s+.*?\s-\s(\S+)\s-\s", line)
            if m:
                last = max(last, int(m.group(1)))
                entries[m.group(2)] = int(m.group(1))
    return entries, last


def document(record):
    """(заглавие, дата, абзаци) от един JSON запис или None, ако отпада."""
    date = (record.get("Pl_Sten_date") or "")[:10]
    if not date or date < DATE_FROM:
        return None
    sub = re.sub(r"\s+", " ", record.get("Pl_Sten_sub") or "").strip()
    title = sub.split(" София")[0].strip(" ,.") if sub else "Пленарно заседание"
    lines = [l for l in html_to_lines(record.get("Pl_Sten_body") or "") if len(l) > 1]
    text = "\n".join(lines)
    if len(text) < MIN_CHARS or len(OLD.findall(text)) * 10000 / len(text) > OLD_LETTERS:
        return None
    return f"{title.capitalize()}, {date}", date, lines


def main():
    files = sorted(INBOX.glob("sten_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        print(f"no sten_*.json in {INBOX}")
        return
    entries, last = read_literature()
    source = entries.get(URL)
    docs, dropped = [], 0
    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            dropped += 1
            continue
        doc = document(record)
        if doc is None:
            dropped += 1
            continue
        docs.append(doc)
    docs.sort(key=lambda d: d[1])                                  # по дата
    years = f"{docs[0][1][:4]}–{docs[-1][1][:4]}" if docs else ""
    status = "свободно, официални документи на Народното събрание, публикувани за свободен достъп (не са нормативни актове; преценка на автора)"
    existing = LITERATURE.read_text(encoding="utf-8-sig") if LITERATURE.exists() else ""
    if source is None:                                             # първо пускане: нов ред в края
        source = last + 1
        line = f"{source}. Народно събрание - Стенограми от пленарни заседания ({years}, {len(docs)} заседания) - {URL} - {status}"
        with open(LITERATURE, "a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write(line + "\n")
        print(f"added to literature.txt: {line[:90]}...")
    else:                                                          # следващо пускане: същият номер, обновени години и брой
        line = f"{source}. Народно събрание - Стенограми от пленарни заседания ({years}, {len(docs)} заседания) - {URL} - {status}"
        lines = [line if l.startswith(f"{source}. ") and URL in l else l for l in existing.splitlines()]
        LITERATURE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"updated literature.txt line {source}: {years}, {len(docs)} sittings")
    for old in RAW.glob(f"{PREFIX}[0-9]*.txt"):
        old.unlink()
    total = 0
    for part in range(0, len(docs), PART):
        out = []
        for title, date, lines in docs[part:part + PART]:
            out += [f"# {title} ({source}.)", ""]
            for l in lines:
                out += [l, ""]
            total += sum(len(l) for l in lines)
        (RAW / f"{PREFIX}{part // PART + 1:02d}.txt").write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    print(f"kept {len(docs):,} sittings in {(len(docs) + PART - 1) // PART} parts, {total:,} chars; dropped {dropped:,} "
          f"(before {DATE_FROM}, old orthography, short or unreadable)")


if __name__ == "__main__":
    main()

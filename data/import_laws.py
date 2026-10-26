"""
Амика · данни за v3 · внасяне на закони от сайта на Народното събрание.

Чете data/raw/_laws/act_<id>.json (отговорите на https://www.parliament.bg/api/v1/act/<id>, свалени
на ръка или със скрипт: цели закони и кодекси, приети от 41-во до 52-ро Народно събрание, без законите
„за изменение и допълнение“, ратификациите и бюджетите) и data/raw/_laws/const.json (Конституцията от
https://www.parliament.bg/api/v1/nav/bg/const). Текстът в тях е HTML от Word. Скриптът го прави на
обикновен текст, маха указа за обнародване пред всеки закон (еднакъв шаблон, 100 пъти) и записва
data/raw/laws_parliament.txt и data/raw/laws_constitution.txt в нашия суров формат: „# Заглавие (N.)“ + абзаци.
Добавя два реда в data/raw/literature.txt: един за законите и един за Конституцията.
Права: официалните текстове на нормативни актове не са обект на авторско право (ЗАПСП, чл. 4, т. 1),
затова статусът е „свободно“. Сайтът няма забрани в robots.txt.
Пускане от папката на проекта: py data/import_laws.py
"""
import html
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INBOX = RAW / "_laws"
LITERATURE = RAW / "literature.txt"
MIN_CHARS = 3000                       # по-късите актове са бланки (например за преброяване) и не носят език
# закони „за изменение и допълнение“ са откъслечни („в чл. 9 думите ... се заличават“) и не влизат; заглавията в
# сайта имат печатни грешки („Законо“, „Законт“, „именение“, двойни интервали), затова шаблонът е широк
AMENDMENT = re.compile(r"^(закон\w*|законопроект)\s+за\s+(изменени|именени|допълнени|отм[яе]н|денонсир|прекратяван)", re.I)

SOURCES = {                            # ключ в literature.txt -> (автор, заглавие, адрес, статус)
    "laws": ("Народно събрание", "Закони и кодекси, приети от 41-во до 52-ро Народно събрание (2009–2026), както са обнародвани",
             "https://www.parliament.bg/bg/laws", "свободно, нормативни актове (ЗАПСП, чл. 4, т. 1)"),
    "const": ("Народно събрание", "Конституция на Република България",
              "https://www.parliament.bg/bg/const", "свободно, нормативен акт (ЗАПСП, чл. 4, т. 1)"),
    # консолидирани текстове на закони, преписани от доброволци в Уикиизточник (отговори на
    # https://bg.wikisource.org/w/api.php?action=parse&page=<заглавие>&prop=text, запазени като wikisource_<име>.json);
    # текстът на закона е обществено достояние, самата страница е CC BY-SA
    "wikisource": ("Уикиизточник", "Закони в консолидиран вид от bg.wikisource.org (Търговски закон)",
                   "https://bg.wikisource.org/wiki/Категория:Закони", "свободно, нормативни актове (ЗАПСП, чл. 4, т. 1); страниците са CC BY-SA"),
}
WIKI_TRAILER = re.compile(r"^(Категори[яи]|Взето от|Скрит[аи] категори|Този текст представлява нормативен)")   # опашката на страницата
# началото на самия закон след указа за обнародване: ред „ЗАКОН“, „КОДЕКС“ или име на кодекс с главни букви
ACT_START = re.compile(r"^(ЗАКОН|КОДЕКС|[А-Я ,\-]+КОДЕКС[А-Я ,\-]*)$")
LAT2CYR = str.maketrans("aeopcxyABCEHKMOPTXY", "аеорсхуАВСЕНКМОРТХУ")   # латински двойници в заглавията


def html_to_lines(body):
    """HTML от Word -> списък от редове с текст. Блоковите тагове стават нови редове, останалото пада."""
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)                 # коментарите на Word
    body = re.sub(r"<(style|script|xml)[^>]*>.*?</\1>", "", body, flags=re.S | re.I)
    body = re.sub(r"<\s*(br|/p|/div|/li|/tr|/h\d)[^>]*>", "\n", body, flags=re.I)
    body = re.sub(r"<\s*/td[^>]*>", " ", body, flags=re.I)             # клетките на една таблична колона
    body = re.sub(r"<[^>]+>", "", body)                                # всички останали тагове
    body = html.unescape(body).replace("\xa0", " ").replace(" ", " ")   # твърди и широки интервали
    body = body.replace("\x7f", "").replace("­", "")                     # боклук от Word: DEL и мек тире-пренос
    lines = []
    for line in body.split("\n"):
        line = re.sub(r"[ \t\r\f\v]+", " ", line).strip()
        if line:
            lines.append(line)
    return lines


def cut_decree(lines):
    """Махa указа на президента пред закона: от началото до реда „Подпечатан с държавния печат.“
    и подписа на министъра, или до първия ред „ЗАКОН“/„КОДЕКС“. Ако няма такова, текстът остава цял."""
    for i, line in enumerate(lines[:60]):
        if line.startswith("Министър на правосъдието"):
            return lines[i + 1:]
    for i, line in enumerate(lines[:60]):
        if ACT_START.match(line):
            return lines[i:]
    return lines


def read_literature():
    """Връща {адрес: номер} за вече вписаните източници и най-големия номер."""
    entries, max_number = {}, 0
    if LITERATURE.exists():
        for line in LITERATURE.read_text(encoding="utf-8-sig").splitlines():
            m = re.match(r"^(\d+)\.\s*(.+?)\s+-\s+(.+?)\s+-\s+(\S+)\s+-\s+(.+)$", line.strip())
            if m:
                max_number = max(max_number, int(m.group(1)))
                entries[m.group(4)] = int(m.group(1))
    return entries, max_number


def source_number(key, entries, counter, new_lines):
    author, title, url, status = SOURCES[key]
    if url not in entries:
        counter[0] += 1
        entries[url] = counter[0]
        new_lines.append(f"{counter[0]}. {author} - {title} - {url} - {status}")
    return entries[url]


def main():
    acts = sorted(INBOX.glob("act_*.json"))
    const = INBOX / "const.json"
    if not acts and not const.exists():
        print(f"no act_*.json or const.json in {INBOX}")
        return
    entries, max_number = read_literature()
    counter, new_lines = [max_number], []

    # --- законите ---
    out, kept, skipped, total_chars = [], 0, 0, 0
    number = source_number("laws", entries, counter, new_lines) if acts else None
    for path in acts:
        act = json.loads(path.read_text(encoding="utf-8"))
        title = re.sub(r"\s+", " ", (act.get("L_ActL_final") or act.get("L_ActL_title") or "").translate(LAT2CYR)).strip()
        body = act.get("L_ActL_final_body") or act.get("L_ActL_body") or ""
        lines = cut_decree(html_to_lines(body))
        text = "\n\n".join(lines)
        if len(text) < MIN_CHARS or not title or AMENDMENT.match(title) or title.startswith("Закона "):
            skipped += 1
            print(f"  skip {path.name}: {len(text)} chars, {title[:60]!r}")
            continue
        issue = f"ДВ, бр. {act.get('L_Act_dv_iss')} от {act.get('L_Act_dv_year')} г." if act.get("L_Act_dv_iss") else ""
        heading = f"{title} ({issue})" if issue else title
        out.append(f"# {heading} ({number}.)\n\n{text}\n")
        kept += 1
        total_chars += len(text)
    if out:
        (RAW / "laws_parliament.txt").write_text("\n".join(out), encoding="utf-8")
        print(f"laws_parliament.txt: {kept} acts kept, {skipped} skipped, {total_chars:,} chars, source number {number}")

    # --- Конституцията ---
    if const.exists():
        data = json.loads(const.read_text(encoding="utf-8"))
        lines = html_to_lines(data.get("C_ArL_body") or "")
        text = "\n\n".join(lines)
        number = source_number("const", entries, counter, new_lines)
        (RAW / "laws_constitution.txt").write_text(f"# Конституция на Република България ({number}.)\n\n{text}\n", encoding="utf-8")
        print(f"laws_constitution.txt: {len(text):,} chars, source number {number}")

    # --- Уикиизточник: консолидирани текстове ---
    wiki = sorted(INBOX.glob("wikisource_*.json"))
    if wiki:
        number = source_number("wikisource", entries, counter, new_lines)
        out = []
        for path in wiki:
            data = json.loads(path.read_text(encoding="utf-8"))["parse"]
            lines = html_to_lines(data["text"]["*"])
            end = next((i for i, line in enumerate(lines) if WIKI_TRAILER.match(line)), len(lines))
            text = "\n\n".join(lines[:end])
            out.append(f"# {data['title']} (консолидиран текст, bg.wikisource.org) ({number}.)\n\n{text}\n")
            print(f"  {path.name}: {data['title']}, {len(text):,} chars")
        (RAW / "laws_wikisource.txt").write_text("\n".join(out), encoding="utf-8")
        print(f"laws_wikisource.txt: {len(wiki)} texts, source number {number}")

    if new_lines:
        existing = LITERATURE.read_text(encoding="utf-8-sig") if LITERATURE.exists() else ""
        with open(LITERATURE, "a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write("\n".join(new_lines) + "\n")
        print("added to literature.txt:\n  " + "\n  ".join(new_lines))


main()

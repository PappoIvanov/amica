"""
Амика · етап 2 · внасяне на книги от Читанка.
Чете всеки архив data/raw/_chitanka/*sfb.zip (свален на ръка от браузъра), разбира
SFB форматa на Читанка, реже книгата на отделни текстове по заглавията, изхвърля
служебните части (данни за изданието, анотации, предговори от други автори, бележки)
и записва data/raw/chitanka_<id>_<slug>.txt в нашия формат: "# Заглавие (N.)" + абзаци.
Добавя и ред в data/raw/literature.txt за всяка нова книга или произведение.
Три вида архиви (от 2026-10-26, за втория списък data/raw/DOWNLOAD_LIST_2.md):
  Автор_-_Заглавие_-_485-b.sfb.zip      книга от линк /book/485: всички .sfb вътре са една книга
  Автор_-_Заглавие_-_3757-sfb.zip       отделно произведение от линк /text/3757
  chitanka-ivan-vazov_Piesa-sfb.zip     група произведения от линк /text/3761,3766,...: по един .sfb на
                                        произведение, номерът е в името на файла вътре („... - 3761.sfb“)
Пиесите не искат нищо особено: имената на говорещите са обикновен текст („ГАВРИЛ. Бай Гунчо...“) и остават,
а подчертавките около сценичните бележки падат като всеки друг акцент.
Пускане от папката на проекта: py data/import_chitanka.py
"""
import re
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INBOX = RAW / "_chitanka"
LITERATURE = RAW / "literature.txt"

# година на смъртта на авторите, за статуса в literature.txt (свободно = починал до 1955)
DEATH_YEAR = {
    "Николай Райнов": 1954, "Елин Пелин": 1949, "Йордан Йовков": 1937, "Иван Вазов": 1921,
    "Георги Райчев": 1947, "Асен Разцветников": 1951, "Чичо Стоян": 1939, "Стоян Михайловски": 1927,
    "Петко Р. Славейков": 1895, "Михалаки Георгиев": 1916, "Алеко Константинов": 1897,
    "Любен Каравелов": 1879,
    # вторият списък, 2026-10-26
    "Антон Страшимиров": 1937, "Добри Немиров": 1945, "Захари Стоянов": 1889, "Петко Тодоров": 1916,
    "Христо Смирненски": 1923, "Добри Войников": 1878, "Тодор Г. Влайков": 1943,
}
# заглавия на текстове, които не са част от произведението
SERVICE_TITLE = re.compile(
    r"предговор|послеслов|бележк|съдържание|информация|издание|речник|коментар|биограф|"
    r"за автора|библиограф|послепис|от съставителя|от издателя|към читателя|^писател на ", re.I)
# блокове, чието съдържание изхвърляме: информация, анотация, епиграф, посвещение, таблица, преформатиран
# (M = стилове, например надпис в рамка; текстът вътре е част от произведението и остава)
DROP_BLOCKS = set("IAEDTF")
INLINE_TAG = re.compile(r"\{/?[a-z]+[^}]*\}")       # {e} {/e} {s} {m x} {img:...} ...
INLINE_NOTE = re.compile(r"\[\*\d*\s*[^\]]*\]")      # [* бележка под линия *] или [*1 бележка]
NOTE_ANCHOR = re.compile(r"(?<=\S)\*+\d*(?=[\s.,;:!?…)“\"]|$)")   # "дума*", "дума*1", "дума!*" = препратка към бележка
HEADING = re.compile(r"^(>+)\t?(.*)$")
MARKER = re.compile(r"^([A-Z])([>$])(?:\t.*)?$")      # I> ... I$, понякога с параметър: "M>\tframe"
# заглавие, което всъщност е подзаглавие с жанр или произход: "Руска приказка", "Китайска приказка"
GENRE = re.compile(r"(приказк|легенд|басн|разказ|повест|сказани|песен|поем|притч|анекдот)", re.I)
# "заглавие", което е име на човек: "Богомил Райнов", "Божанка Константинова"; следващото заглавие е негов текст
PERSON_NAME = re.compile(r"^[А-ЯЁ][а-яё]+(?:\s[А-ЯЁ]\.)?(?:\s[А-ЯЁ][а-яё]+){1,2}$")


def parse_sfb(text, log):
    """Връща (автор на книгата, заглавие на книгата, списък от (заглавие, автор, абзаци))."""
    book_author, book_title = None, None
    header = []                                  # редовете с "|" преди първото заглавие: автор, заглавие
    docs, title, doc_author, paragraphs = [], None, None, []
    blocks, level = [], 0                        # отворените блокове, защото се влагат: E> P> ... P$ E$
    pending_name, signature = None, None         # име от предишно заглавие и подпис "@" в края на текста

    def flush():
        nonlocal title, doc_author, paragraphs, pending_name, signature
        if paragraphs:
            text_author, final_title = doc_author, title
            if pending_name and not text_author:
                # "> Име" + "> Заглавие": предговор от друг човек, ако заглавието е служебно,
                # е името на автора на книгата или текстът е подписан със същото име;
                # иначе е заглавие с подзаглавие: "Баба Недка — Из страшните времена по нашенско"
                if SERVICE_TITLE.search(title or "") or title == book_author or signature == pending_name:
                    text_author = pending_name
                else:
                    final_title = f"{pending_name} — {title}"
            docs.append((final_title, text_author, paragraphs))
        title, doc_author, paragraphs, pending_name, signature = title, None, [], None, None

    for raw_line in text.split("\n"):
        line = raw_line.rstrip("\r")
        if not line.strip():
            continue
        m = MARKER.match(line.strip())
        if m:                                   # начало или край на блок
            if m.group(2) == ">":
                blocks.append(m.group(1))
            elif blocks:
                blocks.pop()
            continue
        dropped = [b for b in blocks if b in DROP_BLOCKS]
        if dropped:
            log[f"dropped block {dropped[0]}"] += 1
            continue
        m = HEADING.match(line)
        if m:                                   # заглавие
            new_level, heading = len(m.group(1)), NOTE_ANCHOR.sub("", INLINE_TAG.sub("", m.group(2))).strip()
            if not heading:
                continue                        # празен ред със знак за заглавие: нищо
            if title is not None and not paragraphs:
                # предишното заглавие остана без текст: раздел, подзаглавие или име на автор
                if new_level > level:
                    pass                        # раздел на книгата над отделните текстове: пада
                elif GENRE.search(heading):
                    log["subtitle with genre dropped"] += 1
                    continue                    # "Руска приказка" под истинското заглавие
                elif PERSON_NAME.match(title) and title != book_author:
                    # "> Божанка Константинова" + "> Чичо Стоян": името се пази, flush решава дали е автор
                    flush()
                    pending_name, title, level = title, heading, new_level
                    continue
                else:
                    heading = f"{title} — {heading}"   # "Басненик — Езопови басни"
            flush()
            title, level = heading, new_level
            continue
        head, _, body = line.partition("\t")
        if head == "|":                          # в началото на книгата: "| Автор", после "| Заглавие"
            if title is None:
                header.append(NOTE_ANCHOR.sub("", body.strip()))   # „Отечество*“: звездичката е препратка към бележка
                if len(header) >= 2:
                    book_author, book_title = header[0], header[1]
                else:
                    book_title = header[0]
            continue
        if head == "@@" or head == "#" or head == "!":
            continue
        if head == "@":                          # автор: на книгата (преди първото заглавие) или на отделен текст
            if blocks:
                continue                         # автор на цитат, епиграф или песен вътре в блок: не е автор на текста
            if paragraphs:
                signature = body.strip()         # подпис в края на текста, например на предговор
                continue
            if title is None:
                book_author = body.strip()
            else:
                doc_author = body.strip()
            continue
        if head in ("", "N"):                    # абзац (ред с табулатор); "N" = авторска забележка в пиеса, пак текст
            body = INLINE_NOTE.sub("", body)
            body = INLINE_TAG.sub("", body)
            body = re.sub(r"__(.+?)__", r"\1", body)               # __силен акцент__
            body = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", body)    # _акцент_
            body = NOTE_ANCHOR.sub("", body)                        # звездичка след дума = препратка към бележка
            body = body.strip()
            if not body or body.startswith("[*") or body.startswith("@@"):
                continue                         # бележка под линия или дата и място ("@@София, 1945")
            if body.startswith("[") and body.endswith("]"):
                continue                         # редакторска бележка в скоби: "[сп. „Мисъл“, 1892]"
            paragraphs.append(body)
        else:
            log[f"unknown line marker: {head[:10]}"] += 1
    flush()
    if docs and all(t is None for t, _, _ in docs):   # книга без заглавия вътре: едно произведение
        docs = [(book_title, a, p) for _, a, p in docs]
    return book_author, book_title, docs


def read_literature():
    # връща {"book/485" или "text/7866": номер в literature.txt} и най-големия използван номер
    entries, max_number = {}, 0
    if LITERATURE.exists():
        for line in LITERATURE.read_text(encoding="utf-8-sig").splitlines():
            m = re.match(r"^(\d+)\.\s*(.+?)\s+-\s+(.+?)\s+-\s+(\S+)\s+-\s+(.+)$", line.strip())
            if m:
                number = int(m.group(1))
                max_number = max(max_number, number)
                source = re.search(r"chitanka\.info/(book|text)/(\d+)", m.group(4))
                if source:
                    entries[f"{source.group(1)}/{source.group(2)}"] = number
    return entries, max_number


def slugify(name):
    # "Az ti toj" или "Az_ti_toj" -> "az_ti_toj", най-много 40 знака, за името на файла
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]


def zip_works(zip_path):
    """Разопакова един архив на произведения: списък от (ключ "book/485" или "text/7866", slug, текст)."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".sfb")]
        if not members:
            return []
        m = re.match(r"^(.+?)_-_(.+)_-_(\d+)-b\.sfb\.zip$", zip_path.name)
        if m:                                    # книга: всички .sfb файлове заедно са една книга
            text = "\n".join(zf.read(n).decode("utf-8-sig", errors="replace") for n in members)
            return [(f"book/{m.group(3)}", slugify(m.group(2)), text)]
        works = []                               # отделни произведения: номерът е в името на всеки файл
        for n in members:
            m = re.search(r"-\s*(\d+)\.sfb$", n, re.I)
            if not m:
                continue
            title_part = re.sub(r"-\s*\d+\.sfb$", "", n, flags=re.I).split(" - ")[-1]   # „Albena“ от „... - . Albena - 7866.sfb“
            works.append((f"text/{m.group(1)}", slugify(title_part), zf.read(n).decode("utf-8-sig", errors="replace")))
        return works


def main():
    from collections import Counter
    zips = sorted(INBOX.glob("*sfb.zip"))
    if not zips:
        print(f"no sfb.zip files in {INBOX}")
        return
    entries, next_number = read_literature()
    new_lines = []
    for zip_path in zips:
        works = zip_works(zip_path)
        if not works:
            print(f"SKIP {zip_path.name}: no .sfb inside, or a name without a Chitanka number")
            continue
        for key, slug, text in works:
            log = Counter()
            kind, work_id = key.split("/")
            book_author, book_title, docs = parse_sfb(text, log)
            author = book_author or "?"
            if "," in author:                    # "Ангел Тодоров, Михалаки Георгиев": редакторът е изписан до автора
                known = [a.strip() for a in author.split(",") if a.strip() in DEATH_YEAR]
                if known:
                    author = known[0]
            url = f"https://chitanka.info/{key}"
            if key in entries:
                number = entries[key]
            else:
                next_number += 1
                number = next_number
                entries[key] = number
                year = DEATH_YEAR.get(author)
                status = f"свободно, починал {year}" if year and year <= 1955 else "неизвестно, провери"
                new_lines.append(f"{number}. {author} - {book_title or slug} - {url} - {status}")
            if kind == "text" and book_title:
                # отделно произведение (пиеса, повест, роман): действията и главите носят заглавието му отпред,
                # а частта преди първото заглавие (например лицата в пиеса) е под самото заглавие
                docs = [((f"{book_title} — {t}" if t and t != book_title else book_title), a, p) for t, a, p in docs]

            kept, out = [], []
            for title, doc_author, paragraphs in docs:
                if title and SERVICE_TITLE.search(title):
                    log[f"dropped service text: {title[:40]}"] += 1
                    continue
                if doc_author and doc_author.split()[-1] not in author.split():
                    # сравняваме по фамилия: "Славейков" е същият като "Петко Р. Славейков"
                    log[f"dropped text by other author: {doc_author} / {title[:30]}"] += 1
                    continue
                kept.append(title)
                out.append(f"# {title or 'Без заглавие'} ({number}.)")
                out.append("")
                for p in paragraphs:
                    out.append(p)
                    out.append("")
            target = RAW / f"chitanka_{work_id if kind == 'book' else 'text' + work_id}_{slug}.txt"
            target.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
            chars = sum(len(p) for _, _, ps in docs for p in ps)
            print(f"\n=== {zip_path.name} -> {target.name}")
            print(f"  author: {author}   title: {book_title}   source number: {number}")
            print(f"  texts: {len(docs)} parsed, {len(kept)} kept, {chars} chars")
            for k, v in log.most_common():
                print(f"  {k}: {v}")
    if new_lines:
        existing = LITERATURE.read_text(encoding="utf-8-sig") if LITERATURE.exists() else ""
        with open(LITERATURE, "a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")                    # иначе първият нов ред се залепва за последния стар
            f.write("\n".join(new_lines) + "\n")
        print(f"\nadded to literature.txt:\n  " + "\n  ".join(new_lines))


main()

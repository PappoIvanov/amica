"""
Амика · данни за v3 · внасяне на двуезичните корпуси bg-en от OPUS.

Чете data/raw/_opus/opus_<източник>.zip (форматът „moses“: вътре bg-en.bg и bg-en.en, едно изречение на ред,
ред N в единия файл е преводът на ред N в другия) и прави от тях текст за предобучението: последователни
изречения се слепват на блокове по BLOCK реда, всеки блок е документ „# Europarl v8, български, блок N (M.)“,
отделно за двата езика: data/raw/opus_<източник>_bg.txt и opus_<източник>_en.txt. Двойките за превода
(етап 5) ще се вадят от същите zip файлове по-късно, не оттук.
Корпусите са големи (DGT е над 600 MB на език), а официалният регистър е сух, затова всеки източник има
таван на знаците за всеки език (CAPS, в знаци; може да се смени от средата, например CAP_DGT_BG=25e6).
За да не се вземе само началото на файла, блоковете се взимат през равни стъпки по цялата дължина.
Права: Europarl е без известни ограничения, Tatoeba е CC BY 2.0 FR, EU Bookshop и DGT са текстове на
Европейския съюз със свободна повторна употреба (Решение 2011/833/ЕС). Статусът „свободно“ влиза в literature.txt
с два реда на източник, по един за всеки език (адресът завършва на #bg или #en, за да ги различава encode_corpus.py).
Пускане от папката на проекта: py data/import_opus.py
"""
import io
import os
import re
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INBOX = RAW / "_opus"
LITERATURE = RAW / "literature.txt"
BLOCK = 40                                     # изречения в един документ: колкото един дълъг абзац
MIN_WORDS = 3                                  # по-късите редове са номера, дати и заглавия на точки

SOURCES = {   # ключ: (заглавие, адрес, статус, таван bg, таван en)
    "europarl": ("Europarl v8 bg-en, стенограми на Европейския парламент",
                 "https://object.pouta.csc.fi/OPUS-Europarl/v8/moses/bg-en.txt.zip",
                 "свободно, без известни ограничения (Europarl, statmt.org)", 25e6, 15e6),
    "tatoeba": ("Tatoeba bg-en, кратки битови изречения",
                "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/bg-en.txt.zip",
                "свободно, CC BY 2.0 FR (tatoeba.org)", 10e6, 10e6),
    "eubookshop": ("EU Bookshop bg-en, публикации на ЕС",
                   "https://object.pouta.csc.fi/OPUS-EUbookshop/v2/moses/bg-en.txt.zip",
                   "свободно, повторна употреба по Решение 2011/833/ЕС", 15e6, 10e6),
    "dgt": ("DGT-TM bg-en, преводаческата памет на Европейската комисия",
            "https://object.pouta.csc.fi/OPUS-DGT/v2019/moses/bg-en.txt.zip",
            "свободно, повторна употреба по Решение 2011/833/ЕС с посочване на източника: DGT-TM, издание v2019 през OPUS, собственост на Европейската комисия", 25e6, 15e6),
}
LANG_NAME = {"bg": "български", "en": "английски"}
WORD = re.compile(r"\w+")


def read_literature():
    entries, max_number = {}, 0
    if LITERATURE.exists():
        for line in LITERATURE.read_text(encoding="utf-8-sig").splitlines():
            m = re.match(r"^(\d+)\.\s*(.+?)\s+-\s+(.+?)\s+-\s+(\S+)\s+-\s+(.+)$", line.strip())
            if m:
                max_number = max(max_number, int(m.group(1)))
                entries[m.group(4)] = int(m.group(1))
    return entries, max_number


def blocks_from_zip(path, lang, cap):
    """Документите от единия език: блокове по BLOCK изречения, взети през равни стъпки, до тавана cap."""
    with zipfile.ZipFile(path) as zf:
        member = next(n for n in zf.namelist() if n.endswith("." + lang))
        total = zf.getinfo(member).file_size                 # байтове в zip-а; кирилицата е по 2 байта на буква
        est_chars = total / (2 if lang == "bg" else 1)
        stride = max(1, int(est_chars / cap)) if cap else 1  # вземаме всеки stride-ти блок, за да покрием целия файл
        docs, current, chars, block_no = [], [], 0, 0
        with io.TextIOWrapper(zf.open(member), encoding="utf-8", errors="replace") as f:
            for line in f:
                line = " ".join(line.split())
                if len(WORD.findall(line)) < MIN_WORDS:
                    continue
                current.append(line)
                if len(current) == BLOCK:
                    if block_no % stride == 0:
                        docs.append(current)
                        chars += sum(len(s) + 1 for s in current)
                    current, block_no = [], block_no + 1
                    if chars >= cap:
                        break
    return docs, chars


def main():
    zips = sorted(INBOX.glob("opus_*.zip"))
    if not zips:
        print(f"no opus_*.zip in {INBOX}")
        return
    entries, number = read_literature()
    new_lines = []
    for path in zips:
        key = path.stem.replace("opus_", "")
        if key not in SOURCES:
            print(f"  skip {path.name}: unknown source (expected opus_{{{'|'.join(SOURCES)}}}.zip)")
            continue
        title, url, status, cap_bg, cap_en = SOURCES[key]
        caps = {"bg": float(os.environ.get(f"CAP_{key.upper()}_BG", cap_bg)),
                "en": float(os.environ.get(f"CAP_{key.upper()}_EN", cap_en))}
        for lang in ("bg", "en"):
            docs, chars = blocks_from_zip(path, lang, caps[lang])
            src_url = f"{url}#{lang}"
            if src_url in entries:
                source = entries[src_url]
            else:
                number += 1
                source = entries[src_url] = number
                new_lines.append(f"{number}. OPUS - {title}, {LANG_NAME[lang]} - {src_url} - {status}")
            out = []
            for i, block in enumerate(docs, 1):
                out.append(f"# {title.split(',')[0]}, {LANG_NAME[lang]}, блок {i} ({source}.)")
                out.append("")
                out.append(" ".join(block))
                out.append("")
            target = RAW / f"opus_{key}_{lang}.txt"
            target.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
            print(f"  {path.name} [{lang}] -> {target.name}: {len(docs)} blocks, {chars:,} chars (cap {caps[lang]:,.0f}), source {source}")
    if new_lines:
        existing = LITERATURE.read_text(encoding="utf-8-sig") if LITERATURE.exists() else ""
        with open(LITERATURE, "a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write("\n".join(new_lines) + "\n")
        print(f"added {len(new_lines)} lines to literature.txt")


main()

"""
Амика · етап 2 · сглобяване на корпуса.
Чете data/clean/*.txt и data/raw/literature.txt и прави два корпуса:
  corpus_local   всички документи, за твоите обучения
  corpus_public  само свободните, за публикуване
Всеки документ носи автор, източник и лиценз: номерът идва от маркера „(N.)“
в заглавието на суровия файл, а авторът и статусът от ред N в literature.txt.
Пускане от папката на проекта: py data/build_corpus.py
"""
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"

TITLE_RAW = re.compile(r"^#\s*(.+?)\s*\((\d+)\.?\)\s*:?\s*$")   # "# Заглавие (1.)" в суровия файл
TITLE_CLEAN = re.compile(r"^#\s*(.+?)\s*$")                      # "# Заглавие" в чистия файл


def read_sources():
    # ред в literature.txt: "N. Автор - Заглавие - URL - статус"
    sources = {}
    for line in (RAW / "literature.txt").read_text(encoding="utf-8-sig").splitlines():
        m = re.match(r"^(\d+)\.\s*(.+)$", line.strip())
        if not m:
            continue
        number = int(m.group(1))
        parts = [p.strip() for p in m.group(2).split(" - ")]
        if len(parts) < 4:
            print(f"Warning: source {number} in literature.txt ther is not 4 parts, we skip it")
            continue
        author, title, url = parts[0], parts[1], parts[2]
        status = " - ".join(parts[3:])
        if "CC BY-SA" in status:
            license_ = "cc-by-sa"
        elif status.startswith("свободно"):
            license_ = "public-domain"
        elif status.startswith("защитено"):
            license_ = "protected"
        else:
            license_ = "unknown"   # при съмнение се държим като със защитен
            print(f"Warning: source {number}: the status does not start with „свободно“ or „защитено“, so we accept that it is private")
        sources[number] = {"author": author, "title": title, "url": url, "status": status, "license": license_}
    return sources


def raw_markers(path):
    # номерата на източниците от заглавията на суровия файл, по ред на появяване
    markers = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        m = TITLE_RAW.match(line.strip())
        if m:
            markers.append(int(m.group(2)))
    return markers


def clean_documents(path):
    # чистият файл: всеки ред с "# " започва документ; текст преди първото заглавие е документ без заглавие
    docs, title, body = [], None, []
    for line in path.read_text(encoding="utf-8").split("\n"):
        m = TITLE_CLEAN.match(line)
        if m:
            if "".join(body).strip():
                docs.append((title, "\n".join(body).strip()))
            title, body = m.group(1), []
        else:
            body.append(line)
    if "".join(body).strip():
        docs.append((title, "\n".join(body).strip()))
    return docs


TXT = os.environ.get("TXT", "1") != "0"       # TXT=0: без .txt изходите (при 7B знака те удвояват записа на диска)
LOCAL = os.environ.get("LOCAL", "1") != "0"   # LOCAL=0: само corpus_public (corpus_local е същото плюс защитените)


class CorpusWriter:
    # Пише документите един по един, докато пристигат, вместо да ги събира в списък:
    # при 7B знака (v4) списъкът не се събира в паметта (две копия на текста са над 17 GB).
    def __init__(self, name):
        self.name, self.docs, self.chars = name, 0, 0
        # .jsonl: по един документ на ред, с всички полета; за обучение и публикуване
        self.jsonl = open(CLEAN / f"{name}.jsonl", "w", encoding="utf-8")
        # .txt: само текстът, за четене с очи и за символните модели
        self.txt = open(CLEAN / f"{name}.txt", "w", encoding="utf-8") if TXT else None

    def add(self, doc):
        self.jsonl.write(json.dumps(doc, ensure_ascii=False) + "\n")
        if self.txt:
            if doc["title"]:
                self.txt.write(doc["title"] + "\n\n")
            self.txt.write(doc["text"] + "\n\n\n")
        self.docs += 1
        self.chars += doc["chars"]

    def close(self):
        self.jsonl.close()
        if self.txt:
            self.txt.close()


sources = read_sources()
writers = {"corpus_public": CorpusWriter("corpus_public")}
if LOCAL:
    writers["corpus_local"] = CorpusWriter("corpus_local")
rows = []
seen_texts = {}   # отпечатък на нормализирания текст -> id на първия документ с този текст (една приказка в два сборника)
for path in sorted(CLEAN.glob("*.txt")):
    if path.name.startswith("corpus_"):   # нашите собствени изходи от предишно пускане
        continue
    docs = clean_documents(path)
    raw_path = RAW / path.name
    markers = raw_markers(raw_path) if raw_path.exists() else []
    if not markers:
        # файл без нито един ред "# Заглавие (N.)" не е текст за корпуса (например разговорите за етап 5)
        print(f"skip {path.name}: no '# Title (N.)' markers, not a corpus source")
        continue
    if len(markers) != len(docs):
        # маркерите не са по един на документ: ползваме най-честия за целия файл
        dominant = Counter(markers).most_common(1)[0][0] if markers else None
        print(f"Warning: {path.name}: {len(docs)} the document, but {len(markers)} marker; i use this source {dominant} for all")
        markers = [dominant] * len(docs)
    file_chars, file_licenses = 0, Counter()
    for i, ((title, body), source_id) in enumerate(zip(docs, markers), 1):
        src = sources.get(source_id)
        license_ = src["license"] if src else "unknown"
        if src is None:
            print(f"Warning: {path.name}: source {source_id} in missing in literature.txt")
        doc = {
            "id": f"{path.stem}/{i:03d}",
            "title": title,
            "author": src["author"] if src else None,
            "source": source_id,
            "license": license_,
            "chars": len(body),
            "text": body,
        }
        # отпечатък от 16 байта вместо самия текст: иначе ключовете са второ копие на целия корпус
        key = hashlib.blake2b(" ".join(body.split()).encode("utf-8"), digest_size=16).digest()
        if key in seen_texts:
            print(f"duplicate: {doc['id']} is the same text as {seen_texts[key]}, skipped")
            continue
        seen_texts[key] = doc["id"]
        if LOCAL:
            writers["corpus_local"].add(doc)
        if license_ in ("public-domain", "cc-by-sa"):
            writers["corpus_public"].add(doc)
        file_chars += len(body)
        file_licenses[license_] += 1
    rows.append((path.name, len(docs), file_chars, ", ".join(f"{k}: {v}" for k, v in file_licenses.items())))
    print(f"{path.name}: {len(docs)} documents, {file_chars:,} chars", flush=True)

for w in writers.values():
    w.close()

print(f"{'file':24} {'doc.':>5} {'signs':>8}  license")
for name, n_docs, chars, lic in rows:
    print(f"{name:24} {n_docs:5d} {chars:8d}  {lic}")
print()
for name in ("corpus_local", "corpus_public"):
    if name in writers:
        w = writers[name]
        print(f"{name}: {w.docs} documents, {w.chars} characters, {w.chars / 1e6:.2f} millions")
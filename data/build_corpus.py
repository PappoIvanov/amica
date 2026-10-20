"""
Амика · етап 2 · сглобяване на корпуса.
Чете data/clean/*.txt и data/raw/literature.txt и прави два корпуса:
  corpus_local   всички документи, за твоите обучения
  corpus_public  само свободните, за публикуване
Всеки документ носи автор, източник и лиценз: номерът идва от маркера „(N.)“
в заглавието на суровия файл, а авторът и статусът от ред N в literature.txt.
Пускане от папката на проекта: py data/build_corpus.py
"""
import json
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


def write_corpus(name, docs):
    # .jsonl: по един документ на ред, с всички полета; за обучение и публикуване
    with open(CLEAN / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    # .txt: само текстът, за четене с очи и за символните модели
    with open(CLEAN / f"{name}.txt", "w", encoding="utf-8") as f:
        for d in docs:
            if d["title"]:
                f.write(d["title"] + "\n\n")
            f.write(d["text"] + "\n\n\n")


sources = read_sources()
local, public, rows = [], [], []
seen_texts = {}   # нормализиран текст -> id на първия документ с този текст (една приказка в два сборника)
for path in sorted(CLEAN.glob("*.txt")):
    if path.name.startswith("corpus_"):   # нашите собствени изходи от предишно пускане
        continue
    docs = clean_documents(path)
    raw_path = RAW / path.name
    markers = raw_markers(raw_path) if raw_path.exists() else []
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
        key = " ".join(body.split())
        if key in seen_texts:
            print(f"duplicate: {doc['id']} is the same text as {seen_texts[key]}, skipped")
            continue
        seen_texts[key] = doc["id"]
        local.append(doc)
        if license_ in ("public-domain", "cc-by-sa"):
            public.append(doc)
        file_chars += len(body)
        file_licenses[license_] += 1
    rows.append((path.name, len(docs), file_chars, ", ".join(f"{k}: {v}" for k, v in file_licenses.items())))

write_corpus("corpus_local", local)
write_corpus("corpus_public", public)

print(f"{'file':24} {'doc.':>5} {'signs':>8}  license")
for name, n_docs, chars, lic in rows:
    print(f"{name:24} {n_docs:5d} {chars:8d}  {lic}")
print()
for name, docs in [("corpus_local", local), ("corpus_public", public)]:
    chars = sum(d["chars"] for d in docs)
    print(f"{name}: {len(docs)} documents, {chars} characters, {chars / 1e6:.2f} millions")
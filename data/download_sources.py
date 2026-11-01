"""
Амика · данни за v4 · сваляне на източниците със скрипт, без ръчно щракане.

Описът на източниците (какво, откъде, права, размер) е в data/raw/DOWNLOAD_LIST_3.md; тук са адресите и
папките. Всеки източник отива в папката, която неговият importer вече чете:
  wikipedia_en       parquet файлове на пълната английска Уикипедия (wikimedia/wikipedia, 20231101.en) ->
                     data/raw/_wikipedia_en_full; кои части: SHARDS от средата (по подразбиране 7 части, разпръснати
                     из 41-те, към вече свалените 0, 20 и 35); после py data/import_wikipedia.py с WIKI=en_full
  wikisource_bg      parquet на Българския Уикиизточник (wikimedia/wikisource, 20231201.bg) -> data/raw/_wikisource_bg
  opus               новите двуезични архиви от OPUS (JRC-Acquis, WikiMatrix, GlobalVoices) -> data/raw/_opus/opus_<име>.zip
  gutenberg_catalog  каталогът на Project Gutenberg (pg_catalog.csv) -> data/raw/_gutenberg; после py data/select_gutenberg.py
  gutenberg_books    книгите от data/raw/_gutenberg/_index.json, които още ги няма като pg<номер>.txt, от огледалото
                     gutenberg.pglaf.org с пауза PAUSE секунди (2 по подразбиране; политиката на gutenberg.org праща
                     програмите към огледалата)
  stenograms         стенограмите от пленарните заседания на Народното събрание през същото API като законите:
                     https://www.parliament.bg/api/v1/pl-sten/<id>, номерата са поредни, но не по дата (архивът стига
                     до 1879 г.; невалидните номера връщат HTML и се записват в _missing.txt, за да не се пробват пак);
                     -> data/raw/_stenograms/sten_<id>.json; STEN_MAX (12500) е горната граница, STEN_PAUSE (0.5 s) паузата;
                     после py data/import_stenograms.py
Свалянето е с продължаване: готов файл с верен размер се прескача, недовършен .part се дотегля с Range.
Пускане от папката на проекта: py data/download_sources.py wikisource_bg opus gutenberg_catalog wikipedia_en
                               py data/download_sources.py gutenberg_books          (след select_gutenberg.py)
                               py data/download_sources.py stenograms
                               py data/download_sources.py all                      (всичко по реда горе)
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
AGENT = "amica-data/1.0 (personal research corpus; https://github.com/PappoIvanov/amica)"
CHUNK = 1 << 20                                                # по 1 MB
RETRIES = 3
PAUSE = float(os.environ.get("PAUSE", 2.0))                    # пауза между книгите от Gutenberg
SHARDS = [int(x) for x in os.environ.get("SHARDS", "1,5,10,15,25,30,40").split(",")]

HF_WIKIPEDIA = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en/"
HF_WIKISOURCE = "https://huggingface.co/datasets/wikimedia/wikisource/resolve/main/20231201.bg/"
OPUS = {   # ключ в import_opus.py: адрес
    "jrc_acquis": "https://object.pouta.csc.fi/OPUS-JRC-Acquis/v3.0/moses/bg-en.txt.zip",
    "wikimatrix": "https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/bg-en.txt.zip",
    "globalvoices": "https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/moses/bg-en.txt.zip",
}
GUTENBERG_CATALOG = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
MIRROR = "https://gutenberg.pglaf.org"
STENOGRAM = "https://www.parliament.bg/api/v1/pl-sten/"
STEN_MAX = int(os.environ.get("STEN_MAX", 12500))
STEN_PAUSE = float(os.environ.get("STEN_PAUSE", 0.5))


def jobs(source):
    """Списък (адрес, целеви файл) за един източник."""
    if source == "wikipedia_en":
        folder = RAW / "_wikipedia_en_full"
        return [(HF_WIKIPEDIA + f"train-{i:05d}-of-00041.parquet", folder / f"train-{i:05d}-of-00041.parquet") for i in SHARDS]
    if source == "wikisource_bg":
        return [(HF_WIKISOURCE + "train-00000-of-00001.parquet", RAW / "_wikisource_bg" / "train-00000-of-00001.parquet")]
    if source == "opus":
        return [(url, RAW / "_opus" / f"opus_{key}.zip") for key, url in OPUS.items()]
    if source == "gutenberg_catalog":
        return [(GUTENBERG_CATALOG, RAW / "_gutenberg" / "pg_catalog.csv")]
    if source == "gutenberg_books":
        index = RAW / "_gutenberg" / "_index.json"
        if not index.exists():
            print(f"missing {index}: run select_gutenberg.py first")
            return []
        books = json.loads(index.read_text(encoding="utf-8"))
        return [(f"{MIRROR}/cache/epub/{b['number']}/pg{b['number']}.txt", RAW / "_gutenberg" / f"pg{b['number']}.txt")
                for b in books if not (RAW / "_gutenberg" / f"pg{b['number']}.txt").exists()]
    raise SystemExit(f"unknown source: {source}")


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length") or 0)


def fetch(url, target, head=True):
    """Сваля адреса в target през target.part с продължаване. Връща True при успех.
    head=False прескача заявката HEAD за размера (при 4000 книги тя е излишен втори път до сървъра)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, RETRIES + 1):
        try:
            expected = remote_size(url) if head else 0
            if target.exists() and (expected == 0 or target.stat().st_size == expected):
                print(f"  ok      {target.name} ({target.stat().st_size:,} bytes, already here)")
                return True
            have = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": AGENT}
            if have and expected and have < expected:
                headers["Range"] = f"bytes={have}-"
            elif have:
                have = 0
                part.unlink()
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:                     # сървърът не продължи: отначало
                    f.seek(0), f.truncate(), f.flush()
                    have = 0
                done, last = have, time.time()
                while True:
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if time.time() - last > 30:
                        print(f"  ...     {target.name} {done / 1e6:,.0f} / {expected / 1e6:,.0f} MB")
                        last = time.time()
            if expected and part.stat().st_size != expected:
                raise OSError(f"size mismatch: {part.stat().st_size} of {expected}")
            part.replace(target)
            print(f"  got     {target.name} ({target.stat().st_size:,} bytes)")
            return True
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"  retry {attempt}/{RETRIES} {target.name}: {e}")
            time.sleep(5 * attempt)
    print(f"  FAILED  {target.name}")
    return False


def download_stenograms():
    """Обхожда номерата 1..STEN_MAX; валиден отговор е JSON с Pl_Sten_body, всичко друго е липсващ номер."""
    folder = RAW / "_stenograms"
    folder.mkdir(parents=True, exist_ok=True)
    missing_path = folder / "_missing.txt"
    missing = set(int(x) for x in missing_path.read_text().split()) if missing_path.exists() else set()
    got = skipped = failed = 0
    started = time.time()
    for sid in range(1, STEN_MAX + 1):
        target = folder / f"sten_{sid}.json"
        if target.exists() or sid in missing:
            skipped += 1
            continue
        req = urllib.request.Request(STENOGRAM + str(sid), headers={"User-Agent": AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
        except urllib.error.HTTPError as e:                      # 404/500: несъществуващ номер, не мрежова грешка
            data = b""
            if e.code not in (404, 500):
                failed += 1
                print(f"  error   sten_{sid}: {e}")
                time.sleep(5)
                continue
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            failed += 1
            print(f"  error   sten_{sid}: {e}")
            time.sleep(5)
            continue
        if data[:1] == b"{" and b"Pl_Sten_body" in data:
            target.write_bytes(data)
            got += 1
        else:
            missing.add(sid)
            with open(missing_path, "a") as f:
                f.write(f"{sid}\n")
        if (got + len(missing)) % 200 == 0:
            print(f"  ...     id {sid}: {got} got, {len(missing)} missing, {(time.time() - started) / 60:.0f} min")
        time.sleep(STEN_PAUSE)
    print(f"stenograms: {got} new, {skipped} already here or known missing, {failed} errors, "
          f"{len(list(folder.glob('sten_*.json')))} files in {folder}")


def main():
    names = sys.argv[1:] or ["all"]
    if names == ["all"]:
        names = ["wikisource_bg", "opus", "gutenberg_catalog", "wikipedia_en", "gutenberg_books", "stenograms"]
    started = time.time()
    for name in names:
        if name == "stenograms":
            download_stenograms()
            continue
        todo = jobs(name)
        print(f"\n{name}: {len(todo)} file(s)")
        ok = 0
        for i, (url, target) in enumerate(todo):
            if name == "gutenberg_books" and i:
                time.sleep(PAUSE)
            ok += fetch(url, target, head=(name != "gutenberg_books"))
        print(f"{name}: {ok} of {len(todo)} done")
    print(f"\nfinished in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()

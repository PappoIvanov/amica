"""
Амика * етап 2 * нормализатор на сурови текстове.
Чете всеки data/raw/*.txt, уеднаквява знаците и подредбата и записва
data/clean/<същото име>.txt. Не променя нищо по смисъл, само формата
и докладва всичко, което е поправил или не разбира,
Пускане от папката на проекта: py data/normalize.py
"""
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
SKIP = {"literature.txt"}   # файлове в data/raw, които не са текст за модела

# латински букви, които изглеждат като кирилица; заменят се само вътре в думата на кирилица
HOMOGLYPHS = str.maketrans("aceopxyABCEHKMOPTX", "асеорхуАВСЕНКМОРТХ")
CYRILLIC = re.compile(r"[\u0400-\u04FF]")
WORD = re.compile(r"[A-Za-z\u0400-\u04FF]+")
# всичко извън кирилицата и този набор се докладва като непознат знак
ALLOWED = set(" \n0123456789.,;:!?-—„“'()…")
# ред със заглави: "# Заглавие", по желание с маркер "(1.)" и с двоеточие накрая
TITLE = re.compile(r"^#\s*(.+?)\s*(\(\d+\.?\))?\s*:?\s*$")
STRAIGHT_QUOTES = re.compile(r'"([^"\n]+)"')       # "текст" -> "текст"
OPENING_QUOTE = re.compile(r"(^|[\s(])“", re.M)   # " в началото на думата е отваряща -> "
ACCENTS = re.compile("[\u0300\u0301]")       # ударения над гласни, остатък от печатното издание
SERVICE_LINE = re.compile(r"[*.\s]+")        # редове само от звездички или точки: разделители на сцени, пропуснати стихове

# прости замени: (какво, с какво, име за доклада)
REPLACEMENTS = [
    ("\u00a0", " ", "твърд интервал"), ("\u00ad", "", "мек пренос"),
    ("\u200b", "", "нулев интервал"), ("\ufeff", "", "BOM"), ("\t", " ", "табулация"),
    ("«", "„", "кавички «"), ("»", "“", "кавички »"), ("”", "“", "кавички ”"),
    ("–", "—", "късо тире"), ("...", "…", "многоточие"),
    ("`", "", "ударение (`)"), ("’", "'", "апостроф ’"),
]


HOMOGLYPH_LETTERS = set("aceopxyABCEHKMOPTX")


def fix_homoglyphs(text, log):
    # дума, в която има и кирилица и латински букви, почти винаги е грешка от източника;
    # същото важи за къса дума само от букви двойници: латинско "e" вместо "е", "My" вместо "Му"
    def repl(m):
        word = m.group(0)
        mixed = CYRILLIC.search(word) and re.search(r"[A-Za-z]", word)
        short_latin = (len(word) <= 3 and all(ch in HOMOGLYPH_LETTERS for ch in word)
                       and not re.fullmatch(r"[IVXLCDM]+", word))   # римските числа като "X" си остават
        if mixed or short_latin:
            fixed = word.translate(HOMOGLYPHS)
            if fixed != word:
                log[f"homoglyph: {word} -> {fixed}"] += 1
            return fixed
        return word
    return WORD.sub(repl, text)

def normalize_text(text, log, english=False):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # NFC: една буква = един знак (напр. "й" да не е "и" + отделен знак за кратката "и")
    text = unicodedata.normalize("NFC", text)
    text, n = ACCENTS.subn("", text)
    if n:
        log["accent"] += n
    for bad, good, name in REPLACEMENTS:
        if english and "кавички" in name:
            continue
        n = text.count(bad)
        if n:
            log[name] += n
            text = text.replace(bad, good)
    
    n = text.count(" - ")
    if n:
        log["slash with spaces"] += n
        text = text.replace(" - ", " — ")   # тире с интервали и тире за диалог, не чертичка в дума
    
    if not english:                              # българските кавички „“ само за български текст
        text, n = STRAIGHT_QUOTES.subn(r"„\1“", text)
        if n:
            log["straight quotes"] += n
        text, n = OPENING_QUOTE.subn(r"\1„", text)
        if n:
            log["opening quote “"] += n

    if not english:                              # на английски "a", "He", "OK" са истински думи, не двойници
        text = fix_homoglyphs(text, log)
    text = re.sub(r" {2,}", " ", text)          # повторени интервали
    kept = []
    for line in text.split("\n"):
        line = line.strip()
        if line == "Край" or SERVICE_LINE.fullmatch(line):
            log[f"official row: {line}"] += 1
            continue
        kept.append(line)
    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)      # най-много един празен ред
    return text.strip() + "\n"


def split_documents(text):
    # всеки ред със заглавие е начало на нов документ; текст преди първото заглавие е документ без заглавие
    docs, title, body = [], None, []
    for line in text.split("\n"):
        m = TITLE.match(line)
        if m:
            docs.append((title, "\n".join(body).strip()))
            title, body = m.group(1), []
        else:
            body.append(line)
    docs.append((title, "\n".join(body).strip()))
    return [(t, b) for t, b in docs if b]


def process (path):
    log = Counter()
    raw = path.read_text(encoding="utf-8-sig")
    text = normalize_text(raw, log, english=path.name.startswith("wikipedia_en_"))
    docs = split_documents(text)
    out = []
    for title, body in docs:
        if title:
            out.append(f"# {title}")   # заглавието остава, маркерът на източника пада
        out.append(body)
        out.append("")
    clean = "\n".join(out).strip() + "\n"
    (CLEAN / path.name).write_text(clean, encoding="utf-8")

    # доклад
    print(f"\n=== {path.name} ===")
    untitled = sum(1 for t, _ in docs if not t)
    print(f"signs: {len(raw)} -> {len(clean)}   documents: {len(docs)}   without title: {untitled}")
    for k, v in log.most_common():
        print(f"  correction: {k}: {v}")
    bodies = "\n".join(b for _, b in docs)
    odd = Counter(ch for ch in bodies if ch not in ALLOWED and not CYRILLIC.match(ch))
    if odd:
        print("  unknown signs:", " ".join(f"{ch!r}:{n}" for ch, n in odd.most_common(20)))
    paragraphs = [p for p in bodies.split("\n") if len(p) > 60]
    dups = [p for p, n in Counter(paragraphs).items() if n > 1]
    if dups:
        print(f"  repeated paragraphs: {len(dups)}  for example: {dups[0][:60]}...")
    print(f"  dictionary: {len(set(bodies))} different signs")



CLEAN.mkdir(exist_ok=True)
files = [f for f in sorted(RAW.glob("*.txt")) if f.name not in SKIP]
if not files:
    print("There isn't files in data/raw")
    sys.exit(1)
for f in files:
    done = CLEAN / f.name
    if done.exists() and done.stat().st_mtime > f.stat().st_mtime:
        print(f"up to date: {f.name}")   # чистият файл е по-нов от суровия; при нови правила изтрий data/clean/*.txt
        continue
    process(f)

# чисти файлове, чийто суров файл вече го няма (преместен или изтрит), се махат, за да не влязат в корпуса
raw_names = {f.name for f in files}
for stale in CLEAN.glob("*.txt"):
    if stale.name not in raw_names and not stale.name.startswith("corpus_"):
        stale.unlink()
        print(f"removed stale output: {stale.name}")
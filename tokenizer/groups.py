"""
Амика · групите на документите по източник (от 2026-11-02, за v4).

Една функция за train_bpe.py и encode_corpus.py, за да не се разминават: групата се познава по адреса
на източника в data/raw/literature.txt, не по твърди номера. Шест групи, всяка в свой файл с токени,
за да се смесват с тегла при обучението и да се мери val за всяка поотделно:
  bg_wiki  Българска Уикипедия и WikiMatrix bg (изречения от Уикипедия)
  sten     стенограмите на Народното събрание: говорим български
  law      законите, Конституцията, Търговският закон и официалните корпуси на ЕС на български
           (Europarl, DGT, EU Bookshop, JRC-Acquis)
  lit      Читанка, Уикиизточник, Tatoeba bg и GlobalVoices bg: литература, битови изречения, новини
  en       всичко английско: Simple English и English Wikipedia, Gutenberg, английските страни на OPUS
  pairs    двойките „български ||| английски“ от OPUS, за превода в предобучението
Проверка от папката на проекта: py tokenizer/groups.py      (колко източника има във всяка група)
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LITERATURE = ROOT / "data" / "raw" / "literature.txt"
GROUP_NAMES = ["bg_wiki", "sten", "law", "lit", "en", "pairs"]


def group_of_url(url):
    """Групата по адреса на източника. Редът на проверките е важен: при OPUS първо се гледа краят на адреса."""
    if url.endswith("#pairs"):
        return "pairs"                            # opus_<източник>_pairs.txt
    if url.endswith("#en"):
        return "en"                               # английската страна на OPUS
    if "20231101.bg" in url or "OPUS-WikiMatrix" in url:
        return "bg_wiki"                          # Уикипедия bg и WikiMatrix bg (адресът завършва на #bg)
    if "parliament.bg/bg/plenaryst" in url:
        return "sten"
    if "OPUS-Tatoeba" in url or "OPUS-GlobalVoices" in url:
        return "lit"                              # битови изречения и новини, не официален регистър
    if "OPUS-" in url or "parliament.bg" in url or "bg.wikisource.org" in url:
        return "law"                              # Europarl, DGT, EU Bookshop, JRC-Acquis bg; законите; Търговският закон
    if "wikimedia/wikipedia" in url or "gutenberg.org" in url:
        return "en"                               # Simple English, English Wikipedia, книгите от Gutenberg
    return "lit"                                  # Читанка, Уикиизточник (wikimedia/wikisource) и всичко останало


def source_groups(path=LITERATURE):
    """{номер на източник: група} за всеки ред „N. Автор - Заглавие - URL - статус“ на literature.txt."""
    groups = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        m = re.match(r"^(\d+)\.\s*(.+)$", line.strip())
        if not m:
            continue
        parts = [p.strip() for p in m.group(2).split(" - ")]
        url = next((p for p in parts if p.startswith("http")), "")   # заглавието също може да съдържа „ - “
        groups[int(m.group(1))] = group_of_url(url)
    return groups


if __name__ == "__main__":
    from collections import Counter
    groups = source_groups()
    counts = Counter(groups.values())
    for name in GROUP_NAMES:
        print(f"{name:8} {counts[name]:5} sources")
    print(f"total    {len(groups):5} sources")

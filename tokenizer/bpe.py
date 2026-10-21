"""
Амика · етап 4 · собствен BPE токенизатор (byte pair encoding).
Част 1: байтова основа и предварително рязане на текста на парчета. Част 2: обучение на сливанията.
Част 3: encode, decode, save, load. Част 5: служебни токени.

Защо байтове: всеки текст, на какъвто и да е език, е редица от байтове в UTF-8,
тоест числа от 0 до 255. Началният речник е винаги 256 токена и няма непознат знак.
Кирилската буква е 2 байта (например „Д“ = 208 148), латинската е 1 байт.
Първите сливания на BPE ще сглобят кирилските букви обратно от двата им байта,
после срички, после цели думи.

Защо парчета: преди сливанията текстът се реже на парчета (дума, число, препинателен
знак, празно място). Сливания стават само вътре в парче, никога през границата му,
за да не се появи токен като „та и“ от края на една дума и началото на друга.
Интервалът пред думата остава към нея (" кон"), както при GPT.

Пускане от папката на проекта: py tokenizer/bpe.py
"""
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path(os.environ.get("CORPUS", ROOT / "data" / "clean" / "corpus_public.txt"))

# Правилата се пробват отляво надясно; печели първото, което пасне на текущото място:
#    ?[^\W\d_]+        дума: букви на всеки език, с евентуалния интервал пред тях
#    ?\d{1,3}          число: най-много по 3 цифри наведнъж, за да не станат дългите числа отделни токени
#    ?(?:[^\s\w]|_)+   препинателни знаци и символи, пак с интервала пред тях
#   \s*[\r\n]+         нов ред заедно с празните места пред него („\n\n“ между абзаци е едно парче)
#   \s+(?!\S)          празни места, но без последния интервал пред дума: той остава за думата
#   \s+                останалите празни места (например в края на текста)
# [^\W\d_] значи „знак от дума, който не е цифра и не е долна черта“, тоест буква на всеки език.
PATTERN = re.compile(r" ?[^\W\d_]+| ?\d{1,3}| ?(?:[^\s\w]|_)+|\s*[\r\n]+|\s+(?!\S)|\s+")


def pretokenize(text):
    """Реже текста на парчета. Слепени обратно, парчетата дават точно същия текст."""
    return PATTERN.findall(text)


def to_bytes(chunk):
    """Парче текст → списък от байтове (числа 0..255), началните токени на BPE."""
    return list(chunk.encode("utf-8"))


def kind(chunk):
    # вид на парчето, само за статистиката
    body = chunk.lstrip(" ")
    if body == "":
        return "space"
    if body.isspace():
        return "newline"
    if body.isdigit():
        return "number"
    if body[0].isalpha():
        return "word"
    return "punct"


def merge(ids, pair, new_id):
    """Заменя всяко срещане на двойката pair в списъка ids с новия токен new_id."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def show(token_bytes):
    # токенът като текст за печат; половин кирилска буква (единичен байт) излиза като „�“
    return repr(token_bytes.decode("utf-8", errors="replace"))


class BPE:
    """Речник от 256 байта плюс научените сливания: сливане номер i дава токен 256 + i."""

    def __init__(self):
        self.merges = []                                 # двойки (a, b) по реда на научаване
        self.vocab = [bytes([i]) for i in range(256)]    # токен → байтовете, които представлява
        self.ranks = {}                                  # двойка → токенът, който дава (за encode)
        self.cache = {}                                  # парче → токените му, за да не се кодира пак
        self.special = {}                                # име на служебен токен → номер

    def add_merge(self, pair):
        # записва сливане: нов токен с номер len(vocab), чиито байтове са байтовете на двата
        new_id = len(self.vocab)
        self.merges.append(pair)
        self.ranks[pair] = new_id
        self.vocab.append(self.vocab[pair[0]] + self.vocab[pair[1]])
        return new_id

    def add_special(self, name):
        # служебен токен, например <|doc|>: получава следващия свободен номер;
        # encode никога не го прави от текст, слага се само нарочно от кода
        new_id = len(self.vocab)
        self.vocab.append(name.encode("utf-8"))
        self.special[name] = new_id
        return new_id

    def train(self, text, vocab_size, verbose=True):
        # 1. различните парчета и колко пъти се срещат: „ на“ се брои веднъж с тегло 8367,
        #    вместо да се обработва 8367 пъти
        counts = Counter(pretokenize(text))
        words = [to_bytes(chunk) for chunk in counts]    # всяко различно парче като списък от токени
        freqs = list(counts.values())                    # теглото му
        n_bytes = sum(len(word) * f for word, f in zip(words, freqs))
        # 2. броим съседните двойки във всички парчета, с тегло честотата на парчето,
        #    и помним в кои парчета се среща всяка двойка
        pair_counts = Counter()
        where = defaultdict(set)
        for wi, (word, f) in enumerate(zip(words, freqs)):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += f
                where[pair].add(wi)
        # 3. сливания: най-честата двойка става нов токен и се пренаписват
        #    само парчетата, в които я има
        n_merges = vocab_size - len(self.vocab)
        t0 = time.time()
        for i in range(n_merges):
            if not pair_counts:
                break                                    # всяко парче е вече един токен
            pair = max(pair_counts, key=pair_counts.get)
            count = pair_counts[pair]
            new_id = self.add_merge(pair)
            for wi in list(where[pair]):
                word, f = words[wi], freqs[wi]
                for p in zip(word, word[1:]):            # старият принос на парчето се маха
                    pair_counts[p] -= f
                    if pair_counts[p] == 0:
                        del pair_counts[p]
                    where[p].discard(wi)
                word = merge(word, pair, new_id)
                words[wi] = word
                for p in zip(word, word[1:]):            # и се добавя новият
                    pair_counts[p] += f
                    where[p].add(wi)
            if verbose and (i % 100 == 0 or i == n_merges - 1):
                print(f"merge {i + 1}/{n_merges}: {pair} -> {new_id} {show(self.vocab[new_id])}"
                      f"  count {count:,}  {time.time() - t0:.0f}s")
        n_tokens = sum(len(word) * f for word, f in zip(words, freqs))
        if verbose:
            print(f"done: {len(self.merges)} merges, {n_bytes:,} bytes -> {n_tokens:,} tokens, "
                  f"{n_bytes / n_tokens:.2f} bytes per token")
        return n_tokens

    def encode_chunk(self, chunk):
        # едно парче: тръгва от байтовете и прилага сливанията в реда, в който са научени,
        # тоест на всяка стъпка най-рано наученото сливане, което пасва някъде в парчето
        ids = to_bytes(chunk)
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break                                    # никоя двойка в парчето не е научена
            ids = merge(ids, pair, self.ranks[pair])
        return ids

    def encode(self, text):
        """Текст → списък от номера на токени."""
        ids = []
        for chunk in pretokenize(text):
            if chunk not in self.cache:                  # повтарящо се парче се кодира само веднъж
                self.cache[chunk] = self.encode_chunk(chunk)
            ids.extend(self.cache[chunk])
        return ids

    def decode(self, ids):
        """Номера на токени → текст."""
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")    # половин буква без другата си половина става „�“

    def save(self, path):
        """Записва сливанията в JSON; речникът се възстановява от тях при load."""
        data = {"merges": self.merges, "special": list(self.special)}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        bpe = cls()
        for a, b in data["merges"]:
            bpe.add_merge((a, b))                        # JSON пази двойките като списъци, връщаме ги в tuple
        for name in data.get("special", []):             # стар файл без служебни токени също се чете
            bpe.add_special(name)
        return bpe


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    # проба: обучение върху 2 милиона знака с речник 512, после кодиране, обратно и запис
    with open(CORPUS, encoding="utf-8") as f:
        text = f.read(2_000_000)
    bpe = BPE()
    bpe.train(text, vocab_size=512, verbose=False)

    sample = "Дойде пролет и насипа цвят, нали? Yes, in 1878!"
    ids = bpe.encode(sample)
    print(f"text   : {sample!r}")
    print(f"ids    : {ids}")
    print(f"tokens : {'|'.join(bpe.decode([i]) for i in ids)}")
    print(f"{len(sample)} chars -> {len(ids)} tokens")
    assert bpe.decode(ids) == sample, "decode(encode(text)) must give the text back"

    path = ROOT / "checkpoints" / "bpe_test_512.json"
    bpe.save(path)
    loaded = BPE.load(path)
    assert loaded.encode(sample) == ids, "the loaded tokenizer must encode the same way"
    print(f"saved and loaded {path.name}: {len(loaded.vocab)} tokens, same ids")

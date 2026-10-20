"""
Амика v0 * символен биграм с броене.
Таблица N[a][b] = колко пъти знакът b следва знакът a в текста.
Предсказване: редът на последния знак, превърнат във вероятности.
"""
import json
import math
import random



class BigramCounts:
    def __init__(self):
        self.chars = []   # речникът: всички различни знаци, подредени
        self.stoi = []    # знак -> индекс в таблицата
        self.P = []       # таблицата с вероятности, ред по ред

    def _build(self, chars, N):
        # от броячите N правим вероятности: всеки ред се дели на сбора си
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.P = [[n / sum(row) for n in row] for row in N]

    def fit(self, text):
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        V = len(chars)
        # броячите тръгват от 1, не от 0 (изглаждане): никоя двойка няма вероятност 0,
        # иначе log(0) в loss гърми при двойка, която липсва в текста
        N = [[1] * V for _ in range(V)]
        for a, b in zip(text, text[1:]):
            N[stoi[a]][stoi[b]] += 1
        self.N = N
        self._build(chars, N)

    def probs(self, ch):
        # вероятности за следващия знак след ch
        return self.P[self.stoi[ch]]

    def loss(self, text):
        # средна отрицателна логаритмична правдоподобност: колко "изненадан" е моделът от всяка следваща буква.
        # По-малко = по-добре. Същата мярка ще ползва Amica v1.
        nll = 0.0
        for a, b in zip(text, text[1:]):
            nll -= math.log(self.probs(a)[self.stoi[b]])
        return nll / (len(text) - 1)

    def generate(self, prefix, n_chars):
        # тръгваме от последния знак на prefix и теглим знак според таблицата
        out = prefix
        ch = prefix[-1] if prefix and prefix[-1] in self.stoi else "\n"
        for _ in range(n_chars):
            ch = random.choices(self.chars, weights=self.probs(ch))[0]
            out += ch
        return out

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars, "N": self.N,}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        m = cls()
        m.N = d["N"]
        m._build(d["chars"], d["N"])
        return m
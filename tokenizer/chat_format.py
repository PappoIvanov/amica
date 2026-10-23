"""
Амика · етап 5 · форматът на разговора.

Разговорът е документ от реплики. Всяка реплика е служебен токен за ролята, обикновен текст и <|end|>:
  <|doc|> <|system|> указания <|end|> <|user|> въпрос <|end|> <|amica|> отговор <|end|> ...
Репликите на Амика са три вида: мисъл (<|amica|><|think|> кратко разсъждение <|end|>), извикване на
инструмент (<|amica|><|tool|> име и аргументи <|end|>) и отговор (<|amica|> текст <|end|>). Резултатът от
инструмента е реплика <|result|>, вписана от приложението. Генерирането винаги спира на <|end|>.
При дообучаване загубата брои само това, което Амика пише: нейния текст, нейните <|tool|> и <|end|>.
Затова encode_conversation връща и маска: 1 за токените, които се учат, 0 за контекста.

Служебни токени (номерата са фиксирани, 8184 е <|doc|> от етап 4):
  8185 <|user|>  8186 <|amica|>  8187 <|end|>  8188 <|tool|>  8189 <|result|>  8190 <|system|>  8191 <|think|>
Пускане от папката на проекта: py tokenizer/chat_format.py
При първото пускане добавя седемте токена в tokenizer/amica_bpe.json; после само показва примера.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # за да намери bpe.py
from bpe import BPE

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tokenizer" / "amica_bpe.json"
SPECIAL = ["<|doc|>", "<|user|>", "<|amica|>", "<|end|>", "<|tool|>", "<|result|>", "<|system|>", "<|think|>"]
FIRST_SPECIAL = 8184
ROLE_TOKEN = {"system": "<|system|>", "user": "<|user|>", "amica": "<|amica|>", "result": "<|result|>"}
AMICA_KINDS = {"think": "<|think|>", "tool": "<|tool|>"}   # реплики на Амика с маркер след <|amica|>


def load_tokenizer(path=TOKENIZER):
    """Зарежда токенизатора и гарантира, че осемте служебни токена са в него с точните номера."""
    bpe = BPE.load(path)
    missing = [name for name in SPECIAL if name not in bpe.special]
    for name in missing:
        bpe.add_special(name)
    if missing:
        bpe.save(path)
        print(f"added {len(missing)} special tokens to {Path(path).name}")
    numbers = [bpe.special[name] for name in SPECIAL]
    assert numbers == list(range(FIRST_SPECIAL, FIRST_SPECIAL + len(SPECIAL))), bpe.special
    return bpe


def encode_conversation(bpe, turns):
    """Разговор → (токени, маска).
    turns е списък от {"role": "system" | "user" | "amica" | "think" | "tool" | "result", "text": "..."}.
    Маската е 1 за това, което Амика трябва да се научи да пише, и 0 за всичко друго."""
    ids = [bpe.special["<|doc|>"]]
    mask = [0]
    for turn in turns:
        role, text = turn["role"], turn["text"]
        if role in AMICA_KINDS:
            # мисъл или извикване: ролевият токен го слага приложението, а маркерът
            # (<|think|> или <|tool|>) е решение на Амика и се учи
            ids += [bpe.special["<|amica|>"], bpe.special[AMICA_KINDS[role]]]
            mask += [0, 1]
            learn = 1
        else:
            ids.append(bpe.special[ROLE_TOKEN[role]])
            mask.append(0)
            learn = 1 if role == "amica" else 0
        body = bpe.encode(text) + [bpe.special["<|end|>"]]
        ids += body
        mask += [learn] * len(body)
    return ids, mask


def prompt_ids(bpe, turns):
    """Токените, от които Амика генерира отговор: разговорът досега плюс отворена нейна реплика."""
    ids, _ = encode_conversation(bpe, turns)
    return ids + [bpe.special["<|amica|>"]]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    bpe = load_tokenizer()
    print("special tokens:", ", ".join(f"{name}={bpe.special[name]}" for name in SPECIAL))
    print(f"vocab: {len(bpe.vocab)} tokens\n")

    conversation = [
        {"role": "user", "text": "Колко е часът?"},
        {"role": "think", "text": "Питат за часа. Имам инструмент clock."},
        {"role": "tool", "text": "clock"},
        {"role": "result", "text": "14:35"},
        {"role": "amica", "text": "Часът е 14:35."},
    ]
    ids, mask = encode_conversation(bpe, conversation)
    print(f"example: {len(ids)} tokens, {sum(mask)} of them are learned")
    for i, m in zip(ids, mask):
        print(f"  {i:5}  {'learn' if m else '  -  '}  {bpe.decode([i])!r}")
    print("\nprompt for the next reply ends with:", bpe.decode(prompt_ids(bpe, conversation)[-3:]))

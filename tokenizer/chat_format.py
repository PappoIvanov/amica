"""
Амика · етап 5 · форматът на разговора.

Разговорът е документ от реплики. Всяка реплика е служебен токен за ролята, обикновен текст и <|end|>:
  <|doc|> <|system|> указания <|end|> <|user|> въпрос <|end|> <|amica|> отговор <|end|> ...
Репликите на Амика са три вида: мисъл (<|amica|><|think|> кратко разсъждение <|end|>), извикване на
инструмент (<|amica|><|tool|> име и аргументи <|end|>) и отговор (<|amica|> текст <|end|>). Резултатът от
инструмента е реплика <|result|>, вписана от приложението. Генерирането винаги спира на <|end|>.
При дообучаване загубата брои само това, което Амика пише: нейния текст, нейните <|tool|> и <|end|>.
Затова encode_conversation връща и маска: 1 за токените, които се учат, 0 за контекста.

Служебни токени: осемте имена в SPECIAL, в този ред, винаги последните осем номера на речника.
Номерата не са записани в кода, а идват от json-а на токенизатора (от 2026-11-02, за v4):
  речник 8192 (amica_bpe.json, v2 и v3): 8184 <|doc|> ... 8191 <|think|>
  речник 16 384 (amica_bpe_16k.json, v4): 16 376 <|doc|> ... 16 383 <|think|>
Кой токенизатор се ползва го казва checkpoint-ът на модела (полето "tokenizer"), не този файл.
Пускане от папката на проекта: py tokenizer/chat_format.py [amica_bpe_16k.json]
При първото пускане добавя липсващите служебни токени във файла; после само показва примера.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # за да намери bpe.py
from bpe import BPE

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tokenizer" / "amica_bpe.json"          # по подразбиране старият речник; v4 подава своя
SPECIAL = ["<|doc|>", "<|user|>", "<|amica|>", "<|end|>", "<|tool|>", "<|result|>", "<|system|>", "<|think|>"]
ROLE_TOKEN = {"system": "<|system|>", "user": "<|user|>", "amica": "<|amica|>", "result": "<|result|>"}
AMICA_KINDS = {"think": "<|think|>", "tool": "<|tool|>"}   # реплики на Амика с маркер след <|amica|>


def load_tokenizer(path=TOKENIZER):
    """Зарежда токенизатора и гарантира, че осемте служебни токена са в него, в реда на SPECIAL,
    като последните осем номера на речника."""
    bpe = BPE.load(path)
    missing = [name for name in SPECIAL if name not in bpe.special]
    for name in missing:
        bpe.add_special(name)
    if missing:
        bpe.save(path)
        print(f"added {len(missing)} special tokens to {Path(path).name}")
    numbers = [bpe.special[name] for name in SPECIAL]
    first = len(bpe.vocab) - len(SPECIAL)                    # 8184 при речник 8192, 16 376 при 16 384
    assert numbers == list(range(first, first + len(SPECIAL))), bpe.special
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
    path = ROOT / "tokenizer" / sys.argv[1] if len(sys.argv) > 1 else TOKENIZER
    bpe = load_tokenizer(path)
    print(f"tokenizer: {path.name}, vocab: {len(bpe.vocab)} tokens")
    print("special tokens:", ", ".join(f"{name}={bpe.special[name]}" for name in SPECIAL), "\n")

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

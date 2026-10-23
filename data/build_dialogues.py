"""
Амика · етап 5 · от data/raw/razgovori.txt към data/clean/dialogues.jsonl.

Чете ръчните разговори (форматът е описан в главата на razgovori.txt), проверява ги и записва
по един ред JSON на разговор: {"id": "...", "turns": [{"role": "user", "text": "..."}, ...]}
с роли system, user, think, tool, result и amica, точно както ги очаква tokenizer/chat_format.py.
Грешките във формата се изброяват всичките наведнъж, с номера на разговора, и нищо не се записва.

Всеки разговор се разгъва VARIANTS пъти (SEED 42). В копията се сменят две неща:
  1. шаблоните {час}, {дата}, {a}, {b}, {a*b}, {a+b}, {a-b}, {име}, {число}, двойката {m} < {n}
     с {n-m} и {n+m}, {град} и {прогноза}: така единственото общо между копията е „каквото пише
     в реда Резултат:, това казвам“ и Амика се учи да преписва от резултата, не да помни стойност;
  2. системният ред: обявените инструменти плюс до два случайни допълнителни, а разговор без ред
     „Система:“ получава случаен списък (понякога празен, „Инструменти: няма.“). Иначе моделът учи
     „има списък с инструменти, значи питат за час“, както стана при първото дообучаване.
Еднаквите копия се пазят веднъж.
Пускане от папката на проекта: py data/build_dialogues.py        (VARIANTS=200 от средата сменя броя)
"""
import json
import os
import random
import re
import sys
from datetime import date, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "razgovori.txt"
OUT = ROOT / "data" / "clean" / "dialogues.jsonl"
VARIANTS = int(os.environ.get("VARIANTS", 200))
SEED = 42
TOOL_POOL = ["clock", "date", "calc", "remember", "read", "web"]   # откъдето идват случайните инструменти
NO_TOOLS = "Инструменти: няма."

ROLES = {"Система": "system", "Аз": "user", "Мисли": "think",
         "Инструмент": "tool", "Резултат": "result", "Амика": "amica"}
WEEKDAYS = ["понеделник", "вторник", "сряда", "четвъртък", "петък", "събота", "неделя"]
MONTHS = ["януари", "февруари", "март", "април", "май", "юни",
          "юли", "август", "септември", "октомври", "ноември", "декември"]
NAMES = ["Павел", "Мария", "Георги", "Елена", "Иван", "Надежда",
         "Петър", "Анна", "Стефан", "Виктория", "Димитър", "Ива"]
CITIES = ["София", "Пловдив", "Варна", "Бургас", "Лондон", "Париж", "Берлин", "Виена", "Рим", "Мадрид"]
WEATHER = ["слънчево", "облачно", "променливо", "мъгливо", "ветровито"]

FIRST = re.compile(r"^(\d+)\.\s+(\S+):\s*(.*)$")        # „20. Система: ...“, първият ред на разговор
ROLE_LINE = re.compile(r"^\s+([^\s:][^:]*):\s*(.*)$")    # „    Аз: ...“, следваща реплика
PLACEHOLDER = re.compile(r"\{([^{}]+)\}")

problems = []   # всички грешки се събират и се показват накрая


def parse(path):
    """Връща списък от (номер, реплики). Коментарите с // падат, празен ред затваря разговора."""
    conversations, current, number = [], None, None
    for lineno, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.rstrip()
        if line.lstrip().startswith("//"):
            continue
        if not line.strip():
            if current:
                conversations.append((number, current))
                current = None
            continue
        m = FIRST.match(line)
        if m and not line[0].isspace():
            if current:
                conversations.append((number, current))
            number, current = int(m.group(1)), []
            role = m.group(2)
            if role not in ROLES:
                problems.append(f"line {lineno}: unknown role '{role}'")
                role = "Аз"
            current.append({"role": ROLES[role], "text": m.group(3).strip()})
            continue
        if current is None:
            problems.append(f"line {lineno}: expected a conversation start like '20. Аз: ...'")
            continue
        m = ROLE_LINE.match(line)
        if m and m.group(1).strip() in ROLES:
            current.append({"role": ROLES[m.group(1).strip()], "text": m.group(2).strip()})
        else:
            if m and len(m.group(1).split()) <= 3 and m.group(1).strip()[:1].isupper():
                # „Мисли от резултата:“ и подобни: прилича на роля, но не е; казваме го, вместо да го скрием
                problems.append(f"line {lineno}: '{m.group(1).strip()}' is not a role, "
                                f"the line was glued to the previous one")
            current[-1]["text"] += " " + line.strip()   # продължение на предишната реплика
    if current:
        conversations.append((number, current))
    return conversations


def check(number, turns):
    # редът на репликите трябва да е такъв, какъвто го очакват chat_format.py и приложението
    roles = [t["role"] for t in turns]
    if "user" not in roles or "amica" not in roles:
        problems.append(f"conversation {number}: needs at least one 'Аз' and one 'Амика'")
    if "system" in roles[1:]:
        problems.append(f"conversation {number}: 'Система' must be the first line")
    for i, role in enumerate(roles):
        if role == "result" and (i == 0 or roles[i - 1] != "tool"):
            problems.append(f"conversation {number}: 'Резултат' must come right after 'Инструмент'")
        if role == "tool" and (i + 1 >= len(roles) or roles[i + 1] != "result"):
            problems.append(f"conversation {number}: 'Инструмент' must be followed by 'Резултат'")
    for t in turns:
        if not t["text"]:
            problems.append(f"conversation {number}: empty '{t['role']}' line")
        for name in PLACEHOLDER.findall(t["text"]):
            if name not in random_values(random.Random(0)):
                problems.append(f"conversation {number}: unknown placeholder {{{name}}}")


def random_values(rng):
    # една съгласувана порция стойности: a и b са едни и същи навсякъде в разговора
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    m = rng.randint(1, 9)                     # малка двойка за задачи с предмети: m е винаги по-малко от n
    n = rng.randint(m + 1, 12)
    d = date(2026, 1, 1) + timedelta(days=rng.randint(0, 729))
    low = rng.randint(-5, 25)                 # прогноза: диапазон на температурите и дъждовни дни
    forecast = (f"{rng.choice(WEATHER)}, от {low} до {low + rng.randint(3, 10)} градуса, "
                f"дъжд в {rng.randint(0, 14)} от 14 дни")
    return {"час": f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}",
            "дата": f"{WEEKDAYS[d.weekday()]}, {d.day} {MONTHS[d.month - 1]} {d.year}",
            "a": str(a), "b": str(b), "a*b": str(a * b), "a+b": str(a + b), "a-b": str(a - b),
            "m": str(m), "n": str(n), "n-m": str(n - m), "n+m": str(n + m),
            "град": rng.choice(CITIES), "прогноза": forecast,
            "име": rng.choice(NAMES), "число": str(rng.randint(1, 1000))}


def fill(turns, values):
    return [{"role": t["role"], "text": PLACEHOLDER.sub(lambda m: values[m.group(1)], t["text"])} for t in turns]


def with_system(turns, rng):
    # системният ред на това копие: обявените инструменти плюс случайни, или случаен списък
    turns = [dict(t) for t in turns]
    if turns[0]["role"] == "system":
        if turns[0]["text"].strip() == NO_TOOLS:
            return turns
        declared = [x.strip(" .") for x in turns[0]["text"].split(":", 1)[1].split(",")]
        extra = [x for x in TOOL_POOL if x not in declared]
        rng.shuffle(extra)
        tools = declared + extra[:rng.randint(0, 2)]
        rng.shuffle(tools)
        turns[0]["text"] = "Инструменти: " + ", ".join(tools) + "."
        return turns
    pool = TOOL_POOL[:]
    rng.shuffle(pool)
    tools = sorted(pool[:rng.randint(0, 4)])
    line = "Инструменти: " + ", ".join(tools) + "." if tools else NO_TOOLS
    return [{"role": "system", "text": line}] + turns


if __name__ == "__main__":
    conversations = parse(SOURCE)
    for number, turns in conversations:
        check(number, turns)
    if problems:
        print(f"{len(problems)} problems in {SOURCE.name}, nothing written:")
        for p in problems:
            print("  " + p)
        sys.exit(1)

    rng = random.Random(SEED)
    records = []
    for number, turns in conversations:
        seen = set()
        for k in range(VARIANTS):
            copy = with_system(fill(turns, random_values(rng)), rng)
            key = json.dumps(copy, ensure_ascii=False)
            if key in seen:
                continue                                   # еднакво копие, без шаблони и със същия системен ред
            seen.add(key)
            records.append({"id": f"razgovori/{number}/{len(seen):02d}", "turns": copy})

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    roles = {}
    for r in records:
        for t in r["turns"]:
            roles[t["role"]] = roles.get(t["role"], 0) + 1
    templated = sum(1 for _, turns in conversations if any(PLACEHOLDER.search(t["text"]) for t in turns))
    print(f"source: {len(conversations)} conversations, {templated} with placeholders, up to {VARIANTS} variants each")
    print(f"written: {len(records)} records to {OUT.name}")
    print("turns by role: " + ", ".join(f"{k} {v}" for k, v in sorted(roles.items())))
    example = records[1]
    print(f"\nexample {example['id']}:")
    for t in example["turns"]:
        print(f"  {t['role']:7} {t['text']}")

"""
Амика · етап 5 · двигателят на разговора.

Едно място, което знае как се говори с Амика; терминалният чат (app/talk.py) и настолното
приложение са само фасади над него. Двигателят:
  1. държи разговора като списък от реплики {"role", "text"}, както в data/clean/dialogues.jsonl;
  2. генерира една реплика на Амика до <|end|> и казва каква е: мисъл, извикване или отговор;
  3. изпълнява инструментите (clock, date, calc) и вписва резултата като реплика <|result|>;
  4. повтаря, докато Амика даде отговор или свършат опитите (MAX_ROUNDS), за да няма въртене в кръг.
Извадката е с температура, top-k и наказание за повторения, както при llama.cpp: малките
модели обичат да повтарят една и съща дума.
"""
import ast
import datetime
import operator
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tokenizer"))
from transformer import Amica, Config
from chat_format import load_tokenizer, prompt_ids

WEEKDAYS = ["понеделник", "вторник", "сряда", "четвъртък", "петък", "събота", "неделя"]
MONTHS = ["януари", "февруари", "март", "април", "май", "юни",
          "юли", "август", "септември", "октомври", "ноември", "декември"]
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
             ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
             ast.USub: operator.neg, ast.UAdd: operator.pos}


# ---- инструменти: връщат текст в точно вида, който Амика е виждала при обучението ----
def tool_clock(args):
    return datetime.datetime.now().strftime("%H:%M")


def tool_date(args):
    d = datetime.date.today()
    return f"{WEEKDAYS[d.weekday()]}, {d.day} {MONTHS[d.month - 1]} {d.year}"


def tool_calc(args):
    # сметка само с числа и + - * / % ** и скоби; никакъв друг код не се изпълнява
    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError("only arithmetic is allowed")
    try:
        value = evaluate(ast.parse(args.replace(",", "."), mode="eval").body)
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError):
        return "грешка: не мога да сметна това"
    return str(int(value)) if float(value).is_integer() else f"{value:.4f}".rstrip("0")


TOOLS = {"clock": tool_clock, "date": tool_date, "calc": tool_calc}
MAX_ROUNDS = 4   # най-много толкова реплики на Амика (мисли и извиквания) преди отговор


class Engine:
    def __init__(self, checkpoint, temperature=0.4, top_k=20, repetition_penalty=1.3, max_tokens=120):
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.model = Amica(Config(**ckpt["config"]))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.bpe = load_tokenizer()
        self.special = self.bpe.special
        self.temperature, self.top_k = temperature, top_k
        self.repetition_penalty, self.max_tokens = repetition_penalty, max_tokens

    def system_turn(self):
        # същият ред, който стои в razgovori.txt: Амика знае само обявените инструменти
        return {"role": "system", "text": "Инструменти: " + ", ".join(TOOLS) + "."}

    @torch.no_grad()
    def generate_turn(self, turns):
        """Една реплика на Амика. Връща (вид, текст), вид е think, tool или answer."""
        ids = prompt_ids(self.bpe, turns)
        stop = {self.special["<|end|>"], self.special["<|doc|>"]}
        markers = {self.special["<|think|>"]: "think", self.special["<|tool|>"]: "tool"}
        kind, out = "answer", []
        for i in range(self.max_tokens):
            idx = torch.tensor([ids[-self.model.cfg.block_size:]], dtype=torch.long)
            logits, _ = self.model(idx)
            logits = logits[0, -1] / self.temperature
            for t in set(out):                                   # наказание за вече казаното в тази реплика
                logits[t] = logits[t] / self.repetition_penalty if logits[t] > 0 else logits[t] * self.repetition_penalty
            if self.top_k:                                       # само най-вероятните top_k токена остават
                cutoff = torch.topk(logits, self.top_k).values[-1]
                logits[logits < cutoff] = float("-inf")
            token = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
            if token in stop:
                break
            if i == 0 and token in markers:                      # първият токен казва каква е репликата
                kind = markers[token]
                ids.append(token)
                continue
            ids.append(token)
            out.append(token)
        return kind, self.bpe.decode(out).strip()

    def run_tool(self, call):
        name, _, args = call.partition(" ")
        if name not in TOOLS:
            return f"непознат инструмент: {name}"
        return TOOLS[name](args.strip())

    def respond(self, turns):
        """Довършва хода на Амика след реплика на потребителя. Връща новите реплики по ред:
        мисли, извиквания с резултатите им и накрая отговорът."""
        new = []
        for _ in range(MAX_ROUNDS):
            kind, text = self.generate_turn(turns + new)
            if kind == "answer":
                new.append({"role": "amica", "text": text})
                return new
            new.append({"role": kind, "text": text})
            if kind == "tool":
                new.append({"role": "result", "text": self.run_tool(text)})
        new.append({"role": "amica", "text": "Обърках се. Може ли да го кажеш другояче?"})   # предпазител
        return new

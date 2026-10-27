"""
Амика · етап 5 · терминален разговор с роли.
За разлика от app/chat.py (който продължава текст), тук Амика отговаря: разговорът се пази като
реплики, мислите ѝ се показват бледо, извикванията на инструменти се изпълняват от app/engine.py.
Пускане от папката на проекта: py app/talk.py            (по подразбиране checkpoints/amica-v3-sft.pt)
                               py app/talk.py checkpoints/amica-v2-sft-local1.pt
Команди в разговора: празен ред за край, /new за нов разговор, /think за показване или скриване на мислите.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from engine import Engine

sys.stdout.reconfigure(encoding="utf-8")

DIM, RESET = "\033[2m", "\033[0m"   # бледо и обратно, работи в PowerShell и Windows Terminal

checkpoint = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "checkpoints" / "amica-v3-sft.pt"
engine = Engine(checkpoint)
show_thoughts = True
turns = [engine.system_turn()]
print(f"Amica ({checkpoint.name}). Tools: {engine.system_turn()['text']}")
print("Empty line to quit, /new for a new conversation, /think to toggle thoughts.")

while True:
    try:
        text = input("you: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not text:
        break
    if text == "/new":
        turns = [engine.system_turn()]
        print("(new conversation)")
        continue
    if text == "/think":
        show_thoughts = not show_thoughts
        print(f"(thoughts {'shown' if show_thoughts else 'hidden'})")
        continue
    turns.append({"role": "user", "text": text})
    for turn in engine.respond(turns):
        turns.append(turn)
        if turn["role"] == "think" and show_thoughts:
            print(f"{DIM}  thinking: {turn['text']}{RESET}")
        elif turn["role"] == "tool":
            print(f"{DIM}  tool: {turn['text']}{RESET}")
        elif turn["role"] == "result":
            print(f"{DIM}  result: {turn['text']}{RESET}")
        elif turn["role"] == "amica":
            print(f"Amica: {turn['text']}")
print("Bye.")

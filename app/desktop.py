"""
Амика · настолното приложение · стъпка 2: разговорът.

Прозорецът е pywebview: роден прозорец на Windows с WebView2 (Edge) вътре, който показва
app/ui/index.html. Когато му дадем локален файл, pywebview сам пуска малък HTTP сървър с корен
папката на файла, затова style.css и app.js до него се зареждат както в обикновен сайт.
Python и JS си говорят през класа Api: pywebview слага обект window.pywebview.api в страницата
и всеки публичен метод тук (без долна черта) става функция в JS, която връща Promise.
Аргументите и резултатите минават като JSON, затова методите връщат само речници, списъци,
низове и числа. Всяко извикване върви в своя нишка, затова прозорецът не замръзва, докато
Python работи. Обратната посока, Python към JS, е window.evaluate_js: така страницата научава,
че моделът е зареден.
Пускане от папката на проекта: py app/desktop.py
"""
import datetime
import threading
import time
from pathlib import Path

import webview

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "app" / "ui"
HOME = Path.home() / "Documents" / "Amica"      # данните на Амика: памет, разговори, дневници
CHECKPOINT = ROOT / "checkpoints" / "amica-v2-sft.pt"
# кога се сменя поздравът: „Добро утро“ от 3:00, „Добър ден“ от 9:00, „Добър вечер“ от 19:01 до 2:59
MORNING_FROM, DAY_FROM, EVENING_FROM = datetime.time(3, 0), datetime.time(9, 0), datetime.time(19, 1)


class Api:
    """Това вижда JS като window.pywebview.api. Методите без долна черта са видими, останалото не."""

    def __init__(self):
        self.engine = None                  # app/engine.py, зарежда се след отварянето на прозореца
        self.tools = []                     # имената на инструментите, за горния ред
        self.turns = []                     # текущият разговор: реплики {"role", "text"}, както в dialogues.jsonl
        self.lock = threading.Lock()        # една генерация наведнъж: всяко извикване от JS е в своя нишка

    def status(self):
        # заглавието на празния чат, горният ред на прозореца и редът „Memory: N lines“ под полето
        lines = [line for line in read_memory().splitlines() if line.strip()]
        return {"greeting": greeting(), "checkpoint": CHECKPOINT.stem, "checkpoint_found": CHECKPOINT.exists(),
                "memory_lines": len(lines), "home": str(HOME), "model": self._model_info()}

    def new_chat(self):
        with self.lock:
            self.turns = [self.engine.system_turn()] if self.engine else []
        return {"greeting": greeting()}

    def send(self, text):
        """Реплика на потребителя. Връща новите реплики на Амика по ред (мисли, извиквания,
        резултати, отговор) и колко секунди е траял целият ход."""
        if self.engine is None:
            return {"error": "The model is still loading, one moment.", "turns": []}
        with self.lock:
            if not self.turns:
                self.turns = [self.engine.system_turn()]
            self.turns.append({"role": "user", "text": text})
            started = time.perf_counter()
            new = self.engine.respond(self.turns)
            self.turns.extend(new)
            return {"turns": new, "seconds": round(time.perf_counter() - started, 1)}

    def _model_info(self):
        if self.engine is None:
            return None                     # JS показва „loading…“ в чипа, докато моделът се зареди
        params = sum(p.numel() for p in self.engine.model.parameters())
        return {"params": f"{params / 1e6:.0f}M", "tools": self.tools}


def read_memory():
    path = HOME / "memory.txt"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def user_name():
    # ред „Име: Павел“ в memory.txt; без такъв ред поздравът е без име
    for line in read_memory().splitlines():
        if line.lower().startswith("име:"):
            return line.split(":", 1)[1].strip()
    return ""


def greeting(now=None):
    # три поздрава по часа; „добър следобед“ не се казва на български, а „лека вечер“ е за сбогуване,
    # затова и нощта е „Добър вечер“. Времето се подава отвън само за проверка, иначе е сегашното.
    now = datetime.datetime.now().time() if now is None else now
    if MORNING_FROM <= now < DAY_FROM:
        words = "Добро утро"
    elif DAY_FROM <= now < EVENING_FROM:
        words = "Добър ден"
    else:
        words = "Добър вечер"
    name = user_name()
    return f"{words}, {name}." if name else f"{words}."


def prepare_home():
    # при първо пускане: папката в Documents и празен memory.txt, за да има какво да отвори Notepad
    HOME.mkdir(parents=True, exist_ok=True)
    (HOME / "memory.txt").touch()


def load_engine(api, window):
    # върви в отделна нишка след отварянето на прозореца, за да се появи той веднага; torch се внася
    # чак тук, защото самото внасяне трае секунда-две и стартовата анимация покрива чакането
    from engine import Engine, TOOLS
    started = time.perf_counter()
    api.engine = Engine(CHECKPOINT)
    api.tools = list(TOOLS)
    api.turns = [api.engine.system_turn()]
    print(f"Model loaded in {time.perf_counter() - started:.1f} s: {api.status()['model']}")
    window.evaluate_js("modelReady()")      # Python → JS: страницата обновява чипа и отключва полето


if __name__ == "__main__":
    prepare_home()
    print(f"Amica data folder: {HOME}")
    api = Api()
    window = webview.create_window("Amica", str(UI / "index.html"), js_api=api,
                                   width=1280, height=820, min_size=(960, 640),
                                   background_color="#0B0B0C")
    webview.start(load_engine, (api, window))

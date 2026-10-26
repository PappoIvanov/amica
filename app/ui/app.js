// Амика · стъпка 2: разговорът в прозореца.
// Всичко, което иска Python, минава през window.pywebview.api.<метод>(...) и връща Promise.

const $ = (id) => document.getElementById(id);
const messages = $("messages"), input = $("input");
let busy = false, modelLoaded = false;

async function showStatus() {
  const s = await window.pywebview.api.status();        // Python → JSON → обект тук
  $("greeting").textContent = s.greeting;               // „Добро утро, Павел.“ според часа
  modelLoaded = Boolean(s.model);
  $("model-chip").textContent = s.checkpoint + (s.checkpoint_found ? "" : " · not found")
                              + (s.model ? " · " + s.model.params : " · loading…");
  $("tools-line").textContent = s.model ? "Tools: " + s.model.tools.join(", ") : "";
  $("memory-hint").textContent = `Memory: ${s.memory_lines} lines loaded from memory.txt`;
}

// Python вика това през window.evaluate_js, когато моделът е зареден в своята нишка
function modelReady() { showStatus(); }

// pywebview пуска събитието, когато обектът api е готов; ако вече е там, викаме направо
if (window.pywebview && window.pywebview.api) showStatus();
else window.addEventListener("pywebviewready", showStatus);

// стартовата анимация: класът play пуска хореографията от style.css (около 3 секунди, веднъж);
// махаме го и го слагаме пак, за да тръгне отначало и при нов чат
function playIntro() {
  const welcome = $("welcome");
  welcome.classList.remove("play");
  void welcome.offsetWidth;                              // принуждава преизчисляване, иначе браузърът слива двете промени
  welcome.classList.add("play");
}
playIntro();

// --- рисуване на репликите ---
const esc = (text) => text.replace(/[&<>]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c]));
const logoHTML = '<div class="logo"><div class="rim"></div><div class="disc"><span class="am">AM</span></div><div class="gloss"></div></div>';

function add(kind, html) {
  const el = document.createElement("div");
  el.className = "msg " + kind;
  el.innerHTML = html;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

function render(turns, seconds) {
  // репликите идват по ред: think, tool, result, ..., amica; tool и result стават една карта
  for (let i = 0; i < turns.length; i++) {
    const t = turns[i];
    if (t.role === "think") {
      add("think", "Thinking · " + esc(t.text));
    } else if (t.role === "tool") {
      const next = turns[i + 1];
      const result = next && next.role === "result" ? next.text : "";
      const [name, ...args] = t.text.split(" ");
      add("tool", `<b>${esc(name)}</b> ${esc(args.join(" "))} <i>→</i> ${esc(result)}`);
      if (result) i++;                                     // резултатът е нарисуван в картата
    } else if (t.role === "amica") {
      add("amica", logoHTML + `<div class="text">${esc(t.text)}<small>${seconds} s</small></div>`);
    }
  }
}

async function send(text) {
  text = text.trim();
  if (!text || busy) return;
  if (!modelLoaded) { add("error", "The model is still loading, one moment."); return; }
  busy = true; $("send").disabled = true;
  input.value = ""; input.style.height = "auto";
  $("welcome").hidden = true;                            // празният екран изчезва с първото съобщение
  if ($("chat-title").textContent === "New chat") $("chat-title").textContent = text.length > 40 ? text.slice(0, 40) + "…" : text;
  add("user", `<span class="bubble">${esc(text)}</span>`);
  const wait = add("wait", "Amica is thinking<span>.</span><span>.</span><span>.</span>");
  try {
    const reply = await window.pywebview.api.send(text);  // чака в Python, прозорецът не замръзва
    wait.remove();
    if (reply.error) add("error", reply.error);
    else render(reply.turns, reply.seconds);
  } catch (err) {                                          // грешка в Python = отхвърлен Promise
    wait.remove();
    add("error", "Python error: " + (err && err.message ? err.message : err));
  }
  busy = false; $("send").disabled = false; input.focus();
}

async function newChat() {
  if (busy) return;
  const r = await window.pywebview.api.new_chat();
  messages.querySelectorAll(".msg").forEach((m) => m.remove());
  $("chat-title").textContent = "New chat";
  $("greeting").textContent = r.greeting;
  $("welcome").hidden = false;
  playIntro();
  input.focus();
}

// Enter изпраща, Shift+Enter е нов ред; полето расте с текста до 200px
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input.value); }
});
input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 200) + "px"; });
$("send").addEventListener("click", () => send(input.value));
$("new-chat").addEventListener("click", newChat);
document.querySelectorAll(".suggestion").forEach((button) => button.addEventListener("click", () => send(button.dataset.text)));

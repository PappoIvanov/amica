# Прави иконата на Амика: вариант 1 (пръстен с AM), четкан метал, златиста палитра, Inter ExtraBold.
# Рисува на 1024 px с numpy + Pillow по същата геометрия като SVG образеца (viewBox 128), смалява до 7 размера, пише .ico и PNG.
import re, sys, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.stdout.reconfigure(encoding="utf-8")

OUT, FONT_DIR = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True); FONT_DIR.mkdir(parents=True, exist_ok=True)

# ---- шрифт: Inter 800 от Google Fonts (лиценз OFL); със стар User-Agent сървърът дава TTF, не woff2 ----
font_path = FONT_DIR / "Inter-800.ttf"
if not font_path.exists():
    req = urllib.request.Request("https://fonts.googleapis.com/css2?family=Inter:wght@800", headers={"User-Agent": "Mozilla/4.0"})
    css = urllib.request.urlopen(req, timeout=30).read().decode()
    m = re.search(r"url\((https://fonts\.gstatic\.com/[^)]+)\)", css)
    if not m or not m.group(1).endswith((".ttf", ".otf")):
        raise SystemExit("no ttf url in css:\n" + css)
    font_path.write_bytes(urllib.request.urlopen(m.group(1), timeout=60).read())
    print("font:", m.group(1), font_path.stat().st_size, "bytes")

# ---- платно: 1024 px = 128 единици от viewBox × 8 ----
S = 1024; K = S / 128
yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) + 0.5
X, Y = xx / K, yy / K                                   # координати в единици на viewBox

def rgb(h): return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)], np.float32)

def gradient(t, stops):                                 # t: (S,S) в [0,1]; stops: [(offset, '#hex'), ...]
    t = np.clip(t, 0, 1); out = np.empty(t.shape + (3,), np.float32); out[:] = rgb(stops[0][1])
    for (o0, c0), (o1, c1) in zip(stops, stops[1:]):
        m = (t >= o0) & (t <= o1)
        f = np.zeros_like(t) if o1 == o0 else np.clip((t - o0) / (o1 - o0), 0, 1)
        seg = rgb(c0) * (1 - f)[..., None] + rgb(c1) * f[..., None]
        out[m] = seg[m]
    out[t > stops[-1][0]] = rgb(stops[-1][1])
    return out

def shape(draw_fn):                                     # маска 0..1 от рисунка с ImageDraw
    im = Image.new("L", (S, S), 0); draw_fn(ImageDraw.Draw(im)); return np.asarray(im, np.float32) / 255

def disk(cx, cy, r, width=0):                           # width 0 = пълен кръг; иначе контур с дебелина width (единици), r е външният радиус
    b = [(cx - r) * K, (cy - r) * K, (cx + r) * K, (cy + r) * K]
    return shape(lambda d: d.ellipse(b, fill=255) if not width else d.ellipse(b, outline=255, width=max(1, round(width * K))))

canvas = np.zeros((S, S, 4), np.float32)
def over(color, alpha):                                 # наслагване „over“: color (3,) или (S,S,3), alpha (S,S)
    a = alpha[..., None]; c = np.broadcast_to(np.asarray(color, np.float32), (S, S, 3))
    ca = canvas[..., 3:4]; out_a = a + ca * (1 - a)
    canvas[..., :3] = (c * a + canvas[..., :3] * ca * (1 - a)) / np.maximum(out_a, 1e-6)
    canvas[..., 3:4] = out_a

RIM = [(0, '#FFE4C2'), (.18, '#F28C28'), (.46, '#7A3A0A'), (.62, '#E08A2E'), (.8, '#F7B46A'), (1, '#5E2B06')]
CHROME = [(0, '#FFF6E8'), (.34, '#FFD9AB'), (.5, '#F28C28'), (.52, '#8A3F0A'), (.62, '#C56912'), (.82, '#F7A64A'), (1, '#FFE6C6')]
GLASS = [(0, '#2C2C32'), (.38, '#131316'), (.72, '#000000')]
WHITE = rgb('#FFFFFF')
brush = ((Y % 2) < 1).astype(np.float32)               # четкани линии: 1 единица светло на всеки 2

# пръстен: диагонален метален градиент, четкани линии, отблясък горе вляво
ring = disk(64, 64, 60)
over(gradient(((X - 4) / 120 + (Y - 4) / 120) / 2, RIM), ring)
over(WHITE, ring * brush * .14)
def arc_fn(d):
    r, w = 57, 2.5; R = r + w / 2                        # Pillow рисува дъгата навътре от рамката, затова рамката е r + w/2
    d.arc([(64 - R) * K, (64 - R) * K, (64 + R) * K, (64 + R) * K], 160, 250, fill=255, width=round(w * K))
    for px, py in ((10.4, 83.5), (44.5, 10.4)):          # кръгли краища
        d.ellipse([(px - w / 2) * K, (py - w / 2) * K, (px + w / 2) * K, (py + w / 2) * K], fill=255)
over(WHITE, shape(arc_fn) * .5)

# черно стъкло с радиален отблясък и тъмен ръб
over(gradient(np.hypot(X - 51.04, Y - 38.08) / 86.4, GLASS), disk(64, 64, 54))
over(rgb('#3A1A06'), disk(64, 64, 54.5, width=1))

# буквите AM: сянка, хром, четкани линии, диагонален лъч
font = ImageFont.truetype(str(font_path), round(52 * K))
try:
    font.set_variation_by_axes([800])                   # ако шрифтът е променлив; при статичен хвърля грешка и я прескачаме
except Exception:
    pass
wA, wM, ls = font.getlength("A"), font.getlength("M"), -1 * K
x0, base = S / 2 - (wA + wM + ls) / 2, 82 * K
def text_fn(d, dx=0, dy=0):
    d.text((x0 + dx, base + dy), "A", font=font, fill=255, anchor="ls")
    d.text((x0 + wA + ls + dx, base + dy), "M", font=font, fill=255, anchor="ls")
txt = shape(text_fn)
over(rgb('#000000'), shape(lambda d: text_fn(d, 1.2 * K, 1.8 * K)) * .55)
ys, xs = np.nonzero(txt > .5)
ty0, ty1, tx0, tx1 = ys.min() / K, ys.max() / K, xs.min() / K, xs.max() / K
over(gradient((Y - ty0) / (ty1 - ty0), CHROME), txt)
over(WHITE, txt * brush * .14)
tt = ((X - tx0) / (tx1 - tx0) + (Y - ty0) / (ty1 - ty0)) / 2
over(WHITE, txt * np.clip(1 - np.abs(tt - .5) / .1, 0, 1) * .6)

# горен отблясък на стъклото
over(WHITE, shape(lambda d: d.ellipse([28 * K, 15 * K, 100 * K, 45 * K], fill=255)) * np.clip(.28 * (1 - (Y - 15) / 30 / .92), 0, None))

# ---- смаляване с предварително умножена алфа (без тъмни ръбове) и запис ----
def to_size(n):
    pm = np.concatenate([canvas[..., :3] * canvas[..., 3:4], canvas[..., 3:4]], axis=-1)
    im = Image.fromarray((pm * 255 + .5).astype(np.uint8), "RGBA").resize((n, n), Image.LANCZOS)
    arr = np.asarray(im).astype(np.float32) / 255
    a = arr[..., 3:4]; col = np.clip(arr[..., :3] / np.maximum(a, 1e-6), 0, 1) * (a > 0)
    return Image.fromarray((np.concatenate([col, a], -1) * 255 + .5).astype(np.uint8), "RGBA")

sizes = [256, 128, 64, 48, 32, 24, 16]
ims = {n: to_size(n) for n in sizes}
ims[256].save(OUT / "amica.ico", format="ICO", sizes=[(n, n) for n in sizes], append_images=[ims[n] for n in sizes[1:]])
ims[256].save(OUT / "amica-256.png")
ims[48].save(OUT / "amica-48.png")

# преглед: тъмна и светла лента с 128, 48, 32 и 16 px
prev = Image.new("RGBA", (680, 320), (0, 0, 0, 0))
d = ImageDraw.Draw(prev)
for row, bg in enumerate(("#0B0B0C", "#F2F2F2")):
    d.rectangle([0, row * 160, 680, row * 160 + 160], fill=bg)
    x = 40
    for n in (128, 48, 32, 16):
        prev.alpha_composite(ims[n], (x, row * 160 + 80 - n // 2)); x += n + 60
prev.save(OUT / "amica-preview.png")

chk = Image.open(OUT / "amica.ico")
print("ico sizes:", sorted(chk.info.get("sizes", []), reverse=True), " bytes:", (OUT / "amica.ico").stat().st_size)
print("text bbox (units): x", round(tx0, 1), "-", round(tx1, 1), " y", round(ty0, 1), "-", round(ty1, 1))

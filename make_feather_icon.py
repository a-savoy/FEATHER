"""Draw a stylized zebra-finch feather and save it as PNG + ICO for F.E.A.T.H.E.R.

Zebra-finch cues: bold black-and-white barring (the tail's "zebra" pattern), a warm
chestnut/orange base (flank + cheek) with white spots (flank spots), cream shaft.
"""
import math
import os
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
S = 512
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

cx = S / 2
top, bot = 72, 470
L = bot - top
maxw = 122
BLACK = (38, 34, 30, 255)
WHITE = (250, 248, 240, 255)
ORANGE = (236, 120, 38, 255)
CHEST = (150, 82, 46, 255)
CREAM = (228, 208, 152, 255)
nband = 7
bandh = L / nband


def xc(t):
    return cx + 20 * math.sin(math.pi * t)          # gentle curve to the shaft


def hw(t):
    return maxw * (math.sin(math.pi * t) ** 0.72)    # taper: 0 at tip/base, wide mid


# barbs (colored per horizontal band -> zebra bars; chestnut/orange base)
nb = 900
for i in range(nb):
    t = i / (nb - 1)
    y = top + t * L
    x0, w = xc(t), hw(t)
    if t > 0.90:
        col = ORANGE
    elif t > 0.78:
        col = CHEST
    else:
        col = BLACK if int((y - top) / bandh) % 2 == 0 else WHITE
    for s in (-1, 1):
        d.line([(x0, y), (x0 + s * w, y - w * 0.5)], fill=col, width=3)

# white flank spots on the chestnut/orange base
for dt, fx in [(0.82, -0.42), (0.85, 0.36), (0.885, -0.28), (0.915, 0.30), (0.8, 0.12)]:
    y, w, x = top + dt * L, hw(dt), xc(dt)
    r = 6
    d.ellipse([x + fx * w - r, y - r, x + fx * w + r, y + r], fill=WHITE)

# shaft (rachis) following the curve, and the quill at the base
d.line([(xc(i / 60), top + (i / 60) * L) for i in range(61)], fill=CREAM, width=5, joint="curve")
d.line([(xc(1.0), bot), (xc(1.0), bot + 30)], fill=CREAM, width=7)

img = img.filter(ImageFilter.SMOOTH_MORE)
img.save(os.path.join(HERE, "feather_512.png"))
img.resize((256, 256), Image.LANCZOS).save(os.path.join(HERE, "feather_256.png"))
img.resize((72, 72), Image.LANCZOS).save(os.path.join(HERE, "feather_72.png"))
img.resize((256, 256), Image.LANCZOS).save(
    os.path.join(HERE, "feather.ico"),
    sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("saved feather_512.png, feather_256.png, feather_72.png, feather.ico")

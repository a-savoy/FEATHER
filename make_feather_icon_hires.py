"""make_feather_icon_hires.py — render the FEATHER feather at print resolution.

Same drawing as make_feather_icon.py, but every coordinate, line width and the
barb count scale with the output size, so a big render is genuinely sharp rather
than an upscaled 512 px icon. Internally it draws at 2x and downsamples
(supersampling) for clean edges, and writes the DPI into the PNG.

    python make_feather_icon_hires.py                  # 2 in @ 600 dpi -> 1200 px
    python make_feather_icon_hires.py --inches 4       # 4 in @ 600 dpi -> 2400 px
    python make_feather_icon_hires.py --px 600         # exact pixel size
"""
import argparse
import math
import os
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))

BLACK = (38, 34, 30, 255)
WHITE = (250, 248, 240, 255)
ORANGE = (236, 120, 38, 255)
CHEST = (150, 82, 46, 255)
CREAM = (228, 208, 152, 255)


def draw_feather(S):
    """Draw the feather on an SxS transparent canvas. S=512 reproduces the icon."""
    k = S / 512.0                       # scale factor from the original design
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    cx = S / 2
    top, bot = 72 * k, 470 * k
    L = bot - top
    maxw = 122 * k
    nband = 7
    bandh = L / nband

    def xc(t):
        return cx + 20 * k * math.sin(math.pi * t)        # gentle curve to the shaft

    def hw(t):
        return maxw * (math.sin(math.pi * t) ** 0.72)     # taper: 0 at tip/base, wide mid

    # barbs (colored per horizontal band -> zebra bars; chestnut/orange base).
    # The barb count scales with size so the vane stays solid at high resolution.
    nb = max(900, int(round(900 * k)))
    bw = max(1, int(round(3 * k)))
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
            d.line([(x0, y), (x0 + s * w, y - w * 0.5)], fill=col, width=bw)

    # white flank spots on the chestnut/orange base
    for dt, fx in [(0.82, -0.42), (0.85, 0.36), (0.885, -0.28), (0.915, 0.30), (0.8, 0.12)]:
        y, w, x = top + dt * L, hw(dt), xc(dt)
        r = 6 * k
        d.ellipse([x + fx * w - r, y - r, x + fx * w + r, y + r], fill=WHITE)

    # shaft (rachis) following the curve, and the quill at the base
    d.line([(xc(i / 60), top + (i / 60) * L) for i in range(61)],
           fill=CREAM, width=max(1, int(round(5 * k))), joint="curve")
    d.line([(xc(1.0), bot), (xc(1.0), bot + 30 * k)],
           fill=CREAM, width=max(1, int(round(7 * k))))
    return img


def render(px, supersample=2):
    """Render at `px` square, drawing at `supersample`x and downsampling for clean edges."""
    img = draw_feather(px * supersample).filter(ImageFilter.SMOOTH_MORE)
    if supersample > 1:
        img = img.resize((px, px), Image.LANCZOS)
    return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dpi", type=float, default=600, help="dots per inch (default 600)")
    ap.add_argument("--inches", type=float, default=2.0,
                    help="printed size of the square icon, in inches (default 2)")
    ap.add_argument("--px", type=int, default=None,
                    help="exact pixel size; overrides --inches")
    ap.add_argument("-o", "--out", default=None, help="output PNG path")
    a = ap.parse_args()

    px = a.px if a.px else int(round(a.inches * a.dpi))
    out = a.out or os.path.join(HERE, f"feather_{px}px_{int(a.dpi)}dpi.png")
    render(px).save(out, dpi=(a.dpi, a.dpi))
    print(f"saved {out}  ({px}x{px} px = {px / a.dpi:.2f} in at {a.dpi:g} dpi)")

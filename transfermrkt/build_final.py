"""Build the final LinkedIn post: a wide cinematic photo band (zoomed onto the
three teammates) with a subtle reddish duotone wash, over the market-value chart.

Tune the look with TINT (0 = natural photo, 1 = full red/charcoal duotone).

Run from the repo root:
    .venv/bin/python transfermrkt/build_final.py
"""
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance

import line_charts as lc

BG = (250, 250, 248)
HERO = (229, 52, 43)
WHITE = (255, 255, 255)
CANVAS_W, CANVAS_H = 2000, 1750
TINT = 0.25          # strength of the reddish duotone wash (subtle)
PHOTO = 'transfermrkt/IMG_3664.JPG'
FONT_DIR = 'fonts'
_FONTS = {
    'regular': 'Montserrat-Regular.ttf', 'medium': 'Montserrat-Medium.ttf',
    'semibold': 'Montserrat-SemiBold.ttf', 'bold': 'Montserrat-Bold.ttf',
    'xbold': 'Montserrat-ExtraBold.ttf',
}


def F(weight, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, _FONTS[weight]), size)


def wide_crop(img, tw, th, top_frac=0.02):
    """Full-width crop that keeps the top of the frame (faces), then resizes."""
    w, h = img.size
    nh = int(w * th / tw)
    y0 = min(int(h * top_frac), h - nh)
    return img.crop((0, y0, w, y0 + nh)).resize((tw, th), Image.Resampling.LANCZOS)


def bottom_gradient(size, strength=215, start=0.28):
    w, h = size
    grad = Image.new('L', (1, h), 0)
    for y in range(h):
        t = max(0.0, (y / h - start) / (1 - start))
        grad.putpixel((0, y), int(strength * t * t))
    black = Image.new('RGBA', (w, h), (0, 0, 0, 255))
    black.putalpha(grad.resize((w, h)))
    return black


# --- shared chart --------------------------------------------------------------
lc.build_chart('B', lc.THEMES['B'], 'transfermrkt/_final_base.png',
               features={'passport'})
chart = Image.open('transfermrkt/_final_base.png').convert('RGB')
photo = Image.open(PHOTO).convert('RGB')

canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), BG)

# --- cinematic photo band with a subtle reddish wash ---------------------------
band_h = 720
base = wide_crop(photo, CANVAS_W, band_h, top_frac=0.10)
gray = ImageOps.grayscale(base)
duo = ImageOps.colorize(gray, black=(28, 20, 20), mid=(150, 60, 55),
                        white=(250, 236, 233))
band = Image.blend(base, duo, TINT)                 # mostly natural, gently red
band = ImageEnhance.Color(band).enhance(1.05)
band = band.convert('RGBA')
band.alpha_composite(bottom_gradient((CANVAS_W, band_h)))

bd = ImageDraw.Draw(band)
bd.rounded_rectangle([70, band_h - 268, 70 + 120, band_h - 256], radius=6, fill=HERO)
bd.text((70, band_h - 232), 'Same pitch.', font=F('xbold', 74), fill=WHITE)
bd.text((70, band_h - 146), 'Different paths.', font=F('xbold', 74), fill=WHITE)
bd.text((72, band_h - 52), 'Lima, 2015   \u2192   2026', font=F('semibold', 38),
        fill=(235, 235, 235))
canvas.paste(band, (0, 0), band)

# --- chart below: fill the remaining canvas ------------------------------------
avail_h = CANVAS_H - band_h - 20
cw = CANVAS_W - 30
ch = int(chart.height * cw / chart.width)
if ch > avail_h:
    ch = avail_h
    cw = int(chart.width * ch / chart.height)
chart = chart.resize((cw, ch), Image.Resampling.LANCZOS)
cx = (CANVAS_W - cw) // 2
cy = band_h + (CANVAS_H - band_h - ch) // 2
canvas.paste(chart, (cx, cy))

canvas.save('transfermrkt/linkedin_post.png')
print(f'Saved -> transfermrkt/linkedin_post.png (TINT={TINT})')

try:
    os.remove('transfermrkt/_final_base.png')
except OSError:
    pass

# .venv/bin/python transfermrkt/build_final.py
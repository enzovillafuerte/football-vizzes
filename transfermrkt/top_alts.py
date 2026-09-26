"""Preview three alternative treatments for the top photo band, each composited
above the same market-value chart. Outputs:
    linkedin_opt1.png  - cinematic wide crop (full-bleed, zoomed)
    linkedin_opt2.png  - split panel (photo left, branded headline panel right)
    linkedin_opt3.png  - duotone wide crop (branded red/charcoal)

Run from the repo root:
    .venv/bin/python transfermrkt/top_alts.py
"""
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps

import line_charts as lc

BG = (250, 250, 248)
TEXT = (34, 34, 34)
MUTED = (138, 143, 148)
HERO = (229, 52, 43)
WHITE = (255, 255, 255)
CHARCOAL = (26, 26, 28)
CANVAS_W, CANVAS_H = 2000, 1750
PHOTO = 'transfermrkt/IMG_3664.JPG'
FONT_DIR = 'fonts'
_FONTS = {
    'regular': 'Montserrat-Regular.ttf', 'medium': 'Montserrat-Medium.ttf',
    'semibold': 'Montserrat-SemiBold.ttf', 'bold': 'Montserrat-Bold.ttf',
    'xbold': 'Montserrat-ExtraBold.ttf',
}


def F(weight, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, _FONTS[weight]), size)


def wide_crop(img, tw, th, top_frac=0.03):
    """Full-width crop that keeps the top of the frame (faces), trimming a little
    sky/roof at the very top, then resizes to (tw, th)."""
    w, h = img.size
    nh = int(w * th / tw)
    y0 = int(h * top_frac)
    y0 = min(y0, h - nh)
    return img.crop((0, y0, w, y0 + nh)).resize((tw, th), Image.Resampling.LANCZOS)


def crop_aspect(img, tw, th, ha='center', va='center'):
    w, h = img.size
    target = tw / th
    if w / h > target:
        nw = int(h * target)
        x0 = {'left': 0, 'center': (w - nw) // 2, 'right': w - nw}[ha]
        box = (x0, 0, x0 + nw, h)
    else:
        nh = int(w / target)
        y0 = {'top': 0, 'center': (h - nh) // 2, 'bottom': h - nh}[va]
        box = (0, y0, w, y0 + nh)
    return img.crop(box).resize((tw, th), Image.Resampling.LANCZOS)


def bottom_gradient(size, strength=225, start=0.28):
    w, h = size
    grad = Image.new('L', (1, h), 0)
    for y in range(h):
        t = max(0.0, (y / h - start) / (1 - start))
        grad.putpixel((0, y), int(strength * t * t))
    black = Image.new('RGBA', (w, h), (0, 0, 0, 255))
    black.putalpha(grad.resize((w, h)))
    return black


def rounded(img, radius):
    mask = Image.new('L', img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, img.size[0] - 1, img.size[1] - 1], radius=radius, fill=255)
    out = Image.new('RGBA', img.size, (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def title_overlay(draw, band_h, color=WHITE, sub=(235, 235, 235)):
    draw.rounded_rectangle([70, band_h - 268, 70 + 120, band_h - 256], radius=6, fill=HERO)
    draw.text((70, band_h - 232), 'From the same pitch', font=F('xbold', 74), fill=color)
    draw.text((70, band_h - 146), 'to the world.', font=F('xbold', 74), fill=color)
    draw.text((72, band_h - 52), 'Lima, 2015   \u2192   2026', font=F('semibold', 38),
              fill=sub)


# --- shared chart --------------------------------------------------------------
lc.build_chart('B', lc.THEMES['B'], 'transfermrkt/_final_base.png',
               features={'passport'})
CHART = Image.open('transfermrkt/_final_base.png').convert('RGB')
PH = Image.open(PHOTO).convert('RGB')


def place_chart(canvas, band_h):
    avail_h = CANVAS_H - band_h - 20
    cw = CANVAS_W - 30
    ch = int(CHART.height * cw / CHART.width)
    if ch > avail_h:
        ch = avail_h
        cw = int(CHART.width * ch / CHART.height)
    ch_img = CHART.resize((cw, ch), Image.Resampling.LANCZOS)
    cx = (CANVAS_W - cw) // 2
    cy = band_h + (CANVAS_H - band_h - ch) // 2
    canvas.paste(ch_img, (cx, cy))


# ============================================================ OPTION 1
def option1():
    band_h = 720
    canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), BG)
    band = wide_crop(PH, CANVAS_W, band_h, top_frac=0.02).convert('RGBA')
    band.alpha_composite(bottom_gradient((CANVAS_W, band_h)))
    title_overlay(ImageDraw.Draw(band), band_h)
    canvas.paste(band, (0, 0), band)
    place_chart(canvas, band_h)
    canvas.save('transfermrkt/linkedin_opt1.png')
    print('Saved -> transfermrkt/linkedin_opt1.png')


# ============================================================ OPTION 2
def option2():
    band_h = 760
    canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), BG)
    # right branded panel
    ImageDraw.Draw(canvas).rectangle([0, 0, CANVAS_W, band_h], fill=CHARCOAL)
    # left photo (tight, full-height, keep faces + watermark side)
    ph_w = 940
    photo = crop_aspect(PH, ph_w, band_h, ha='center').convert('RGBA')
    canvas.paste(photo, (0, 0), photo)
    # headline on the charcoal panel
    d = ImageDraw.Draw(canvas)
    tx = ph_w + 80
    d.rounded_rectangle([tx, 150, tx + 120, 162], radius=6, fill=HERO)
    d.text((tx, 196), 'From the same pitch', font=F('xbold', 70), fill=WHITE)
    d.text((tx, 278), 'to the world.', font=F('xbold', 70), fill=WHITE)
    d.text((tx, 392), 'Three teammates. One pitch in Lima, 2015 \u2014',
           font=F('medium', 30), fill=(210, 210, 212))
    d.text((tx, 432), 'eleven years and several countries later.',
           font=F('medium', 30), fill=(210, 210, 212))
    d.text((tx, 520), 'Fernando  \u00b7  ', font=F('semibold', 32), fill=(170, 172, 176))
    w1 = d.textlength('Fernando  \u00b7  ', font=F('semibold', 32))
    d.text((tx + w1, 520), 'Enzo', font=F('bold', 32), fill=HERO)
    w2 = d.textlength('Enzo', font=F('bold', 32))
    d.text((tx + w1 + w2, 520), '  \u00b7  Franco', font=F('semibold', 32),
           fill=(170, 172, 176))
    place_chart(canvas, band_h)
    canvas.save('transfermrkt/linkedin_opt2.png')
    print('Saved -> transfermrkt/linkedin_opt2.png')


# ============================================================ OPTION 3
def option3():
    band_h = 720
    canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), BG)
    base = wide_crop(PH, CANVAS_W, band_h, top_frac=0.02)
    gray = ImageOps.grayscale(base)
    duo = ImageOps.colorize(gray, black=(22, 14, 14), mid=(140, 38, 33),
                            white=(250, 228, 224)).convert('RGBA')
    duo.alpha_composite(bottom_gradient((CANVAS_W, band_h), strength=200))
    title_overlay(ImageDraw.Draw(duo), band_h)
    canvas.paste(duo, (0, 0), duo)
    place_chart(canvas, band_h)
    canvas.save('transfermrkt/linkedin_opt3.png')
    print('Saved -> transfermrkt/linkedin_opt3.png')


option1()
option2()
option3()

try:
    os.remove('transfermrkt/_final_base.png')
except OSError:
    pass

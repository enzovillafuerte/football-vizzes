"""Generate LinkedIn portrait (4:5) mockups showing how the 2015 group photo
could be integrated into the market-value chart. Produces four variants:

  B - Origin inset + 2015 star (photo lives inside the plot, near the shared origin)
  A - Header hero band (wide photo strip across the top + title overlay)
  E - Diptych "11 years later" (photo on top, chart below)
  C - Left origin card (photo + copy in a card, chart below)

Run from the repo root:
    .venv/bin/python transfermrkt/photo_mockups.py
"""
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import line_charts as lc  # reuses build_chart / THEMES (regenerates line_charts.png)

# ---- palette (mirrors THEME B) ---------------------------------------------
BG = (250, 250, 248)
TEXT = (34, 34, 34)
MUTED = (138, 143, 148)
HERO = (229, 52, 43)
WHITE = (255, 255, 255)

PHOTO = 'transfermrkt/IMG_3664.JPG'
FONT_DIR = 'fonts'
CANVAS_W, CANVAS_H = 1600, 2000  # 4:5 portrait

_FONTS = {
    'regular': 'Montserrat-Regular.ttf', 'medium': 'Montserrat-Medium.ttf',
    'semibold': 'Montserrat-SemiBold.ttf', 'bold': 'Montserrat-Bold.ttf',
    'xbold': 'Montserrat-ExtraBold.ttf',
}


def F(weight, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, _FONTS[weight]), size)


def crop_aspect(img, tw, th, ha='center', va='center'):
    """Crop img to the tw:th ratio (anchored) then resize to (tw, th)."""
    w, h = img.size
    target = tw / th
    if w / h > target:                      # too wide -> crop width
        nw = int(h * target)
        x0 = {'left': 0, 'center': (w - nw) // 2, 'right': w - nw}[ha]
        box = (x0, 0, x0 + nw, h)
    else:                                    # too tall -> crop height
        nh = int(w / target)
        y0 = {'top': 0, 'center': (h - nh) // 2, 'bottom': h - nh}[va]
        box = (0, y0, w, y0 + nh)
    return img.crop(box).resize((tw, th), Image.Resampling.LANCZOS)


def rounded(img, radius):
    mask = Image.new('L', img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, img.size[0] - 1, img.size[1] - 1], radius=radius, fill=255)
    out = Image.new('RGBA', img.size, (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def bottom_gradient(size, strength=210):
    """Transparent-to-dark vertical gradient (dark at the bottom)."""
    w, h = size
    grad = Image.new('L', (1, h), 0)
    for y in range(h):
        t = max(0.0, (y / h - 0.35) / 0.65)
        grad.putpixel((0, y), int(strength * t * t))
    alpha = grad.resize((w, h))
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    overlay.putalpha(alpha)
    black = Image.new('RGBA', (w, h), (0, 0, 0, 255))
    black.putalpha(alpha)
    return black


def wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ''
    for wd in words:
        trial = (cur + ' ' + wd).strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def scale_to_w(img, w):
    r = w / img.width
    return img.resize((w, int(img.height * r)), Image.Resampling.LANCZOS)


def finalize(content):
    """Center content on a 4:5 BG canvas, scaling down if it overflows."""
    canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), BG)
    c = content
    if c.width > CANVAS_W or c.height > CANVAS_H:
        r = min(CANVAS_W / c.width, CANVAS_H / c.height)
        c = c.resize((int(c.width * r), int(c.height * r)), Image.Resampling.LANCZOS)
    x = (CANVAS_W - c.width) // 2
    y = (CANVAS_H - c.height) // 2
    canvas.paste(c, (x, y), c if c.mode == 'RGBA' else None)
    return canvas


# ---- render the two base charts we composite on top of ---------------------
lc.build_chart('B', lc.THEMES['B'], 'transfermrkt/_base_plain.png', features={'passport'})
lc.build_chart('B', lc.THEMES['B'], 'transfermrkt/_base_origin.png',
               features={'passport', 'origin'})
base_plain = Image.open('transfermrkt/_base_plain.png').convert('RGB')
base_origin = Image.open('transfermrkt/_base_origin.png').convert('RGB')
photo = Image.open(PHOTO).convert('RGB')


# ============================================================ VARIANT A
def variant_A():
    W = CANVAS_W
    band_h = 940
    band = crop_aspect(photo, W, band_h, va='bottom').convert('RGBA')  # keep watermark
    band.alpha_composite(bottom_gradient((W, band_h), strength=225))
    d = ImageDraw.Draw(band)
    # accent bar
    d.rounded_rectangle([70, band_h - 300, 70 + 110, band_h - 288], radius=6, fill=HERO)
    d.text((70, band_h - 268), 'From the same pitch', font=F('xbold', 78), fill=WHITE)
    d.text((70, band_h - 176), 'to the world.', font=F('xbold', 78), fill=WHITE)
    d.text((72, band_h - 78), 'Lima, 2015   \u2192   2026', font=F('semibold', 40),
           fill=(235, 235, 235))

    chart = scale_to_w(base_plain, W - 60).convert('RGBA')
    gap = 46
    total_h = band_h + gap + chart.height + 40
    content = Image.new('RGBA', (W, total_h), BG + (255,))
    content.paste(band, (0, 0), band)
    content.paste(chart, (30, band_h + gap), chart)
    return finalize(content)


# ============================================================ VARIANT B
def variant_B():
    W = CANVAS_W
    chart = scale_to_w(base_origin, W).convert('RGBA')
    # framed photo thumbnail near the shared-origin star (upper-left of the plot)
    thumb_w = 560
    thumb = crop_aspect(photo, thumb_w, int(thumb_w * 0.66), va='bottom').convert('RGBA')
    frame_pad = 10
    frame = Image.new('RGBA', (thumb.width + 2 * frame_pad, thumb.height + 2 * frame_pad + 60),
                      WHITE + (255,))
    frame = rounded(frame, 18)
    fd = ImageDraw.Draw(frame)
    fd.rounded_rectangle([0, 0, frame.width - 1, frame.height - 1], radius=18,
                         outline=HERO, width=4)
    frame.paste(thumb, (frame_pad, frame_pad), thumb)
    fd.text((frame_pad + 4, thumb.height + frame_pad + 12),
            'Same pitch \u2014 Lima, 2015', font=F('bold', 30), fill=TEXT)
    # place inside the upper-left plot region
    chart.paste(frame, (150, 150), frame)

    header_h = 190
    total_h = header_h + chart.height + 30
    content = Image.new('RGBA', (W, total_h), BG + (255,))
    d = ImageDraw.Draw(content)
    d.rounded_rectangle([70, 60, 70 + 110, 72], radius=6, fill=HERO)
    d.text((70, 92), 'Where it started', font=F('xbold', 70), fill=TEXT)
    content.paste(chart, (0, header_h), chart)
    return finalize(content)


# ============================================================ VARIANT E
def variant_E():
    W = CANVAS_W
    top = crop_aspect(photo, W - 120, 760, va='bottom').convert('RGBA')  # keep watermark
    top = rounded(top, 22)
    # "2015" tag on the photo
    td = ImageDraw.Draw(top)
    td.rounded_rectangle([28, 28, 190, 92], radius=14, fill=HERO)
    td.text((52, 38), '2015', font=F('xbold', 44), fill=WHITE)

    chart = scale_to_w(base_plain, W - 60).convert('RGBA')

    pad = 60
    gap = 40
    connector_h = 120
    total_h = pad + top.height + connector_h + chart.height + 40
    content = Image.new('RGBA', (W, total_h), BG + (255,))
    content.paste(top, (60, pad), top)
    d = ImageDraw.Draw(content)
    # center pill: 2015 -> 2026 . 11 years later
    cy = pad + top.height + connector_h // 2
    label = '2015   \u2192   2026      \u00b7      11 years later'
    f = F('bold', 40)
    tw = d.textlength(label, font=f)
    px, py = (W - tw) // 2, cy - 30
    d.rounded_rectangle([px - 40, py - 16, px + tw + 40, py + 56], radius=36,
                        fill=(240, 240, 236))
    d.text((px, py), label, font=f, fill=TEXT)
    content.paste(chart, (30, pad + top.height + connector_h), chart)
    return finalize(content)


# ============================================================ VARIANT C
def variant_C():
    W = CANVAS_W
    card_m = 60
    card_w = W - 2 * card_m
    card_h = 640
    card = Image.new('RGBA', (card_w, card_h), WHITE + (255,))
    card = rounded(card, 26)
    cd = ImageDraw.Draw(card)
    cd.rounded_rectangle([0, 0, card_w - 1, card_h - 1], radius=26,
                         outline=(226, 224, 218), width=3)
    # photo on the left (anchor right so the watermark stays)
    ph_w = 560
    ph = crop_aspect(photo, ph_w, card_h - 60, ha='right').convert('RGBA')
    ph = rounded(ph, 18)
    card.paste(ph, (30, 30), ph)
    # copy on the right
    tx = 30 + ph_w + 50
    max_w = card_w - tx - 40
    cd.rounded_rectangle([tx, 60, tx + 90, 70], radius=5, fill=HERO)
    y = 96
    for line in ['Three teammates.', 'One pitch.']:
        cd.text((tx, y), line, font=F('xbold', 56), fill=TEXT)
        y += 66
    y += 24
    body = ('Lima, Peru \u2014 2015. Where our paths began, before football '
            'and life scattered us across the world.')
    for line in wrap(cd, body, F('regular', 30), max_w):
        cd.text((tx, y), line, font=F('regular', 30), fill=MUTED)
        y += 42
    y += 26
    cd.text((tx, y), 'Fernando  \u00b7  ', font=F('semibold', 30), fill=MUTED)
    w1 = cd.textlength('Fernando  \u00b7  ', font=F('semibold', 30))
    cd.text((tx + w1, y), 'Enzo', font=F('bold', 30), fill=HERO)
    w2 = cd.textlength('Enzo', font=F('bold', 30))
    cd.text((tx + w1 + w2, y), '  \u00b7  Franco', font=F('semibold', 30), fill=MUTED)

    chart = scale_to_w(base_plain, W - 60).convert('RGBA')
    gap = 46
    total_h = card_m + card_h + gap + chart.height + 40
    content = Image.new('RGBA', (W, total_h), BG + (255,))
    content.paste(card, (card_m, card_m), card)
    content.paste(chart, (30, card_m + card_h + gap), chart)
    return finalize(content)


for name, fn in [('A', variant_A), ('B', variant_B), ('E', variant_E), ('C', variant_C)]:
    out = f'transfermrkt/mock_{name}.png'
    fn().save(out)
    print(f'Saved -> {out}')

for tmp in ['transfermrkt/_base_plain.png', 'transfermrkt/_base_origin.png']:
    try:
        os.remove(tmp)
    except OSError:
        pass

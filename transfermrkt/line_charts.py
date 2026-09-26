import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import json
from datetime import datetime
from matplotlib.ticker import FuncFormatter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import os

# Set the style for dark background
plt.style.use('dark_background')

# --- Typography: register Montserrat and expose a weight helper --------------
import matplotlib.font_manager as fm

_FONT_DIR = 'fonts'
_FONT_FILES = {
    'regular': 'Montserrat-Regular.ttf',
    'medium': 'Montserrat-Medium.ttf',
    'semibold': 'Montserrat-SemiBold.ttf',
    'bold': 'Montserrat-Bold.ttf',
    'xbold': 'Montserrat-ExtraBold.ttf',
}
for _f in _FONT_FILES.values():
    _p = os.path.join(_FONT_DIR, _f)
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
plt.rcParams['font.family'] = 'Montserrat'


def font(weight='regular', size=10):
    """Return a Montserrat FontProperties at the given weight/size."""
    path = os.path.join(_FONT_DIR, _FONT_FILES.get(weight, _FONT_FILES['regular']))
    if os.path.exists(path):
        return fm.FontProperties(fname=path, size=size)
    return fm.FontProperties(size=size)

def get_club_logo(club_name, zoom=0.1, target_px=None, square=False, alpha=1.0):
    """Load a club/company logo.

    If target_px is set, logos are scaled to that max side.
    If square=True, transparent padding is cropped first, then the logo is
    fitted into a target_px x target_px canvas so all marks share equal size.
    alpha < 1 dims the logo (used to keep friends' crests visually secondary).
    """
    logo_path = os.path.join('team_logos', f'{club_name}.png')
    try:
        img = Image.open(logo_path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        if target_px is not None:
            if square:
                bbox = img.getbbox()
                if bbox:
                    img = img.crop(bbox)
            w, h = img.size
            scale = target_px / max(w, h)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            if square:
                canvas = Image.new('RGBA', (target_px, target_px), (0, 0, 0, 0))
                ox = (target_px - img.width) // 2
                oy = (target_px - img.height) // 2
                canvas.paste(img, (ox, oy), img)
                img = canvas
        return OffsetImage(img, zoom=zoom, alpha=alpha)
    except (FileNotFoundError, IOError):
        print(f"Logo not found for: {club_name}")  # Debug print
        return None

with open('transfermrkt/comparison_players.json', 'r') as f:
    data = json.load(f)

# Build long format DataFrame
rows = []
for player, pdata in data.items():
    prev_club = None
    for entry in pdata['market_value_data']['marketValueDevelopment']['list']:
        timestamp = entry['x'] // 1000  # Convert ms to seconds
        current_club = entry.get('verein', 'Unknown Club')
        is_transfer = current_club != prev_club
        prev_club = current_club
        
        rows.append({
            'player': player.title(),
            'date': datetime.fromtimestamp(timestamp),
            'market_value_eur': entry['y'],
            'market_value_label': entry['mw'],
            'club': current_club,
            'is_transfer': is_transfer
        })

df = pd.DataFrame(rows)

# Add Enzo Villafuerte dummy data
enzo_data = [
    {'date': 'Jan 2019', 'club': 'Cedarville University', 'market_value_eur': 15560},
    {'date': 'Aug 2020', 'club': 'Ohio University', 'market_value_eur': 20560},
    {'date': 'Apr 2022', 'club': 'Kandle', 'market_value_eur': 28667},
    {'date': 'Sep 2022', 'club': 'Ohio University', 'market_value_eur': 32000},
    {'date': 'Dec 2022', 'club': 'Graduation', 'market_value_eur': 35000},
    {'date': 'May 2023', 'club': 'Hungary', 'market_value_eur': 37000},
    {'date': 'Aug 2023', 'club': 'Ohio University', 'market_value_eur': 38800},
    {'date': 'Apr 2024', 'club': 'Nucor', 'market_value_eur': 55800},
    {'date': 'Sep 2024', 'club': 'Ohio University', 'market_value_eur': 56800},
    {'date': 'Jan 2025', 'club': 'Ohio University', 'market_value_eur': 58000},
    {'date': 'Jul 2025', 'club': 'Hungary', 'market_value_eur': 61000},
    {'date': 'Oct 2025', 'club': 'Sporting Cristal', 'market_value_eur': 75000},
    {'date': 'Jun 2026', 'club': 'Deloitte', 'market_value_eur': 85000},
]

enzo_rows = []
prev_club = None
for entry in enzo_data:
    date = pd.to_datetime(entry['date'], format='%b %Y')
    club = entry['club']
    value = entry['market_value_eur']
    label = f"€{int(value/1000)}k" if value >= 1000 else f"€{value}"
    is_transfer = club != prev_club
    prev_club = club
    enzo_rows.append({
        'player': 'Enzo Villafuerte',
        'date': date,
        'market_value_eur': value,
        'market_value_label': label,
        'club': club,
        'is_transfer': is_transfer
    })

df = pd.concat([df, pd.DataFrame(enzo_rows)], ignore_index=True)

HERO = 'Enzo Villafuerte'

# --- Geography layer: club -> country, flags, and helpers --------------------
from PIL import ImageOps, ImageDraw


def _hex_rgba(color, alpha=255):
    r, g, b, _ = mcolors.to_rgba(color)
    return (int(r * 255), int(g * 255), int(b * 255), alpha)


def make_avatar(path, size=150, ring_color=None, ring_width=8):
    """Circular headshot (top-biased square crop) with an optional colored ring."""
    img = Image.open(path).convert('RGBA')
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = 0 if h >= w else (h - s) // 2  # keep the head for tall portraits
    img = img.crop((left, top, left + s, top + s)).resize((size, size))
    mask = Image.new('L', (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    avatar = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    avatar.paste(img, (0, 0), mask)
    if ring_color is not None and ring_width > 0:
        pad = ring_width
        canvas = Image.new('RGBA', (size + 2 * pad, size + 2 * pad), (0, 0, 0, 0))
        ImageDraw.Draw(canvas).ellipse(
            (pad // 2, pad // 2, size + 2 * pad - pad // 2 - 1,
             size + 2 * pad - pad // 2 - 1), outline=ring_color, width=ring_width)
        canvas.paste(avatar, (pad, pad), avatar)
        avatar = canvas
    return avatar

FLAG_DIR = 'flags'
COUNTRY_NAME = {'pe': 'Peru', 'br': 'Brazil', 'nl': 'Netherlands',
                'il': 'Israel', 'us': 'USA', 'es': 'Spain', 'hu': 'Hungary'}
COUNTRY_LONLAT = {'pe': (-75, -10), 'br': (-51, -10), 'nl': (5.3, 52.1),
                  'il': (34.8, 31.0), 'us': (-98, 39), 'es': (-3.7, 40.4),
                  'hu': (19.5, 47.2)}
COUNTRY_COLOR = {'us': '#3b5bdb', 'pe': '#e03131', 'es': '#f08c00',
                 'br': '#2f9e44', 'nl': '#f76707', 'il': '#7048e8',
                 'hu': '#c2255c'}
CLUB_COUNTRY = {
    # Enzo (personal)
    'Cedarville University': 'us', 'Ohio University': 'us', 'Kandle': 'us',
    'Graduation': 'us', 'Nucor': 'us', 'Sporting Cristal': 'pe', 'Deloitte': 'es',
    'Hungary': 'hu',
    # Fernando Pacheco / Franco Medina (clubs)
    'Club Sporting Cristal': 'pe', 'Fluminense Football Club': 'br',
    'Esporte Clube Juventude': 'br', 'FC Emmen': 'nl',
    'Deportivo Municipal': 'pe', 'Club Cienciano': 'pe',
    'Ironi Kiryat Shmona': 'il', 'Universidad César Vallejo': 'pe',
    'Club Alianza Lima': 'pe', 'Comerciantes Unidos': 'pe',
    'Ayacucho FC': 'pe', 'Monsoon Futebol Clube': 'br',
}


def _load_flag(iso, border=2, border_color=(90, 90, 90, 255)):
    img = Image.open(os.path.join(FLAG_DIR, f'{iso}.png')).convert('RGBA')
    if border:
        img = ImageOps.expand(img, border=border, fill=border_color)
    return img


def get_flag(iso, zoom=0.5, border=2):
    try:
        return OffsetImage(_load_flag(iso, border=border), zoom=zoom)
    except Exception:
        return None


def make_passport_strip(isos, flag_h=26, gap=5):
    """Horizontal strip of a player's visited-country flags (passport stamps)."""
    imgs = []
    for iso in isos:
        im = _load_flag(iso, border=1)
        w = max(1, int(im.width * flag_h / im.height))
        imgs.append(im.resize((w, flag_h)))
    if not imgs:
        return None
    total_w = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    strip = Image.new('RGBA', (total_w, flag_h), (0, 0, 0, 0))
    x = 0
    for i in imgs:
        strip.paste(i, (x, 0), i)
        x += i.width + gap
    return strip


def country_trail(group):
    """Country codes in travel order, collapsing consecutive repeats (keeps revisits)."""
    seq = []
    for club in group['club']:
        iso = CLUB_COUNTRY.get(club)
        if iso and (not seq or seq[-1] != iso):
            seq.append(iso)
    return seq


def unique_countries(group):
    """First-occurrence-unique country codes for a player's timeline."""
    seen = []
    for club in group['club']:
        iso = CLUB_COUNTRY.get(club)
        if iso and iso not in seen:
            seen.append(iso)
    return seen


def format_currency(x, pos):
    """Format y-axis values in millions/thousands."""
    if x >= 1_000_000:
        return f'€{x/1_000_000:.1f}M'
    elif x >= 1_000:
        return f'€{x/1_000:.0f}k'
    return f'€{x:.0f}'


# Three candidate palettes, all on a light editorial background.
# "hero" = Enzo (protagonist), "friends" = supporting cast (present but calmer).
THEMES = {
    'A': {
        'label': 'Editorial Light',
        'bg': '#F6F5F1',
        'panel': '#F6F5F1',
        'hero': '#F4511E',            # bold coral-orange
        'friends': ['#7A8CA3', '#B0B8C4'],  # calm slate grays
        'text': '#2B2B2B',
        'muted_text': '#6B7280',
        'grid': '#E4E2DC',
    },
    'B': {
        'label': 'Hero vs. Grayed Friends',
        'bg': '#FAFAF8',
        'panel': '#FAFAF8',
        'hero': '#E5342B',            # vivid red hero
        'friends': ['#9AA0A6', '#C4C8CD'],  # pure neutral grays
        'text': '#222222',
        'muted_text': '#8A8F94',
        'grid': '#EAE9E4',
    },
    'C': {
        'label': 'Colorful but Ranked',
        'bg': '#F7F6F2',
        'panel': '#F7F6F2',
        'hero': '#00A86B',            # emerald hero
        'friends': ['#D98A86', '#93B7CE'],  # muted red / muted blue
        'text': '#2B2B2B',
        'muted_text': '#6B7280',
        'grid': '#E6E4DE',
    },
}


def build_chart(theme_key, theme, out_path, features=frozenset(), figsize=(15, 8)):
    if 'flag_lane' in features:
        fig, (ax, lane) = plt.subplots(
            2, 1, figsize=(15, 9), height_ratios=[5, 1.1], sharex=True,
            gridspec_kw={'hspace': 0.08})
        lane.set_facecolor(theme['panel'])
    else:
        fig, ax = plt.subplots(figsize=figsize)
        lane = None
    fig.patch.set_facecolor(theme['bg'])
    ax.set_facecolor(theme['panel'])

    groups = list(df.groupby('player'))
    # Assign colors: hero gets the accent, friends cycle through muted tones.
    friend_idx = 0
    styles = {}
    for player, _ in groups:
        if player == HERO:
            styles[player] = dict(color=theme['hero'], lw=3.6, alpha=1.0,
                                  ms=9, zorder=6, hero=True)
        else:
            c = theme['friends'][friend_idx % len(theme['friends'])]
            friend_idx += 1
            styles[player] = dict(color=c, lw=2.0, alpha=0.9,
                                  ms=6, zorder=3, hero=False)

    last_date = df['date'].max()

    # Draw friends first, hero last so it sits on top.
    ordered = sorted(groups, key=lambda g: styles[g[0]]['hero'])
    first_points = {}
    end_info = {}

    for player, group in ordered:
        st = styles[player]
        group = group.sort_values('date')
        first_points[player] = (group.iloc[0]['date'], group.iloc[0]['market_value_eur'])
        ax.plot(group['date'], group['market_value_eur'],
                marker='o', linewidth=st['lw'], color=st['color'],
                markersize=st['ms'], alpha=st['alpha'], zorder=st['zorder'],
                solid_capstyle='round')

        # ---- Value labels ----
        if st['hero']:
            # Only label a "new club" signing (a transfer point that has a logo),
            # always placed above the point like the friends' lines.
            for _, r in group[group['is_transfer']].iterrows():
                if os.path.exists(os.path.join('team_logos', f"{r['club']}.png")):
                    ax.annotate(r['market_value_label'],
                                (r['date'], r['market_value_eur']),
                                xytext=(0, 13), textcoords='offset points',
                                ha='center', va='bottom', color=theme['text'],
                                fontproperties=font('bold', 12), zorder=7)
        else:
            # Friends: label every transfer (club change) + their latest value.
            show_idx = set(group[group['is_transfer']].index)
            show_idx.add(group.index[-1])
            for ridx in show_idx:
                r = group.loc[ridx]
                ax.annotate(r['market_value_label'], (r['date'], r['market_value_eur']),
                            xytext=(0, 4), textcoords='offset points', ha='center',
                            va='bottom', color=st['color'],
                            fontproperties=font('semibold', 10.5),
                            alpha=0.95, zorder=4)

        # ---- Transfer markers: club logos, or country flags (feature #2) ----
        for _, row in group[group['is_transfer']].iterrows():
            if 'flags_transfers' in features:
                iso = CLUB_COUNTRY.get(row['club'])
                if iso:
                    fl = get_flag(iso, zoom=0.55 if st['hero'] else 0.42)
                    y_m, z_m = (34, 8) if st['hero'] else (30, 5)
                    if fl is not None:
                        ax.add_artist(AnnotationBbox(
                            fl, (row['date'], row['market_value_eur']),
                            xybox=(0, y_m), frameon=False, xycoords='data',
                            boxcoords='offset points', box_alignment=(0.5, 0),
                            pad=0, zorder=z_m))
                continue
            if st['hero']:
                logo = get_club_logo(row['club'], zoom=0.38, target_px=56,
                                     square=True, alpha=1.0)
                y_logo, z_logo = 34, 8
            else:
                logo = get_club_logo(row['club'], zoom=0.34, target_px=58,
                                     square=True, alpha=0.9)
                y_logo, z_logo = 16, 5
            if logo is not None:
                ax.add_artist(AnnotationBbox(
                    logo, (row['date'], row['market_value_eur']),
                    xybox=(0, y_logo), frameon=False, xycoords='data',
                    boxcoords='offset points', box_alignment=(0.5, 0),
                    pad=0, zorder=z_logo))

        # ---- Collect end-of-line info; faces are drawn later in an aligned column ----
        last_row = group.iloc[-1]
        end_info[player] = dict(
            x=last_row['date'], y=last_row['market_value_eur'],
            color=st['color'], hero=st['hero'],
            countries=unique_countries(group))

    # ---- Annotate a key moment on the hero line (the Sporting Cristal jump) ----
    hero_group = df[(df['player'] == HERO) & (df['club'] == 'Sporting Cristal')]
    if not hero_group.empty:
        kr = hero_group.iloc[0]
        ax.annotate('Back to the same pitch.\nNow as an engineer',
                    (kr['date'], kr['market_value_eur']),
                    xytext=(-95, 78), textcoords='offset points',
                    ha='center', va='bottom', fontsize=12,
                    color=theme['hero'], fontweight='bold',
                    arrowprops=dict(arrowstyle='-|>', color=theme['hero'],
                                    lw=1.4, alpha=0.9,
                                    connectionstyle='arc3,rad=0.2'),
                    zorder=10)

    # ---- Feature #5: country chapters, filled ONLY under the hero line ----
    if 'bands' in features:
        hero_g = df[df['player'] == HERO].sort_values('date')
        xs = list(hero_g['date'])
        ys = list(hero_g['market_value_eur'])
        isos = [CLUB_COUNTRY.get(c) for c in hero_g['club']]
        # Extend the final chapter a touch to the right so it reads as a block.
        xs2 = xs + [xs[-1] + pd.DateOffset(months=3)]
        ys2 = ys + [ys[-1]]
        isos2 = isos + [isos[-1]]
        # Colour the area between the hero line and 0, per active country.
        for i in range(len(xs2) - 1):
            iso = isos2[i]
            if not iso:
                continue
            ax.fill_between([xs2[i], xs2[i + 1]], [ys2[i], ys2[i + 1]], 0,
                            color=COUNTRY_COLOR.get(iso, '#888'), alpha=0.22,
                            linewidth=0, zorder=1)
        # Merge consecutive same-country points into labelled chapters.
        chapters, ci, cs = [], isos2[0], xs2[0]
        for i in range(1, len(isos2)):
            if isos2[i] != ci:
                chapters.append((ci, cs, xs2[i]))
                ci, cs = isos2[i], xs2[i]
        chapters.append((ci, cs, xs2[-1]))
        xnum = np.array([d.value for d in xs])
        ynum = np.array(ys, dtype=float)
        for iso, s, e in chapters:
            if not iso:
                continue
            mid = s + (e - s) / 2
            ly = float(np.interp(mid.value, xnum, ynum))
            ax.text(mid, max(ly * 0.5, 3500), COUNTRY_NAME.get(iso, iso),
                    ha='center', va='center', fontproperties=font('bold', 7.5),
                    color=COUNTRY_COLOR.get(iso, '#888'), alpha=0.95, zorder=3)

    # ---- Feature #1: shared 2015 origin ("same pitch") ----
    if 'origin' in features:
        origin_date = pd.Timestamp('2015-05-01')
        origin_y = 150000
        for player, (fx, fy) in first_points.items():
            ax.plot([origin_date, fx], [origin_y, fy], linestyle=':',
                    color=styles[player]['color'], lw=1.3, alpha=0.6, zorder=2)
        ax.scatter([origin_date], [origin_y], s=260, marker='*',
                   color=theme['text'], zorder=11, edgecolor='white', linewidth=1.0)
        # Only print the standalone caption when we're NOT also showing the photo
        # inset (the inset carries its own "Same pitch — Lima, 2015" caption).
        if 'photo_inset' not in features:
            ax.annotate('Same pitch — Lima, Peru, 2015',
                        (origin_date, origin_y), xytext=(6, 52),
                        textcoords='offset points', ha='left', va='bottom',
                        fontsize=10.5, fontweight='bold', color=theme['text'],
                        arrowprops=dict(arrowstyle='-', color=theme['text'],
                                        lw=1.0, alpha=0.7), zorder=12)

    # ---- 2015 group photo as an inset anchored to the shared origin ----------
    if 'photo_inset' in features:
        origin_date = pd.Timestamp('2015-05-01')
        origin_y = 150000
        pim = np.asarray(Image.open('transfermrkt/IMG_3664.JPG').convert('RGB'))
        iax = ax.inset_axes([0.015, 0.60, 0.34, 0.36])
        iax.imshow(pim, aspect='auto')
        iax.set_xticks([])
        iax.set_yticks([])
        for sp in iax.spines.values():
            sp.set_edgecolor(theme['hero'])
            sp.set_linewidth(3.2)
        iax.set_title('Same pitch — Lima, 2015', fontproperties=font('bold', 11.5),
                      color=theme['text'], loc='left', pad=7)
        # Arrow from the bottom of the inset down to the origin star.
        ax.annotate('', xy=(origin_date, origin_y), xycoords='data',
                    xytext=(0.185, 0.585), textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle='-|>', color=theme['hero'], lw=1.8,
                                    alpha=0.9,
                                    connectionstyle='arc3,rad=-0.25'), zorder=12)

    # ---- Feature #6: world-map inset with journey routes ----
    if 'map' in features:
        world = np.asarray(Image.open(os.path.join(FLAG_DIR, 'world.png')).convert('RGB'))
        mp = ax.inset_axes([0.015, 0.52, 0.36, 0.45])
        mp.imshow(world, extent=[-180, 180, -90, 90], aspect='auto', zorder=0)
        mp.set_xlim(-115, 45)
        mp.set_ylim(-25, 62)
        mp.set_xticks([])
        mp.set_yticks([])
        for sp in mp.spines.values():
            sp.set_edgecolor(theme['grid'])
        for player, group in ordered:
            trail = country_trail(group.sort_values('date'))
            col = styles[player]['color']
            lw = 2.6 if player == HERO else 1.5
            for a, b in zip(trail[:-1], trail[1:]):
                (x1, y1), (x2, y2) = COUNTRY_LONLAT[a], COUNTRY_LONLAT[b]
                mp.annotate('', xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle='-|>', color=col, lw=lw,
                                            alpha=0.9,
                                            connectionstyle='arc3,rad=0.2'),
                            zorder=4)
            for iso in trail:
                x, y = COUNTRY_LONLAT[iso]
                fl = get_flag(iso, zoom=0.26)
                if fl is not None:
                    mp.add_artist(AnnotationBbox(fl, (x, y), frameon=False,
                                                 box_alignment=(0.5, 0.5), pad=0,
                                                 zorder=5))
        mp.set_title('Journeys', fontsize=10, color=theme['text'],
                     fontweight='bold', loc='left')

    # ---- Feature #4: flag lane below the chart ----
    if lane is not None:
        players_order = [HERO] + [p for p, _ in groups if p != HERO]
        row_y = {p: idx for idx, p in enumerate(reversed(players_order))}  # hero on top
        for player, group in ordered:
            g = group.sort_values('date')
            y = row_y[player]
            lane.plot([g['date'].min(), g['date'].max()], [y, y],
                      color=styles[player]['color'], lw=2, alpha=0.5, zorder=1)
            prev = None
            for _, row in g.iterrows():
                iso = CLUB_COUNTRY.get(row['club'])
                if iso and iso != prev:
                    prev = iso
                    fl = get_flag(iso, zoom=0.34)
                    if fl is not None:
                        lane.add_artist(AnnotationBbox(
                            fl, (row['date'], y), frameon=False,
                            box_alignment=(0.5, 0.5), pad=0, zorder=3))
            lane.annotate(player, (g['date'].min(), y), xytext=(-10, 0),
                          textcoords='offset points', ha='right', va='center',
                          fontsize=9, color=styles[player]['color'],
                          fontweight='bold' if player == HERO else 'normal')
        lane.set_ylim(-0.6, len(players_order) - 0.4)
        lane.set_yticks([])
        for sp in ['top', 'right', 'left']:
            lane.spines[sp].set_visible(False)
        lane.spines['bottom'].set_color(theme['grid'])
        lane.grid(False)
        lane.text(0.0, 1.04, 'Countries over time', transform=lane.transAxes,
                  fontsize=10, color=theme['muted_text'], ha='left', va='bottom',
                  fontweight='bold')

    # ---- Aligned face column on the right (consistent x, size, + leaders) ----
    x_col = last_date + pd.DateOffset(months=6)
    face_h = 150  # normalise every headshot to the same pixel height
    ymax_col = ax.get_ylim()[1]
    gap = ymax_col * 0.16
    floor = ymax_col * 0.05
    targets, prev = {}, None
    for player, info in sorted(end_info.items(), key=lambda kv: kv[1]['y']):
        ty = max(info['y'], floor)
        if prev is not None and ty < prev + gap:
            ty = prev + gap
        targets[player] = ty
        prev = ty
    for player, info in end_info.items():
        ty = targets[player]
        col = info['color']
        # leader from the true endpoint to the aligned face row
        ax.plot([info['x'], x_col], [info['y'], ty], linestyle=':',
                color=col, lw=1.2, alpha=0.55, zorder=2)
        try:
            path = os.path.join('whoscored-vizzes/players_png', f'{player}.png')
            # Hero gets a bold ring in the accent colour; friends a subtle grey one.
            ring = _hex_rgba(theme['hero']) if info['hero'] else (170, 170, 170, 255)
            ring_w = 6 if info['hero'] else 4
            avatar = make_avatar(path, size=face_h, ring_color=ring, ring_width=ring_w)
            ax.add_artist(AnnotationBbox(
                OffsetImage(avatar, zoom=0.28),
                (x_col, ty), frameon=False, xycoords='data',
                box_alignment=(0.5, 0.5), pad=0, zorder=8, annotation_clip=False))
        except Exception:
            pass
        # Offset text clear of the (now smaller) avatar.
        tx = 30 if info['hero'] else 26
        display_name = 'Me' if info['hero'] else player
        ax.annotate(display_name, (x_col, ty), xytext=(tx, 7),
                    textcoords='offset points', ha='left', va='center',
                    color=(theme['text'] if info['hero'] else col),
                    fontproperties=font('bold' if info['hero'] else 'semibold',
                                        14 if info['hero'] else 12),
                    zorder=9, annotation_clip=False)
        if 'passport' in features:
            strip = make_passport_strip(info['countries'], flag_h=22, gap=3)
            if strip is not None:
                ax.add_artist(AnnotationBbox(
                    OffsetImage(strip, zoom=0.5), (x_col, ty), xybox=(tx, -13),
                    frameon=False, xycoords='data', boxcoords='offset points',
                    box_alignment=(0, 0.5), pad=0, zorder=9,
                    annotation_clip=False))

    # ---- Thin vertical divider between the plot and the roster column ----
    div_x = last_date + pd.DateOffset(months=2)
    ax.axvline(div_x, color=theme['grid'], lw=1.3, alpha=0.9, zorder=0)

    # ---- Titles + hero-red accent bar ----
    ax.plot([0.0, 0.075], [1.145, 1.145], transform=ax.transAxes,
            color=theme['hero'], lw=4.5, solid_capstyle='round',
            clip_on=False, zorder=5)
    ax.set_title('Market Value Progression Over Time',
                 fontproperties=font('xbold', 22), color=theme['text'],
                 loc='left', pad=30)
    ax.text(0.0, 1.02, 'Me vs. my former teammate & rival',
            transform=ax.transAxes, fontproperties=font('medium', 13),
            color=theme['muted_text'], ha='left', va='bottom')

    # ---- Axes ----
    ax.set_ylabel('Market Value', fontproperties=font('medium', 14),
                  color=theme['muted_text'], labelpad=12)
    ax.set_xlabel('')
    bottom_ax = lane if lane is not None else ax
    bottom_ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    bottom_ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    bottom_ax.tick_params(colors=theme['muted_text'])
    for lbl in bottom_ax.get_xticklabels():
        lbl.set_color(theme['muted_text'])
        lbl.set_fontproperties(font('regular', 13))
    if lane is not None:
        plt.setp(ax.get_xticklabels(), visible=False)
    for lbl in ax.get_yticklabels():
        lbl.set_color(theme['muted_text'])
        lbl.set_fontproperties(font('regular', 13))
    ax.tick_params(colors=theme['muted_text'])
    ax.yaxis.set_major_formatter(FuncFormatter(format_currency))

    # Give room on the right for the aligned face column (and left for origin).
    left = (pd.Timestamp('2014-10-01') if 'origin' in features
            else df['date'].min() - pd.DateOffset(months=4))
    ax.set_xlim(left, last_date + pd.DateOffset(months=17))

    # ---- Grid: faint horizontal only ----
    ax.grid(True, axis='y', linestyle='-', alpha=0.9, color=theme['grid'], zorder=0)
    ax.grid(False, axis='x')
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(theme['grid'])

    # ---- Source note ----
    fig.text(0.01, 0.01, 'Source: Transfermarkt (teammates) · inflated personal data (Mine)',
             fontproperties=font('regular', 8), color=theme['muted_text'],
             ha='left', va='bottom')

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved -> {out_path}')


# Final chart: passport strips only (no shading under the hero line).
build_chart('B', THEMES['B'], 'transfermrkt/line_charts.png',
            features={'passport'})

# python transfermrkt/line_charts.py
# .venv/bin/python transfermrkt/line_charts.py




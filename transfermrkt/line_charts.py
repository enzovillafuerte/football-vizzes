import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from matplotlib.ticker import FuncFormatter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import os

# Set the style for dark background
plt.style.use('dark_background')

def get_club_logo(club_name, zoom=0.1):  # Reduced zoom for better fit
    """Load and return a club logo image if it exists."""
    logo_path = os.path.join('team_logos', f'{club_name}.png')
    try:
        img = Image.open(logo_path)
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        return OffsetImage(img, zoom=zoom)
    except (FileNotFoundError, IOError):
        print(f"Logo not found for: {club_name}")  # Debug print
        return None

target_players = ['fernando pacheco', 'franco medina']

with open('transfermrkt/peruvian_league_players.json', 'r') as f:
    data = json.load(f)

filtered_players = {player: info for player, info in data.items() if player in target_players}

# print(filtered_players)

# Build long format DataFrame
rows = []
for player, pdata in filtered_players.items():
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
    {'date': 'Aug 2018', 'club': 'Ohio University', 'market_value_eur': 10560},
    {'date': 'Jun 2022', 'club': 'Kandle', 'market_value_eur': 18667},
    {'date': 'Sep 2022', 'club': 'Ohio University', 'market_value_eur': 12000},
    {'date': 'Dec 2022', 'club': 'Graduation', 'market_value_eur': 12000},
    {'date': 'Aug 2023', 'club': 'Ohio University', 'market_value_eur': 16800},
    {'date': 'May 2024', 'club': 'Nucor', 'market_value_eur': 27800},
    {'date': 'Sep 2024', 'club': 'Ohio University', 'market_value_eur': 16800},
    {'date': 'Jan 2025', 'club': 'Ohio University', 'market_value_eur': 15000}
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

# Create figure with dark background and larger size
plt.figure(figsize=(15, 8))  # Increased figure size
fig = plt.gcf()
fig.patch.set_facecolor('#1a1a1a')  # Dark blue background
ax = plt.gca()
ax.set_facecolor('#1a1a1a')

# Custom color palette
colors = ['#00ff9d', '#ff6b6b', '#4cc9f0']  # Bright green, coral red, bright blue

# Format y-axis to show values in millions/thousands
def format_currency(x, pos):
    if x >= 1_000_000:
        return f'€{x/1_000_000:.1f}M'
    elif x >= 1_000:
        return f'€{x/1_000:.0f}k'
    return f'€{x:.0f}'

# Plot lines for each player
for i, (player, group) in enumerate(df.groupby('player')):
    line = plt.plot(group['date'], group['market_value_eur'], 
                   marker='o', linewidth=2.5, color=colors[i], 
                   label=player, markersize=8)
    
    # Add data labels for market values
    for x, y, label in zip(group['date'], group['market_value_eur'], group['market_value_label']):
        plt.annotate(label, 
                    (x, y), 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    color='white',
                    fontsize=9)
    
    # Add club logo annotations for transfers
    for idx, row in group[group['is_transfer']].iterrows():
        if player == 'Enzo Villafuerte':
            logo = get_club_logo(row['club'], zoom=0.04)  # Smaller logos for Enzo
        else:
            logo = get_club_logo(row['club'])  # Default size for others
        if logo is not None:
            y_offset = 30 if i == 0 else 30  # in points, above/below
            ab = AnnotationBbox(
                logo,
                (row['date'], row['market_value_eur']),
                xybox=(0, y_offset),
                frameon=False,
                xycoords='data',
                boxcoords='offset points',
                pad=0
            )
            ax.add_artist(ab)

    # --- Add player headshot at the end of the line ---
    last_row = group.iloc[-1]
    player_img_path = os.path.join('whoscored-vizzes/players_png', f'{player}.png')
    try:
        img = Image.open(player_img_path)
        img = img.convert('RGBA')
        zoom = 0.20 if player == 'Enzo Villafuerte' else 0.24
        headshot = OffsetImage(img, zoom=zoom)
        ab_headshot = AnnotationBbox(
            headshot,
            (last_row['date'], last_row['market_value_eur']),
            xybox=(50, 0),  # 50 points to the right
            frameon=False,
            xycoords='data',
            boxcoords='offset points',
            pad=0
        )
        ax.add_artist(ab_headshot)
    except Exception:
        pass

# Customize the plot
plt.title('Market Value Progression Over Time', 
          fontsize=16, 
          fontweight='bold', 
          color='white',
          pad=20)

# Format axes
plt.ylabel('Market Value', fontsize=12, color='white', labelpad=15)
plt.xlabel('Date', fontsize=12, color='white', labelpad=15)

# Format x-axis dates
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
plt.xticks(rotation=45, color='white')
plt.yticks(color='white')

# Format y-axis
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))

# Remove top and right spines
sns.despine()

# Add legend (text and color line only)
plt.legend(title='Player', 
          loc='upper left',
          frameon=True,
          facecolor='#1a1a1a',
          edgecolor='white',
          title_fontsize=12,
          fontsize=10)

# Add grid
plt.grid(True, linestyle='--', alpha=0.2, color='white')

# Adjust layout with more padding
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)

plt.tight_layout(pad=3.0)
plt.savefig('transfermrkt/line_charts.png', dpi=200)

# plt.show()











# python transfermrkt/line_charts.py




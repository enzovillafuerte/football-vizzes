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

# Create figure with dark background and larger size
plt.figure(figsize=(15, 8))  # Increased figure size
fig = plt.gcf()
fig.patch.set_facecolor('#1a1a1a')  # Dark blue background
ax = plt.gca()
ax.set_facecolor('#1a1a1a')

# Custom color palette
colors = ['#00ff9d', '#ff6b6b']  # Bright green and coral red

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
        logo = get_club_logo(row['club'])
        if logo is not None:
            y_offset = 30 if i == 0 else -40  # in points, above/below
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

# Add legend
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




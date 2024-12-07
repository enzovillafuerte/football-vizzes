"""
Takes a url from whoscored and generates a montecarlo report
# Probabilities of each team winning based on xG
# Probabilities of a game having over 2.5 goals or 1.5 goals based on xG
# xG flowchart
# Shot map (Maybe) ~ Keep it simple first
# See statsbomb xG flowchart for reference, try to replicate
# Use scraping code from other projects understat.com
"""

import json
import time
import numpy as np
import pandas as pd
import seaborn as sns
import requests
from bs4 import BeautifulSoup
from typing import List, Optional
from selenium import webdriver
from supabase import create_client, Client
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import to_rgba
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from mplsoccer import Pitch, FontManager, Sbopen
import urllib
from PIL import Image
from mplsoccer import Pitch, VerticalPitch, FontManager, Sbopen
import sys


############################################################################################
## Section 1 - Scraping from understat -> Ideally all the scrapers we put them in module -> main.py file as for V2.1 (Do later)
############################################################################################

# Storing the url provided by the user
# Ensuring a url is provided as input/argument
if len(sys.argv) <2:
    print("Provide a valid url -> python montecarlo/post_game_viz.py 'url'")
    sys.exit(1)

# Storing the url     
url = sys.argv[1]

"""
# Scraping from Understat
# Using requests to get the webpage and let Beautiful soup to parse the page
res = requests.get(url)  
soup = BeautifulSoup(res.content, 'lxml')  
scripts = soup.find_all('script')

# Getting only the shotsData
strings = scripts[1].string

# strip unnecessary symbols and get only JSON data 
ind_start = strings.index("('")+2 
ind_end = strings.index("')") 
json_data = strings[ind_start:ind_end] 
json_data = json_data.encode('utf8').decode('unicode_escape')

#convert string to json format
data = json.loads(json_data)

# we are going to create lists for the different variables we want to store. In this case we want to get the coordinates x and y out, the xG values,
#the result of the shot and the minute of the shot.

# After we created both data_away and data_home variables, we can manually use indexing to see a specific record (example: data_home[1] and see the 
#                                                                                                                columns available.
                                                                                                                
# Then we will use a loop to iterate through all the records and assign values to the variable lists just created

x = []
y = []
xG = []
result = []
minute = []
player = []
team = []

data_away = data['a']
data_home = data['h']

data_home[1]

for index in range(len(data_home)):
    for key in data_home[index]:
        if key == 'X':
            x.append(data_home[index][key])
        if key == 'Y':
            y.append(data_home[index][key])
        if key == 'h_team':
            team.append(data_home[index][key])
        if key == 'xG':
            xG.append(data_home[index][key])
        if key == 'result':
            result.append(data_home[index][key])   
        if key == 'minute':
            minute.append(data_home[index][key])
        if key == 'player':
            player.append(data_home[index][key])    

for index in range(len(data_away)):
    for key in data_away[index]:
        if key == 'X':
            x.append(data_away[index][key])
        if key == 'Y':
            y.append(data_away[index][key])
        if key == 'a_team':
            team.append(data_away[index][key])
        if key == 'xG':
            xG.append(data_away[index][key])
        if key == 'result':
            result.append(data_away[index][key])
        if key == 'minute':
            minute.append(data_away[index][key])   
        if key == 'player':
            player.append(data_away[index][key])      
 
# Extracting home and away teams programatically
home_team = data_home[0].get('h_team', 'Unknown') if data_home else 'Unknown'
away_team = data_away[0].get('a_team', 'Unknown') if data_away else 'Unknown'

# create the pandas dataframe with the lists previously created with the loops
            
col_names = ['x','y','xG','result','team', 'minute', 'player']
df = pd.DataFrame([x,y,xG,result,team, minute, player],index=col_names)
df = df.T      

# Save the dataset to use it as baseline for the Montecarlo so we don't send too many requests to the website while testing
#df.to_csv('montecarlo/baseline_df.csv', index=False)
"""

############################################################################################
## Section 2 - Montecarlo Simulations - work with the sample: Arsenal vs Man Utd
############################################################################################
# """
# sample dataset to use as baseline
df = pd.read_csv('montecarlo/baseline_df.csv')

# sample team naming to use as baseline
home_team = 'Arsenal'
away_team = 'Manchester United'
# """

# Run the montecarlo simulations
def montecarlo_simulation(df, home_team, away_team):

    def simulate_game(df, home_team, away_team):
        # Filtering the dataset to keep two different dataframes from each team
        home_shots = df[df['team'] == home_team]
        away_shots = df[df['team'] == away_team]

        # Simulating Home Team goals
        # Defining a counter to count the number of goals
        home_goals = 0

        # Making sure we only execute if there are records (shots) in the dataframe
        # Some teams may have done so bad that did not shoot in the entire game. Rare but possible.
        if home_shots['xG'].shape[0] > 0:
            for shot in home_shots['xG']:

                # Sampling a random number from 0 and 1 following uniform distribution
                prob = np.random.random()

                # If the random number is less than the Expected Goals (xG) then it counts as a goal
                if prob < shot:
                    home_goals += 1

        # Repeat for away team
        away_goals = 0

        if away_shots['xG'].shape[0] >0:
            for shot in away_shots['xG']:
                prob = np.random.random()
                if prob < shot:
                    away_goals += 1

        return {'home_goals': home_goals, 'away_goals': away_goals}

    
    # Define the number of iterations
    k = 10000
    
    # H2H (Head to Head occurrences)
    home = 0
    draw = 0
    away = 0

    # O/U ocurrences
    o_2_5 = 0
    u_2_5 = 0

    # Creating a for loop to populate the H2H and O2.5 variables based on simulation
    for i in range(k):

        # Apply the simulation for iteration i
        simulation = simulate_game(df, home_team, away_team)

        # If statements to assign winner and O/u result
        if simulation['home_goals'] > simulation['away_goals']:
            home += 1
            if simulation['home_goals'] + simulation['away_goals'] > 2.5:
                o_2_5 += 1
            else:
                u_2_5 += 1
        elif simulation['home_goals'] < simulation['away_goals']:
            away += 1
            if simulation['home_goals'] + simulation['away_goals'] > 2.5:
                o_2_5 += 1
            else:
                u_2_5 += 1
        else:
            draw += 1
            if simulation['home_goals'] + simulation['away_goals'] > 2.5:
                o_2_5 += 1
            else:
                u_2_5 += 1

    # Calculating probabilities of each outcome
    home_prob = home / k
    draw_prob = draw / k
    away_prob = away / k

    o2_5_prob = o_2_5 / k 
    u2_5_prob = u_2_5 / k

    return {home_team: home_prob, 'Draw': draw_prob, away_team: away_prob, '+2.5':o2_5_prob, '-2.5':u2_5_prob}


# Running the montecarlo function
predictions_output = montecarlo_simulation(df, home_team, away_team)


############################################################################################
## Section 3 - Viz Generation | Statsbomb as Reference
############################################################################################
# color palette:
main_background = '#ffffff'
widget_background = '#ffffff'
primary_text = '#223459'
home_color = '#0F3B99'
away_color = '#5886E9'
goal_tickmark_color = '#B45082'

# ------------- xG Flowchart ---------------------
# Creating dummy data for minute axis (min 0 to 100)
teams = df['team'].unique()
dummy_data = pd.DataFrame({
    'x': [0] * len(teams) * 2,  # Dummy values for x-coordinate
    'y': [0] * len(teams) * 2,  # Dummy values for y-coordinate
    'xG': [0] * len(teams) * 2,  # No xG for dummy rows
    'result': ['Dummy'] * len(teams) * 2,  # Dummy result
    'team': [team for team in teams for _ in range(2)],
    'minute': [0, 100] * len(teams),
    'player': [''] * len(teams) * 2  # No player for dummy rows
})

df = pd.concat([df, dummy_data], ignore_index=True)
df = df.sort_values(by=['team', 'minute'])

df['cumulative_xG'] = df.groupby('team')['xG'].cumsum()

# Plotting xG flowchart
fig, ax = plt.subplots(figsize=(6.5, 3.5))  # Create figure and axis

# Set background of the axis to light gray
ax.set_facecolor(widget_background)

colors = [home_color, away_color]  
for i, (team, group) in enumerate(df.groupby('team')):
    ax.plot(group['minute'], group['cumulative_xG'], label=team, marker='x', color=colors[i % len(colors)])

#for team, group in df.groupby('team'):
#    ax.plot(group['minute'], group['cumulative_xG'], label=team, marker='x')

# Overlay markers for goals
goal_shots = df[df['result'] == 'Goal']
ax.scatter(goal_shots['minute'], goal_shots['cumulative_xG'], color=goal_tickmark_color, label='Goal', zorder=5, edgecolor='black', s=50)

# Annotate with player names for goals
for _, row in goal_shots.iterrows():
    ax.text(row['minute'] - 18.5, row['cumulative_xG'], row['player'], fontsize=7, color='black', va='bottom', ha='left')

# Adding titles and labels
ax.set_title(f'{home_team} vs {away_team} xG Flowchart', fontsize=8)
ax.set_xlabel('Minute', fontsize=6)
ax.set_ylabel('Cumulative xG', fontsize=6)
ax.legend(fontsize=6)
ax.grid(True, linestyle='--', alpha=0.6)

# Smaller tick labels
ax.tick_params(axis='both', which='major', labelsize=5)  # Adjust tick label size
ax.set_xticks(range(0, 105, 5))  # Set ticks from 0 to 100 with a step of 5

# Interpolation of datasets for fillbetween()
# Do it at the end!

plt.tight_layout()
# Save figure with blue outer background and light gray inner axis background
plt.savefig('montecarlo/Figures/xGFlowchart.png', facecolor=widget_background)


# ------------- Montecarlo Bars ---------------------

# H2H
# Extract probabilities
labels = list(predictions_output.keys())[:3]
probs = list(predictions_output.values())[:3]

# Colors matching reference
colors = [home_color, '#d3d3d3', away_color, 'black', 'blue']  # Blue, Gray, Yellow -> CHANGE COLOR PALETTE

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(2, 0.5))  # Compact size for embedding

# Horizontal bars
left_pos = 0
for idx, prob in enumerate(probs):
    ax.barh(0, prob, left=left_pos, color=colors[idx])
    left_pos += prob

# Adjust aesthetics
ax.axis('off')  # Hide axes
ax.set_xlim(0, 1)  # Probabilities add up to 1
plt.tight_layout()

# Let's add the text in the PDF Canvas straight
# Add probability text underneath the bar chart
# prob_text = "\n".join([f"{label} (%): {prob:.0%}" for label, prob in predictions_output.items()])
# fig.text(0.5, -0.2, prob_text, ha='center', va='center', fontsize=5)  # Adjust position as needed

# Display the chart

plt.tight_layout()
plt.savefig('montecarlo/Figures/H2HMontecarlo.png', facecolor=widget_background)
#plt.show()

# O/U
# Extract probabilities
labels = list(predictions_output.keys())[3:]
probs = list(predictions_output.values())[3:]

# Colors matching reference
colors = [home_color, '#d3d3d3', away_color, 'black', 'blue']  # Blue, Gray, Yellow -> CHANGE COLOR PALETTE

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(2, 0.5))  # Compact size for embedding

# Horizontal bars
left_pos = 0
for idx, prob in enumerate(probs):
    ax.barh(0, prob, left=left_pos, color=colors[idx])
    left_pos += prob

# Adjust aesthetics
ax.axis('off')  # Hide axes
ax.set_xlim(0, 1)  # Probabilities add up to 1
plt.tight_layout()

# Let's add the text in the PDF Canvas straight
# Add probability text underneath the bar chart
# prob_text = "\n".join([f"{label} (%): {prob:.0%}" for label, prob in predictions_output.items()])
# fig.text(0.5, -0.2, prob_text, ha='center', va='center', fontsize=5)  # Adjust position as needed

# Display the chart

plt.tight_layout()
plt.savefig('montecarlo/Figures/OUMontecarlo.png', facecolor=widget_background)
#plt.show()

# ------ Logos plot -----

# Load the home and away team logos
home_logo = mpimg.imread(f'team_logos/{home_team}.png')
away_logo = mpimg.imread(f'team_logos/{away_team}.png')

# ------ Home Team Logo Plot -----
# Create a new figure for the home team logo
fig_home, ax_home = plt.subplots(figsize=(4, 4), facecolor=main_background)  # Adjust the size of the plot as needed
ax_home.imshow(home_logo, aspect='auto')
ax_home.set_facecolor(main_background)
ax_home.axis('off')  # Hide the axis

# Save the home team logo plot as a PNG file
fig_home.savefig('montecarlo/Figures/home_team_logo.png', bbox_inches='tight', transparent=True)
plt.close(fig_home)  # Close the figure to release memory

# ------ Away Team Logo Plot -----
# Create a new figure for the away team logo
fig_away, ax_away = plt.subplots(figsize=(4, 4), facecolor=main_background)  # Adjust the size of the plot as needed
ax_away.imshow(away_logo, aspect='auto')
ax_away.axis('off')  # Hide the axis

# Save the away team logo plot as a PNG file
fig_away.savefig('montecarlo/Figures/away_team_logo.png', bbox_inches='tight', transparent=True)
plt.close(fig_away)  # Close the figure to release memory

# ------------- CANVAS SECTION ---------------------
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from highlight_text import fig_text
from PIL import Image
import urllib
from mplsoccer import Radar, FontManager, grid
import numpy as np
import matplotlib.patheffects as path_effects
from highlight_text import ax_text, fig_text
from pathlib import Path
from urllib.request import urlopen
from io import BytesIO
import requests
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from PIL import Image
import pandas as pd
import urllib
from reportlab.lib.colors import HexColor 


# See Euro Report Canvas section
# Save previous figures in montecarlo/images/name.png
# Bring figures into the canvas, change colors, add logos, etc.
# save the report dynamically according to name in montecarlo/reports/home_team_away_team.pdf

# Prefixing the sizes of the canvas
width, height = 700, 600

# Creating the canvas in the appropiate directory
c = canvas.Canvas(f'montecarlo/Reports/{home_team} vs {away_team}.pdf', pagesize=(width, height))

# Setting up background color for white (#ffffff)
c.setFillColorRGB(1, 1, 1)  # RGB values for white
c.rect(0, 0, 780, 900, fill=True)  # Draw a filled rectangle with the white background

# xG Flowchart- Sample of Images Import
c.drawImage(f'montecarlo/Figures/xGFlowchart.png', 10, 30) #, width=270, height=220) - Avoid fixing sizes at it messes up image quality

# Montecarlo Graphs
c.drawImage(f'montecarlo/Figures/H2HMontecarlo.png', 35, 450)
c.drawImage(f'montecarlo/Figures/OUMontecarlo.png', 450, 450)

# Adding Logos Programatically
c.drawImage(f'montecarlo/Figures/home_team_logo.png', 500, 500, width = 90, height=90)
c.drawImage(f'montecarlo/Figures/away_team_logo.png', 585, 500, width = 90, height=90)

# Text
URL4 = 'https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Thin.ttf'
font_data = BytesIO(urlopen(URL4).read())
pdfmetrics.registerFont(TTFont('RobotoThin', font_data))

############ Title
c.setFont('RobotoThin', 30)
c.setFillColorRGB(1,1,1)  # Set text color to white
c.setFillColor(primary_text) # change color dynamically to match home_team
title = f"{home_team} vs {away_team}"
c.drawString(50, 550, title)

############ Subtitle - xG
# Filtering to sum the xG Data per team
home_ov_data = df[df['team']==home_team]
home_ov_data = round(home_ov_data['xG'].sum(),2)
away_ov_data = df[df['team']==away_team]
away_ov_data = round(away_ov_data['xG'].sum(),2)

c.setFont('RobotoThin', 15)
c.setFillColorRGB(1,1,1)  # Set text color to white
c.setFillColor(primary_text) # change color dynamically to match home_team
title = f"Expected Goals (xG)"
c.drawString(285, 500, title)

c.setFont('RobotoThin', 30)
c.setFillColorRGB(1,1,1)  # Set text color to white
c.setFillColor(primary_text) # change color dynamically to match home_team
title = f"{home_ov_data} - {away_ov_data}"
c.drawString(280, 470, title)



############ Subtitle - Score
home_score_data = len(df[(df['team']==home_team) & (df['result'] == 'Goal')])
away_score_data = len(df[(df['team']==away_team) & (df['result'] == 'Goal')])

c.setFont('RobotoThin', 30)
c.setFillColorRGB(1,1,1)  # Set text color to white
c.setFillColor(primary_text) # change color dynamically to match home_team
title = f"{home_score_data} - {away_score_data}"
c.drawString(50, 520, title)

# xG Over/Underperformance
h_xg_performance = home_score_data - home_ov_data
a_xg_performance = away_score_data - away_ov_data

# Plotting the over and underperfomance
# Define colors for positive and negative values
positive_color = HexColor("#00FF00")  # Green
negative_color = HexColor("#FF0000")  # Red

# Define symbols for upward and downward arrows
upward_arrow = u'\u2191'  # Unicode for upward arrow
downward_arrow = u'\u2193'  # Unicode for downward arrow

# Set font and size
c.setFont('RobotoThin', 20)

# Determine and set the color for h_xg_performance and add the arrow
if h_xg_performance >= 0:
    c.setFillColor(positive_color)
    h_arrow = upward_arrow  # Green upward arrow
else:
    c.setFillColor(negative_color)
    h_arrow = downward_arrow  # Red downward arrow

# Draw h_xg_performance with the arrow
h_title = f"{round(h_xg_performance, 2)} {h_arrow}"
c.drawString(280, 435, h_title)

# Determine and set the color for a_xg_performance and add the arrow
if a_xg_performance >= 0:
    c.setFillColor(positive_color)
    a_arrow = upward_arrow  # Green upward arrow
else:
    c.setFillColor(negative_color)
    a_arrow = downward_arrow  # Red downward arrow

# Draw a_xg_performance with the arrow
a_title = f"{round(a_xg_performance, 2)} {a_arrow}"
text_width = c.stringWidth(h_title, 'RobotoThin', 20)  # Calculate width of the first part
c.drawString(360, 435, a_title)

### Finalization of xG performance section

# Simulation Section Subtitles
c.setFont('RobotoThin', 12)
c.setFillColorRGB(1,1,1)  # Set text color to white
c.setFillColor(primary_text) # change color dynamically to match home_team
title = f"H2H Simulation"
c.drawString(55, 435, title)

c.setFont('RobotoThin', 12)
c.setFillColorRGB(1,1,1)  # Set text color to white
c.setFillColor(primary_text) # change color dynamically to match home_team
title = f"O/U Simulation"
c.drawString(465, 435, title)

############ Text - Probabilities for H2H
c.setFont('RobotoThin', 10)
c.setFillColorRGB(1,1,1)  # Set text color to white
c.setFillColor(primary_text) # change color dynamically to match home_team


# Example labels for the predictions
labels = [f"{home_team} wins:", "Draw:", f"{away_team} wins:"]
predictions = list(predictions_output.values())[:3]
formatted_predictions = [f"{label} {value*100:.2f}%" for label, value in zip(labels, predictions)]

# Set initial y-coordinate for text and step for line spacing
x, y = 55, 420
line_spacing = 12  # Adjust line spacing as needed

# Draw each formatted value on a new line
for prediction in formatted_predictions:
    c.drawString(x, y, prediction)
    y -= line_spacing  # Move down for the next line

############ Text Probabilities for O/U
c.setFont('RobotoThin', 10)
c.setFillColorRGB(1,1,1)  # Set text color to white
c.setFillColor(primary_text) # change color dynamically to match home_team


# Example labels for the predictions
labels = [f"Over 2.5 Goals: ", "Under 2.5 goals: "]
predictions = list(predictions_output.values())[3:]
formatted_predictions = [f"{label} {value*100:.2f}%" for label, value in zip(labels, predictions)]

# Set initial y-coordinate for text and step for line spacing
x, y = 466, 420
line_spacing = 12  # Adjust line spacing as needed

# Draw each formatted value on a new line
for prediction in formatted_predictions:
    c.drawString(x, y, prediction)
    y -= line_spacing  # Move down for the next line




# Save and close the PDF file
c.save()

# Running section
print(predictions_output)
print('Success')

# Dynamic string: c.drawString(50 + text_width, 435, a_title)
# Color palette from: https://brandguides.brandfolder.com/beautiful-dashboards/themes#starry-night
# To run python montecarlo/post_game_viz.py 'url'
# python montecarlo/post_game_viz.py 'https://understat.com/match/26733'

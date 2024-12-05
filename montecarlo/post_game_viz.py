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

# xG Flowchart
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
plt.figure(figsize=(12, 6))
for team, group in df.groupby('team'):
    plt.plot(group['minute'], group['cumulative_xG'], label=team, marker='o')

# Adding titles and labels
plt.title('xG Flowchart', fontsize=16)
plt.xlabel('Minute', fontsize=12)
plt.ylabel('Cumulative xG', fontsize=12)
plt.legend(title='Team')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


print(predictions_output)
print('Success')

# To run python montecarlo/post_game_viz.py 'url'
# Sample: python montecarlo/post_game_viz.py 'https://understat.com/match/26733'

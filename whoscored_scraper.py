import json
import time
import numpy as np
import pandas as pd
import seaborn as sns
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
## Section 1 - Scraping from the URL & Data Cleansing - Do the scraper as function and import it from module
############################################################################################

# Setting up the web driver
driver = webdriver.Chrome()

# Storing the url provided by the user
# Ensuring a url is provided as input/argument
if len(sys.argv) <2:
    print("Provide a valid url -> python montecarlo/post_game_viz.py 'url'")
    sys.exit(1)

# Storing the url     
whoscored_url = sys.argv[1]

# Let the driver access the url
driver.get(whoscored_url)

#------- Entering the Web Scraping Section ------- #
# Creating the soup element to access the HTML of the page
soup = BeautifulSoup(driver.page_source, 'html.parser')

# Selecting the block of code containing the data we are interested in
# Right click -> View Source Code -> Look for MatchCentreData
element = soup.select_one('script:-soup-contains("matchCentreData")')

# Extracting the dictionary of events data.
# There is a primary key 'id' and an eventId associated with each record
# Coming in JSON semi-structured format
matchdict = json.loads(element.text.split("matchCentreData: ")[1].split(',\n')[0])

# ------ Data Cleansing Section ------- #
# For more specifics on the data cleaning refer to Euros repository: https://github.com/enzovillafuerte/Euros2024-Reports/blob/main/main_file.ipynb
# Filtering using variable definition for only storing events dictionary data
match_events = matchdict['events']

# Converting JSON data into a pandas dataframe
df = pd.DataFrame(match_events)

# Dropping all rows that do not include a player ID
df.dropna(subset='playerId', inplace=True)

# Replacing all NaN values to None
df = df.where(pd.notnull(df), None)

# Renaming columns to ensure consistency and data integrity
df = df.rename(
    {
        'eventId': 'event_id',
        'expandedMinute': 'expanded_minute',
        'outcomeType': 'outcome_type',
        'isTouch': 'is_touch',
        'playerId': 'player_id',
        'teamId': 'team_id',
        'endX': 'end_x',
        'endY': 'end_y',
        'blockedX': 'blocked_x',
        'blockedY': 'blocked_y',
        'goalMouthZ': 'goal_mouth_z',
        'goalMouthY': 'goal_mouth_y',
        'isShot': 'is_shot',
        'cardType': 'card_type',
        'isGoal': 'is_goal'
    },
    axis=1
)

# Creating new columns from the dictionaries within the dataset variables (df['period', 'type'], etc)
df['period_display_name'] = df['period'].apply(lambda x: x['displayName'])  # The displayname variable is a key within the dictionary within the dataset (json)
df['type_display_name'] = df['type'].apply(lambda x: x['displayName'])
df['outcome_type_display_name'] = df['outcome_type'].apply(lambda x: x['displayName'])

# Creating a column of 'is_goal' for games without goals. 
# Otherwise it will create errors
if 'is_goal' not in df.columns:
    print('missing goals')
    df['is_goal'] = False
    
# Fixing for offside given
# Dropping rows that have the offisde given
df = df[~(df['type_display_name'] == "OffsideGiven")]

# Dropping the initial dictionary columns since we don't need them anymore
df.drop(columns = ['period', 'type', 'outcome_type'], inplace=True)


#Before moving forward, make sure you verify the columns of the dataset.
#Depending on the league, some variables will differ. 
#For example, the Spanish league has a 'card_type' variable, while the Eredivisie does not.

# Defining and keeping only desired columns
df = df[[ 
    'id', 'event_id', 'minute', 'second', 'team_id', 'player_id', 'x','y', 'end_x', 'end_y', 
    'qualifiers', 'is_touch', 'blocked_x', 'blocked_y', 'goal_mouth_z', 'goal_mouth_y', 'is_shot', 'is_goal', 'type_display_name', 'outcome_type_display_name',
    'period_display_name'
]]

# -- Variables not used: , 'card_type'

# Defining the types of each variable
df[['id', 'event_id', 'minute', 'team_id', 'player_id']] = df[['id', 'event_id', 'minute', 'team_id', 'player_id']].astype(int) 
df[['second', 'x', 'y', 'end_x', 'end_y']] = df[['second', 'x', 'y', 'end_x', 'end_y']].astype(float)
df[['is_shot', 'is_goal']] =df[['is_shot', 'is_goal']].astype(bool)

df['is_goal'] = df['is_goal'].fillna(False)
df['is_shot'] = df['is_shot'].fillna(False)

# -- Variables not used: , 'card_type'

# Loop for ensuring accuracy of the columns
# Pandas & Numpy treat the NaN differently, this will help to assign as None all of those values that previously didn't change from NaN to None
for column in df.columns:
    if df[column].dtype == np.float64 or df[column].dtype == np.float32:
        df[column] = np.where(
            np.isnan(df[column]),
            None,
            df[column]
        )

# ----- Appending Player Data Section ------
# Create a new variable to store the new coming information
# We then later will merge it with initial database
team_info = []

# Appending player information of Home team
team_info.append({
    'team_id': matchdict['home']['teamId'],
    'name': matchdict['home']['name'],
    'country_name': matchdict['home']['countryName'],
    'manager_name': matchdict['home']['managerName'],
    'players': matchdict['home']['players'],
})

# Appending player information of Away team
team_info.append({
    'team_id': matchdict['away']['teamId'],
    'name': matchdict['away']['name'],
    'country_name': matchdict['away']['countryName'],
    'manager_name': matchdict['away']['managerName'],
    'players': matchdict['away']['players'],
})


# Creating function for storing player information into new list
def insert_players(team_info):
    players = []
    
    for team in team_info:
        for player in team['players']:
            players.append({
                'player_id': player['playerId'],
                'team_id': team['team_id'],
                'shirt_no': player['shirtNo'],
                'name': player['name'],
                'position': player['position'],
                'age': player['age'],
                'MOTM': player['isManOfTheMatch'] # Might not have in Euro (Doble check later)
            })
    return players

# Creating function for storing team information and name
def insert_team(team_info):
    teams = []
    for team in team_info:
        teams.append({
          'team_id': team['team_id'],
          'team': team['name']
        })
    return teams

# Applying functions
players = insert_players(team_info)
teams = insert_team(team_info)

# Converting JSON data into a pandas dataframe
players_df = pd.DataFrame(players)
teams_df = pd.DataFrame(teams)

# ----- Merging events data with player data ------
# We are going to do the merge on player_id.
# SQL Schema -> primary key in Players table and foreign key in events
players_df = pd.merge(players_df, teams_df, on='team_id')
final_df = pd.merge(df, players_df, on='player_id')

# Sorting the df in ascending for minute and second
final_df = final_df.sort_values(by=['minute', 'second'], ascending=True)

# Resetting the index if needed
final_df = final_df.reset_index(drop=True)

# Setting up the name for the file
# Finding all positions of '-'
positions = [pos for pos, char in enumerate(whoscored_url) if char == '-']

# Getting the position of the second to last '-'
second_to_last_dash_position = positions[-2]

# Slicing the string from the second to last '-' to the end
new_variable = whoscored_url[second_to_last_dash_position + 1:]

# Saving the file for later in CSV
#final_df.to_csv(f'montecarlo/{new_variable}.csv', index=False)

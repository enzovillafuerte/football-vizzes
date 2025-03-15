import pandas as pd
import sys
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


############################################################################################
## Section 0 - User Input
############################################################################################

# Storing the url provided by the user
# Ensuring a url is provided as input/argument
if len(sys.argv) <2:
    print("Provide a valid url -> main.py 'url'")
    sys.exit(1)

# Storing the url     
whoscored_url = sys.argv[1]

############################################################################################
## Section 1 - Scraping from Whoscored with Selenium
############################################################################################

def scraping_whoscored(whoscored_url):

    ''' Explanation...'''

    # Setting up the driver
    driver = webdriver.Chrome()

    # Set up the Driver for the URL
    driver.get(whoscored_url)

    # Creating the soup element. We will get the HTML code of the page
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Selecting the block of code we are interested in, where the JSON data lies
    # Right click -> View Source Code -> Look for the MatchCentreData
    element = soup.select_one('script:-soup-contains("matchCentreData")')

    # Extracting the dictionary of events data.
    # There is a primary key 'id' and an eventId associated with each record
    # Coming in JSON format, preferred by web/app developers
    matchdict = json.loads(element.text.split("matchCentreData: ")[1].split(',\n')[0])

    # --------------- Data Cleaning -----------------------

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

    # Working til here

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

    # Defining and keeping only desired columns
    # ~~ Watch out here. Some leagues will have different columns
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

    # --------------- Appending Player Data -----------------------

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
                    'MOTM': player['isManOfTheMatch'] # Might not have in Libertadores (Doble check later)
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

     # --------------- Merging Events Data with Player Data -----------------------

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
    # positions = [pos for pos, char in enumerate(whoscored_url) if char == '-']

    # Getting the position of the second to last '-'
    # second_to_last_dash_position = positions[-2]

    # Slicing the string from the second to last '-' to the end
    # new_variable = whoscored_url[second_to_last_dash_position + 1:]

    # Saving the file for later in CSV
    # final_df.to_csv(f'Datasets/{new_variable}.csv', index=False)

    return final_df

# Maybe change the focus on Expected Threat, Momentum and Network Science metrics
# Most dangerous in terms of shots
# FRAME IT AS POSSESSION-ANALYSIS

def montecarlo_analysis(df):

    # No xG in the dataset unfortunately



    pass


def pass_network():

    pass

def individual_stats():

    # Player with more xG
    # Player with most passes
    # Most Central Player
    # Player with more xT

    pass

def report_generation():

    pass



def main():
    
    df = scraping_whoscored(whoscored_url)
    df.to_csv('whoscored-vizzes/sample.csv', index=False)
    print(df.head())


if __name__ == "__main__":

    main()

# Test
# python whoscored-vizzes/main.py 'https://www.whoscored.com/matches/1811539/live/south-america-copa-libertadores-2024-universitario-de-deportes-junior-fc'
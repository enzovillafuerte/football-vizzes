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

from pathlib import Path


from matplotlib.colors import LinearSegmentedColormap

from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.formatters import decimal_to_percent
from plottable.plots import circled_image # image
from plottable.plots import image


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

import urllib
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


import networkx as nx

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

    # Extracting Home and Away Team Names
    home_team_name = matchdict['home']['name']
    away_team_name = matchdict['away']['name']

    return home_team_name, away_team_name, final_df

# Maybe change the focus on Expected Threat, Momentum and Network Science metrics
# Most dangerous in terms of shots
# FRAME IT AS POSSESSION-ANALYSIS

def extract_goals(df, home_team, away_team):

    # This one is the one that works!!
    # lambda for creating new column named 'goal_scored'
    df['goal_scored'] = df.apply(lambda x: 1 if x['is_goal'] == True else 0, axis=1)

    all_teams = [home_team, away_team]
    goals_df = df.groupby('team', as_index=False).agg({'goal_scored': 'sum'}).set_index('team')

    # Ensure all teams are included, filling missing values with 0
    goals_df = goals_df.reindex(all_teams, fill_value=0).reset_index()

    return goals_df



def pass_network_networkx(df, list_of_teams):

    # Filter only to keep records that correspond to a team
    df = df[df['team'].isin(list_of_teams)]

    # Creating dataframe only of substitution records
    subs = df[df['type_display_name'] == 'SubstitutionOff']
    subs = subs['minute']
    first_sub = subs.min()

    # Keeping data before first substitution
    df = df[df['minute'] < first_sub].reset_index()

    # Creating new variables
    df['passer'] = df['name']
    df['receiver'] = df['name'].shift(-1)

    # Only successful passes
    df = df[df['type_display_name'] == 'Pass']
    df = df[df['outcome_type_display_name'] == 'Successful']

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges (passes) to the graph with weights
    for _, row in df.iterrows():
        passer = row['passer']
        receiver = row['receiver']
        if G.has_edge(passer, receiver):
            G[passer][receiver]['weight'] += 1
        else:
            G.add_edge(passer, receiver, weight=1)

    # Compute network metrics
    degree_centrality = nx.degree_centrality(G)  # How connected a player is
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')  # How often a player is in shortest paths
    clustering_coeff = nx.clustering(G.to_undirected())  # How well players form triangles
    triangles = nx.triangles(G.to_undirected())  # Number of triangles each player is involved in

    # Store results in a DataFrame
    metrics_df = pd.DataFrame({
        'name': list(G.nodes),
        'Degree Centrality': [degree_centrality[p] for p in G.nodes],
        'Betweenness Centrality': [betweenness_centrality[p] for p in G.nodes],
        'Clustering Coefficient': [clustering_coeff[p] for p in G.nodes],
        'Triangles': [triangles[p] for p in G.nodes]
    }).sort_values(by='Degree Centrality', ascending=False)

    # Plot the network
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # Positioning of nodes
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", edge_color="gray", font_size=8, width=[d['weight'] / 2 for (_, _, d) in G.edges(data=True)])
    
    plt.title("Pass Network")
    plt.savefig('whoscored-vizzes/figures_temp/networkx_graph.png')
    # plt.show()

    return metrics_df







def pass_network(df, main_color, marker_color, szobo_color, szobo_2_color, list_of_teams, nx_df):

    # Player of interest - Player with higest Degree Centrality
    szobo_player = nx_df['name'][0]

    # Filter to keep only records that correspond to a team of interest
    df = df[df['team'].isin(list_of_teams)]

    # Creating dataframe only of substitution records
    subs = df[df['type_display_name'] == 'SubstitutionOff']
    # Only keeping the minute variable of this new sub df
    subs = subs['minute']
    first_sub = subs.min()
    
    # Keeping the data only with records before first substitution
    df = df[df['minute'] < first_sub].reset_index()
    
    # Creating new variables
    df['passer'] = df['name']
    df['receiver'] = df['name'].shift(-1)
    
    # Only interested in successful passes
    df = df[df['type_display_name'] == 'Pass']
    df = df[df['outcome_type_display_name'] == 'Successful']
    
    # Calculating Average Locations of Players
    avg_locations = df.groupby('passer').agg({'x':['mean'], 'y':['mean', 'count']})
    avg_locations.columns = ['x', 'y', 'count']
    avg_locations
    
    # Passes between players (Count of Associations)
    pass_between = df.groupby(['passer', 'receiver']).id.count().reset_index()
    pass_between.rename({'id':'pass_count'}, axis='columns', inplace=True)
    
    # Merging DataFrames
    pass_between = pass_between.merge(avg_locations, left_on='passer',right_index=True)
    pass_between = pass_between.merge(avg_locations, left_on='receiver',right_index=True,suffixes=['','_end'])
    
    ################## Pitch Generation Section ##################
    
    # Setting the text color (labels and texts)
    rcParams['text.color'] = marker_color
    
    # Setting parameters for Plotting
    max_line_width = 10
    max_marker_size = 85
    szobo_marker_size = 275

    # Adjusting marker size based on the passer (Szoboslai)
    pass_between['marker_size'] = pass_between.apply(lambda row: szobo_marker_size if row['passer'] == szobo_player else max_marker_size, axis=1)
    pass_between['marker_color'] = pass_between.apply(lambda row: szobo_color if row['passer'] == szobo_player else marker_color, axis=1)

    # Setting up the width of the pass lines
    pass_between['width'] = (pass_between.pass_count / pass_between.pass_count.max() * max_line_width)

    # Setting color for pass connections involving Szoboslai
    # Setting up a Player B just in case for future modifications
    pass_between['line_color'] = pass_between.apply(lambda row: szobo_2_color if row['passer'] == szobo_player or row['passer'] == 'Player B' else marker_color, axis=1)

    rcParams['text.color'] = '#c7d5cc'
    
    # Plotting
    pitch = Pitch(pitch_type='opta', pitch_color=main_color, line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(9.5, 8), constrained_layout=True, tight_layout=False) # (8.5, 8.5
    fig.set_facecolor(main_color)

    # Drawing the pass lines (links/edges)
    pass_lines = pitch.lines(pass_between.x, pass_between.y,
                         pass_between.x_end, pass_between.y_end, lw=pass_between.width,
                         color=pass_between.line_color, zorder=1, ax=ax)

    # Drawing the nodes (players)
    nodes = pitch.scatter(pass_between.x, pass_between.y, s=pass_between.marker_size,
                      color=pass_between.marker_color, edgecolors=marker_color, linewidth=1.5, alpha=1, zorder=1, ax=ax)

    # Labeling the nodes with player name
    #for i, txt in enumerate(pass_between['passer']):
     #   ax.annotate(txt, (pass_between.x.iloc[i], pass_between.y.iloc[i]), color='#A3D3D3', fontsize=5, ha='center', va='bottom',xytext=(0, 12), textcoords='offset points')

    # Setting the title
    #ax_title = ax.set_title(f'Pass Network', fontsize=15) #, color='white')
    
    fig.savefig('whoscored-vizzes/figures_temp/pass_network.png', dpi=100, bbox_inches='tight') #dpi=300
    

def individual_stats():

    # Player with more xG
    # Player with most passes
    # Most Central Player
    # Player with more xT

    pass

def report_generation():

    pass



def main():
    
    # Colors
    colors_a = ['#192745', '#4B2583', '#b29400', '#d1d3d4'] # https://issuu.com/vistaprevia/docs/manual_de_identidad_de_marca_-_al
    colors_u = ['#FFFEF4', '#A6192E', '#000000', '#C6AA76', '#A7A8A9'] # https://universitario.pe/media/download/prensa/ID_Manual_Universitario_2020.pdf
    colors_sc = ['#E20A17', '#3ABFF0', '#FCDB18'] # FONDO BLANCO # https://issuu.com/andrebendezu777/docs/manual_identidad_sc_bendezu_andre

    # List of tems of interest
    list_of_teams = ['Universitario de Deportes', 'Alianza Lima', 'Sporting Cristal']


    # ------------------ Scrapping Function ---------------------
    # home_team, away_team, df = scraping_whoscored(whoscored_url)
    # df.to_csv('whoscored-vizzes/sample.csv', index=False)
    # print(df.head())
    df = pd.read_csv('whoscored-vizzes/sample.csv')

    # ------------------ Network Science Function ---------------------
    networkx_df = pass_network_networkx(df, list_of_teams)
    
    # ------------------ Pass Network Function ---------------------
    pass_network(df, colors_u[0] , colors_u[1] , colors_u[3] , colors_u[4], list_of_teams, networkx_df)

    


if __name__ == "__main__":

    main()

# Test
# python whoscored-vizzes/main.py 'https://www.whoscored.com/matches/1811539/live/south-america-copa-libertadores-2024-universitario-de-deportes-junior-fc'
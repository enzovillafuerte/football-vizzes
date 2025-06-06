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
import matplotlib

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
#from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
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

# Mapping
mapping = {
    
    'Rodrigo Ureña':'R. Ureña', 
    'Matías Di Benedetto':'Di Benedetto', 
    'Williams Riveros':'William Riveros',
    'Aldo Corzo': 'Aldo Corzo', 
    'Andy Polo':'Andy Polo' , 
    'Martín Pérez Guedes': 'Perez Guedes', 
    'Jairo Concha': 'J. Concha',
    #'Segundo Portocarrero', 
    'Édison Flores':'E. Flores', 
    'Sebastián Britos': 'S. Britos',
    'Jorge Murrugarra': 'Murrugarra', 
    'Álex Valera':'A. Valera'
    
}

############################################################################################
## Section 1 - Scraping from Whoscored with Selenium
############################################################################################

def scraping_whoscored(whoscored_url, mapping):

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

    # Mapping to rename players
    final_df['name'] = final_df['name'].replace(mapping) 

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

    metrics_df[['Betweenness Centrality', 'Clustering Coefficient']] = metrics_df[['Betweenness Centrality', 'Clustering Coefficient']].round(2)

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
    


def xT_grid(df, list_of_teams):

    # Keeping only specific Team records
    df = df.loc[(df['team'].isin(list_of_teams)) & (df['type_display_name']=='Pass') & (df['outcome_type_display_name']=='Successful')]

    # xT Grid
    xT = np.array([
        [0.006383, 0.007796, 0.008449, 0.009777, 0.011263, 0.012483, 0.014736, 0.017451, 0.021221, 0.027563, 0.034851, 0.037926],
        [0.007501, 0.008786, 0.009424, 0.010595, 0.012147, 0.013845, 0.016118, 0.018703, 0.024015, 0.029533, 0.040670, 0.046477],
        [0.008880, 0.009777, 0.010013, 0.011105, 0.012692, 0.014291, 0.016856, 0.019351, 0.024122, 0.028552, 0.054911, 0.064426],
        [0.009411, 0.010827, 0.010165, 0.011324, 0.012626, 0.014846, 0.016895, 0.019971, 0.023851, 0.035113, 0.108051, 0.257454],
        [0.009411, 0.010827, 0.010165, 0.011324, 0.012626, 0.014846, 0.016895, 0.019971, 0.023851, 0.035113, 0.108051, 0.257454],
        [0.008880, 0.009777, 0.010013, 0.011105, 0.012692, 0.014291, 0.016856, 0.019351, 0.024122, 0.028552, 0.054911, 0.064426],
        [0.007501, 0.008786, 0.009424, 0.010595, 0.012147, 0.013845, 0.016118, 0.018703, 0.024015, 0.029533, 0.040670, 0.046477],
        [0.006383, 0.007796, 0.008449, 0.009777, 0.011263, 0.012483, 0.014736, 0.017451, 0.021221, 0.027563, 0.034851, 0.037926]
    ])

    xT_rows, xT_cols = xT.shape
    
    # Categorizing each record in a bin for starting point and ending point
    df['x1_bin'] = pd.cut(df['x'], bins = xT_cols, labels=False)
    df['y1_bin'] = pd.cut(df['y'], bins = xT_rows, labels=False)

    df['x2_bin'] = pd.cut(df['end_x'], bins = xT_cols, labels=False)
    df['y2_bin'] = pd.cut(df['end_y'], bins = xT_rows, labels=False)
    
    # Defining start zone and end zone values of passes (kinda like x,y coordinates in a map plot)
    df['start_zone_value'] = df[['x1_bin', 'y1_bin']].apply(lambda x: xT[x[1]][x[0]],axis=1)
    df['end_zone_value'] = df[['x2_bin', 'y2_bin']].apply(lambda x: xT[x[1]][x[0]],axis=1)

        # The difference of end_zone and start_zone is the expected threat value for the action (pass) - not accounting for dribble xT here
    # Value can be negative or positive (progressive)
    df['Pass xT'] = df['end_zone_value'] - df['start_zone_value']
    # Progressive xT measures progressive passes
    df['Progressive xT'] = 0.0  # Initialize with zeros instead of empty string
    
    # Replace the while loop with a more pandas-friendly approach
    df['Progressive xT'] = df['Pass xT'].apply(lambda x: x if x > 0 else 0.0)

    # xT chart
    xT_gb = df.groupby(by='name', as_index=False).agg({'Pass xT': 'sum', 'Progressive xT': 'sum'}).sort_values(by='Progressive xT', ascending=False).reset_index(drop=True)

    xT_gb[['Pass xT', 'Progressive xT']] = xT_gb[['Pass xT', 'Progressive xT']].astype(float).round(2)

    #xT_gb.sort_values(by='Progressi
    xT_gb[['Pass xT', 'Progressive xT']] = xT_gb[['Pass xT', 'Progressive xT']].round(2) 

    return xT_gb


def general_pass_actions(df, list_of_teams):

    passes_df = df[(df['type_display_name'] == 'Pass') & df['team'].isin(list_of_teams)] 

    passes_gp = passes_df.groupby(by='name', as_index=False).agg({'type_display_name':'count'}).sort_values(by='type_display_name', ascending=False).reset_index(drop=True)

    succ_pass_df = passes_df[passes_df['outcome_type_display_name'] == 'Successful']
    succ_passes_gp = succ_pass_df.groupby(by='name', as_index=False).agg({'type_display_name':'count'}).sort_values(by='type_display_name', ascending=False).reset_index(drop=True)

    merged_passes = pd.merge(passes_gp, succ_passes_gp, how='inner', on='name')

    merged_passes.rename(columns={'type_display_name_x' : 'Total Passes' , 'type_display_name_y': 'Successful Passes'}, inplace=True)

    merged_passes['Succ (%)'] = round(merged_passes['Successful Passes'] / merged_passes['Total Passes'] * 100,2)


    # merged_passes['Succ (%)'] = merged_passes['Succ (%)'].apply(lambda x: f"{x:.1f}%")

    # Keeping only record with more than 10 passes
    merged_passes = merged_passes[merged_passes['Total Passes'] >= 10]

    return merged_passes


def beaut_table_passes(df, main_color):

    overall = df # Creating a copy for the cmap
    df = df.sort_values(by='Succ (%)', ascending=False)
    
    # Keep the TOP 3 Passers
    df = df.head(3)
    # Reset the index (to avoid any existing index) and then add 1 to the index values
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    
    # Read in the logos
    df['badge'] = df['name'].apply(
        lambda x: f"whoscored-vizzes/players_png/{x}.png"
    )
    
    #define color map
    # We should change the color map depending on the team we are analyzing
    cmap = LinearSegmentedColormap.from_list(
    name="bugw", colors=["#ffffff", "#f2fbd2", "#c9ecb4", "#93d3ab", "#35b0ab"], N=256
    )
    
    df = df[['badge', 'name', 'Total Passes', 'Successful Passes', 'Succ (%)']]
    
    col_defs = (
        
        [
            ColumnDefinition(
                name='name',
                title='Player',
                textprops={"ha": "left", "weight": "bold", 'color':'black'},
                width=2.5,
                
            ),
            
            ColumnDefinition(
                name='index',
                title="",
                textprops={"ha": "left", 'color':'black'},
                width=0.25,
            ),
            
            ColumnDefinition(
                name='Total Passes',
                textprops={"ha": "left", 'color':'black'},
                width=2,
            ),
            
            ColumnDefinition(
                name='Successful Passes',
                title='Succ. Passes',
                textprops={"ha": "left", 'color':'black'},
                width=2,
            ),
            

            
            ColumnDefinition(
                name="Succ (%)",
                width=2,
                textprops={
                    "ha":"center",
                    "bbox": {"boxstyle": "circle", "pad": 0.45},
                    'color':'black',
                },
                cmap=normed_cmap(overall["Succ (%)"], cmap=matplotlib.cm.PiYG, num_stds=2.5),
            ),
            
            ColumnDefinition(
                name="badge",
                title="",
                textprops={"ha": "center", "va": "center"}, #, 'color': bg_color},
                width=0.6,
                plot_fn=image,
            )
        ]
        
        
    )
    
    # Graph
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["savefig.bbox"] = "tight"
    
    fig, ax = plt.subplots(figsize=(4.5, 2))
    
    # **Set Background Color**
    bg_color = main_color  # Change this to any color you want
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    table = Table(
        df,
        column_definitions=col_defs,
        row_dividers=True,
        footer_divider=True,
        ax=ax,
        textprops={"fontsize": 6},
        row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
        col_label_divider_kw={"linewidth": 0.1, "linestyle": "-"},
        column_border_kw={"linewidth": 0.4, "linestyle": "-"},
    )#.autoset_fontcolors(colnames=["xG", "xGA","xPTS", "xG per Game", "xGA per Game"])
    
    fig.savefig(f"whoscored-vizzes/figures_temp/table_passes.png", facecolor=ax.get_facecolor(), dpi=200)
    
    return df


def beaut_table_xT(df, main_color):

    overall = df 
    # df = df.sort_values(by='', ascending=False)
    
    df = df.head(3)
    # Reset the index (to avoid any existing index) and then add 1 to the index values
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    
    # Read in the logos
    df['badge'] = df['name'].apply(
        lambda x: f"whoscored-vizzes/players_png/{x}.png"
    )
    
    #define color map
    cmap = LinearSegmentedColormap.from_list(
    name="bugw", colors=["#ffffff", "#f2fbd2", "#c9ecb4", "#93d3ab", "#35b0ab"], N=256
    )
    
    df = df[['badge', 'name', 'Progressive xT']]
    
    # df = df[['badge', 'name', 'Total Passes', 'Successful Passes', 'Succ (%)']]
    
    col_defs = (
        
        [
            ColumnDefinition(
                name='name',
                title='Player',
                textprops={"ha": "left", "weight": "bold", 'color':'black'},
                width=0.5,
                
            ),
            
            ColumnDefinition(
                name='index',
                title="",
                textprops={"ha": "left", 'color':'black'},
                width=0.05,
            ),

            
            ColumnDefinition(
                name="Progressive xT",
                width=2,
                textprops={
                    "ha":"center",
                    "bbox": {"boxstyle": "roundtooth", "pad": 0.85},
                    'color':'black',
                },
                cmap=normed_cmap(overall["Progressive xT"], cmap=matplotlib.cm.PiYG, num_stds=3.5),
            ),
            
            ColumnDefinition(
                name="badge",
                title="",
                textprops={"ha": "center", "va": "center"}, #, 'color': bg_color},
                width=0.6,
                plot_fn=image,
            )
        ]
        
        
    )
    
    # Graph
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["savefig.bbox"] = "tight"
    
    fig, ax = plt.subplots(figsize=(2.5, 2))
    
    # **Set Background Color**
    bg_color = main_color  # Change this to any color you want
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    table = Table(
        df,
        column_definitions=col_defs,
        row_dividers=True,
        footer_divider=True,
        ax=ax,
        textprops={"fontsize": 6},
        row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
        col_label_divider_kw={"linewidth": 0.1, "linestyle": "-"},
        column_border_kw={"linewidth": 0.4, "linestyle": "-"},
    )#.autoset_fontcolors(colnames=["xG", "xGA","xPTS", "xG per Game", "xGA per Game"])
    
    fig.savefig(f"whoscored-vizzes/figures_temp/table_xT.png", facecolor=ax.get_facecolor(), dpi=200)
    
    return df


def beaut_table_network(df, metric, main_color):

    overall = df 
    df = df.sort_values(by=f'{metric}', ascending=False)
    
    df = df.head(1)
    # Reset the index (to avoid any existing index) and then add 1 to the index values
    
    # Read in the logos
    df['badge'] = df['name'].apply(
        lambda x: f"whoscored-vizzes/players_png/{x}.png"
    )
    
    #define color map
    cmap = LinearSegmentedColormap.from_list(
    name="bugw", colors=["#ffffff", "#f2fbd2", "#c9ecb4", "#93d3ab", "#35b0ab"], N=256
    )
    
    df = df[['badge', 'name', metric]]
    
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    
    # df = df[['badge', 'name', 'Total Passes', 'Successful Passes', 'Succ (%)']]
    
    col_defs = (
        
        [
            ColumnDefinition(
                name='name',
                title='Player',
                textprops={"ha": "left", "weight": "bold", 'color':'black'},
                width=0.5,
                
            ),
            
            ColumnDefinition(
                name='index',
                title="",
                textprops={"ha": "left", 'color':'black'},
                width=0.1,
            ),

            
            ColumnDefinition(
                name=f"{metric}",
                width=2,
                textprops={
                    "ha":"center",
                    "bbox": {"boxstyle": "darrow", "pad": 0.85},
                    'color':'black',
                },
                cmap=normed_cmap(overall[f"{metric}"], cmap=matplotlib.cm.PiYG, num_stds=3.5),
            ),
            
            ColumnDefinition(
                name="badge",
                title="",
                textprops={"ha": "center", "va": "center"}, #, 'color': bg_color},
                width=0.6,
                plot_fn=image,
            )
        ]
        
        
    )
    
    # Graph
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["savefig.bbox"] = "tight"
    
    fig, ax = plt.subplots(figsize=(2.5, 1))
    
    # **Set Background Color**
    bg_color = main_color  # Change this to any color you want
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    table = Table(
        df,
        column_definitions=col_defs,
        row_dividers=True,
        footer_divider=True,
        ax=ax,
        textprops={"fontsize": 6},
        row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
        col_label_divider_kw={"linewidth": 0.1, "linestyle": "-"},
        column_border_kw={"linewidth": 0.4, "linestyle": "-"},
    )#.autoset_fontcolors(colnames=["xG", "xGA","xPTS", "xG per Game", "xGA per Game"])

    fig.savefig(f"whoscored-vizzes/figures_temp/table_nwx_{metric}.png", facecolor=ax.get_facecolor(), dpi=200)
    
    return df

def create_logo_figure(home_team, away_team, main_color):

    import matplotlib.image as mpimg
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase import pdfmetrics

    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=300)  # Adjust figure size
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def add_logo(team, x_center):
        """Loads a team logo and places it centered at x_center."""
        logo_path = f"whoscored-vizzes/teams_png/{team}.png"
        if os.path.exists(logo_path):
            logo = mpimg.imread(logo_path)

            # Get image aspect ratio
            height, width, _ = logo.shape
            aspect_ratio = width / height

            # Define a square bounding box (0.35 width, 0.35 height)
            logo_width = 0.2
            logo_height = 0.2 / aspect_ratio  # Maintain aspect ratio

            # Center the logo horizontally
            x_left = x_center - (logo_width / 2)
            y_bottom = 0.325  # Center it vertically

            ax.imshow(logo, extent=[x_left, x_left + logo_width, y_bottom, y_bottom + logo_height])
        else:
            print(f"Missing logo: {team}.png")

    # Add both team logos
    add_logo(home_team, x_center=0.25)  # Left side
    add_logo(away_team, x_center=0.48)  # Right side

    ax.set_facecolor(main_color)  # Transparent background
    plt.axis("off")

    plt.savefig("whoscored-vizzes/figures_temp/logos_combined.png", bbox_inches="tight", pad_inches=0,facecolor=ax.get_facecolor())
    plt.close()


def goal_generation(df, home_team, away_team):

    # lambda for creating new column named 'goal_scored'
    df['goal_scored'] = df.apply(lambda x: 1 if x['is_goal'] == True else 0, axis=1)

    all_teams = [home_team, away_team]
    goals_df = df.groupby('team', as_index=False).agg({'goal_scored': 'sum'}).set_index('team')

    # Ensure all teams are included, filling missing values with 0
    goals_df = goals_df.reindex(all_teams, fill_value=0).reset_index()
    
    return goals_df







def report_generation(home_team, away_team, goals_df, main_color):

    # Importing all needed libraries here to avoid conflicts
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
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase import pdfmetrics



    def hex_to_rgb(hex):
        hex = hex.lstrip('#')  # Remove leading '#'
        r = int(hex[0:2], 16) / 255.0
        g = int(hex[2:4], 16) / 255.0
        b = int(hex[4:6], 16) / 255.0
        return r, g, b
    

    width, height = 1300, 1300
    c = canvas.Canvas(f"whoscored-vizzes/Reports/{home_team} vs {away_team} Report.pdf", pagesize=(width, height))
    
    #home_team = home_team
    #away_team = away_team
    
    home_score = goals_df[goals_df['team'] == home_team]['goal_scored'].tolist()
    away_score = goals_df[goals_df['team'] == home_team]['goal_scored'].tolist()
    
    

    # **REGISTER DEJAVU SANS FONT**
    pdfmetrics.registerFont(TTFont('DejaVuSans', 'whoscored-vizzes/DejaVuSans.ttf'))
    #pdfmetrics.registerFont(TTFont('GothamBold', 'GothamBold.ttf'))
    
    # Set font
    c.setFont("DejaVuSans", 38)  # Title font size
    #c.setFont("GothamBold", 56)  # Title font size

    ''' BACKGROUND COLOR '''
    c.setFillColorRGB(*hex_to_rgb(main_color))  # Ensure correct hex color
    c.rect(0, 0, width, height, fill=True)
    
    ''' LOGOS '''
    c.drawImage(f"whoscored-vizzes/figures_temp/logos_combined.png", 950,970)
    
    ''' ADD TITLE '''
    c.setFillColorRGB(0, 0, 0)  # White color for title text

    c.drawString(30, 1180, f"{home_team} {home_score} vs {away_team} {away_score}")
    



    

    ''' PASS NETWORK GRAPH '''
    c.drawImage('whoscored-vizzes/figures_temp/pass_network.png', 20, 450)  # Adjust positioning as needed

    # Add tables
    c.drawImage('whoscored-vizzes/figures_temp/table_passes.png', 20, 30)
    c.drawImage('whoscored-vizzes/figures_temp/table_xT.png', 825, 30)
    c.drawImage('whoscored-vizzes/figures_temp/table_nwx_Betweenness Centrality.png', 825, 440)
    c.drawImage('whoscored-vizzes/figures_temp/table_nwx_Clustering Coefficient.png', 825, 620)
    c.drawImage('whoscored-vizzes/figures_temp/table_nwx_Degree Centrality.png', 825, 800)
    
    ''' ADD SUBTITLE '''
    c.setFont("DejaVuSans", 28)  # Subtitle font size
    c.setFillColorRGB(0, 0, 0)  # White text / Black
    c.drawString(40, 352, "Overall Pass Stats")  # Subtitle positioned manually
    c.drawString(835, 352, "Most Dangerous (xT)")  # Subtitle positioned manually
    c.drawString(835, 998, "Network Science")  # Subtitle positioned manually
    
    c.drawString(30, 1230, "Copa Libertadores: Posession Match Report")  # Subtitle positioned manually
    
    c.setFont("DejaVuSans", 18)  # Subtitle font size
    c.drawString(30, 1030, "* xT: Expected Threat")  # Subtitle positioned manually
    #c.drawString(30, 1230, "*")  # Subtitle positioned manually
    #c.drawString(30, 1230, "*")  # Subtitle positioned manually
    #c.drawString(30, 1230, "*")  # Subtitle positioned manually

    # Save the report
    c.save()




def main():
    
    # Colors
    colors_a = ['#192745', '#4B2583', '#b29400', '#d1d3d4'] # https://issuu.com/vistaprevia/docs/manual_de_identidad_de_marca_-_al
    colors_u = ['#FFFEF4', '#A6192E', '#000000', '#C6AA76', '#A7A8A9'] # https://universitario.pe/media/download/prensa/ID_Manual_Universitario_2020.pdf
    colors_sc = ['#E20A17', '#3ABFF0', '#FCDB18'] # FONDO BLANCO # https://issuu.com/andrebendezu777/docs/manual_identidad_sc_bendezu_andre

    # List of tems of interest
    list_of_teams = ['Universitario de Deportes', 'Alianza Lima', 'Sporting Cristal']


    # ------------------ Scrapping Function ---------------------
    # home_team, away_team, df = scraping_whoscored(whoscored_url, mapping)
    # df.to_csv('whoscored-vizzes/sample.csv', index=False)
    # print(df.head())
    df = pd.read_csv('whoscored-vizzes/sample.csv')

    # ------------------ Extracting Score and Goal ---------------------
    # goals_df = goal_generation(df, home_team, away_team)
    goals_df = goal_generation(df, 'Universitario de Deportes', 'Junior FC')

    # ------------------ Network Science Function ---------------------
    networkx_df = pass_network_networkx(df, list_of_teams)
    
    # ------------------ Pass Network Function ---------------------
    pass_network(df, colors_u[0] , colors_u[1] , colors_u[3] , colors_u[4], list_of_teams, networkx_df)

    # ------------------ Expected Threat Function ---------------------
    xT_gb = xT_grid(df, list_of_teams)

    # ------------------ General Pass Stats Function ---------------------
    passes_df = general_pass_actions(df, list_of_teams)

    # ------------------ Beautiful Tables Generation ---------------------
    beaut_table_passes(passes_df, colors_u[0])
    beaut_table_xT(xT_gb, colors_u[0])
    beaut_table_network(networkx_df, 'Degree Centrality', colors_u[0])
    beaut_table_network(networkx_df, 'Betweenness Centrality', colors_u[0])
    beaut_table_network(networkx_df, 'Clustering Coefficient', colors_u[0])

    # ------------------ Logos Generation ---------------------
    # create_logo_figure(home_team, away_team, colors_u[0]) # This should work, commeting out for testing
    create_logo_figure('Universitario de Deportes', 'Junior FC', colors_u[0])

    # ------------------ Report Generation ---------------------
    # report_generation(home_team, away_team, goals_df, colors_u[0]) # This should work, commeting out for testing
    report_generation('Universitario de Deportes', 'Junior FC', goals_df, colors_u[0])





if __name__ == "__main__":

    main()

# Test
# python whoscored-vizzes/main.py 'https://www.whoscored.com/matches/1811539/live/south-america-copa-libertadores-2024-universitario-de-deportes-junior-fc'
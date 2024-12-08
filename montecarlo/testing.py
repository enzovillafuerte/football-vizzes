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
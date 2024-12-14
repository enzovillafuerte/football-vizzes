import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import requests 
from bs4 import BeautifulSoup
import seaborn as sns
import json
import time 


##################################################################
################### SCRAPER SECTION #############################
##################################################################

"""
# Storing the url provided by the user
# Ensuring a url is provided as input/argument
if len(sys.argv) <2:
    print("Provide a valid url -> python montecarlo/post_game_viz.py 'url'")
    sys.exit(1)

# Storing the url     
url = sys.argv[1]
"""


base_url =  'https://understat.com/team/' # base url
team = 'Barcelona' # coming from input
year = '2024' # Year of the analysis, iterate over a loop of list if needed

# Consolidation of the url 
url = f'{base_url}{team}/{year}'




# Begin scraping
res = requests.get(url)
soup = BeautifulSoup(res.content, 'lxml')
scripts = soup.find_all('script')

# We are interested in the vardatesData section as the information is stored there
# Index = [1]
strings = scripts[1].string

# strip unnecessary symbols and get only JSON data 
ind_start = strings.index("('")+2 
ind_end = strings.index("')") 
json_data = strings[ind_start:ind_end] 
json_data = json_data.encode('utf8').decode('unicode_escape')

# convert string to json format
data = json.loads(json_data)

# Create a dataframe
# Normalize the data and extract nested dictionaries into separate columns
df = pd.json_normalize(
    data,
    sep='_',
    meta=['id', 'isResult', 'side', 'datetime', 'result'],
    record_prefix=None
)


##################################################################
################### VIZ SECTION #############################
##################################################################

# Used the following repo as guide: https://github.com/enzo23isco/xGRolling/blob/main/xG%20Rolling%20Plot%20-%20FCB.ipynb

df = df[df['isResult'] == True] # keeping only matches that already happened

# Modify the dataframe to get two new columns, team_xG and team_xGA
df['team_xG'] = df.apply(
    lambda row: row['xG_h'] if row['h_title'] == team else row['xG_a'], axis=1
)

df['team_xGA'] = df.apply(
    lambda row: row['xG_a'] if row['h_title'] == team else row['xG_h'], axis=1
)

# Further manipulation before the time series


# xG concedede and xG created
#Y_for = df[df["variable"] == "xG_for"].reset_index(drop=True)["value"]
#Y_ag = df[df["variable"] == "xG_ag"].reset_index(drop=True)["value"]
#X = pd.Series(range(len(Y_for)))
Y_for = df['team_xG']
Y_ag = df['team_xGA']
X = pd.Series(range(len(Y_for)))

# Compute the rolling average (min_periods is used for the partial average)
# 3 game rolling average

Y_for = Y_for.rolling(window=3, min_periods = 0).mean()
Y_ag = Y_ag.rolling(window=3, min_periods = 0).mean()

# Plot the data
fig = plt.figure(figsize=(4, 2.5), dpi = 200)
ax = plt.subplot(111)

ax.plot(X, Y_for, label = "xG created")
ax.plot(X, Y_ag, label = "xG conceded")

ax.legend()
plt.show()

print(df)
print('Success')

# df columns -> Index(['id', 'isResult', 'side', 'datetime', 'result', 'h_id', 'h_title',
       #'h_short_title', 'a_id', 'a_title', 'a_short_title', 'goals_h',
       #'goals_a', 'xG_h', 'xG_a', 'forecast_w', 'forecast_d', 'forecast_l'],
      #dtype='object')
# To run:
# python xg_moving_average/xg_moving_average.py
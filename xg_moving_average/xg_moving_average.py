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





print(df)
print('Success')

# To run:
# python xg_moving_average/xg_moving_average.py
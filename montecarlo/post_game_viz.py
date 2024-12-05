"""
Takes a url from whoscored and generates a montecarlo report
# Probabilities of each team winning based on xG
# Probabilities of a game having over 2.5 goals or 1.5 goals based on xG
# xG flowchart
# Shot map (Maybe) ~ Keep it simple first
# See statsbomb xG flowchart for reference, try to replicate
# Use scraping code from other projects
"""

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
## Section 2 - Montecarlo Simulations - work with the sample: Mallorca vs Barcelona
############################################################################################
# sample dataset to use as baseline
df = pd.read_csv('Mallorca-Barcelona.csv')

# Further data Cleansing to only keep shot_related data
df = df[df['']]

############################################################################################
## Section 3 - Viz Generation
############################################################################################

print(whoscored_url)
print('Success')

# To run python montecarlo/post_game_viz.py 'url'
# Sample: python montecarlo/post_game_viz.py 'https://www.whoscored.com/Matches/1821600/Live/Spain-LaLiga-2024-2025-Mallorca-Barcelona'

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import requests 
from bs4 import BeautifulSoup
import seaborn as sns
import json
import time 
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
from highlight_text import fig_text
from PIL import Image
import urllib
import sys

##################################################################
################### SCRAPER SECTION #############################
##################################################################


# Storing the url provided by the user
# Ensuring a url is provided as input/argument
if len(sys.argv) <2:
    print("Provide a valid url -> python xg_moving_average/xg_moving_average.py 'team'")
    sys.exit(1)

# Storing the team    
team = sys.argv[1]


base_url =  'https://understat.com/team/' # base url
#team = 'Barcelona' # coming from input
#team='VfB_Stuttgart'
#year = '2024' # Year of the analysis, iterate over a loop of list if needed
#years_list = ['2020', '2021', '2022', '2023', '2024']
years_list = ['2021', '2022', '2023', '2024']
# Consolidation of the url 
#url = f'{base_url}{team}/{year}'


# Adding error handling in case the specific team wasn't in first division in the given year
def season_scraper(url):

    try:
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

        return df

    except Exception as e:
        print(f"Failed to scrape data for URL {url}: {e}")
        return pd.DataFrame()


# Consolidate data for all seasons defined
all_data = []

for year in years_list:
    url = f'{base_url}{team}/{year}' # consolidation of the url
    print(f'Scraping data for {team} in {year}')
    df = season_scraper(url) # Calling the function to scrape the data
    if not df.empty:
        df['year'] = year  # adding the year column. Potential redundnacy with datetime column
        all_data.append(df)

# Combine all DataFrames into one
if all_data:
    consolidated_df = pd.concat(all_data, ignore_index=True)
    print("Data successfully consolidated!")
else:
    consolidated_df = pd.DataFrame()  # Handle the case where no data was scraped
    print("No data was scraped.")


# Run the function
# Rename the dataset back to df for viz development
df = consolidated_df

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

Y_for = Y_for.rolling(window=6, min_periods = 0).mean()
Y_ag = Y_ag.rolling(window=6, min_periods = 0).mean()

#####
# Plot the data
fig = plt.figure(figsize=(4.5, 2.5), dpi = 200, facecolor = "#EFE9E6")
ax = plt.subplot(111, facecolor = "#EFE9E6") # 111

# Remove top & right spines and change the color.
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("grey")

# Set the grid
ax.grid(
    visible = True, 
    lw = 0.75,
    ls = ":",
    color = "lightgrey"
)

line_1 = ax.plot(X, Y_for, color = "#004D98", zorder = 4)
line_2 = ax.plot(X, Y_ag, color = "#A50044", zorder = 4)

ax.set_ylim(0)


# Season Annotation in Viz Section
# Adding vertical lines and annotations to separate seasons
season_boundaries = df.groupby('year').apply(lambda x: x.index[0]).tolist()

# Iterate over the season boundaries and annotate the plot
for i, boundary in enumerate(season_boundaries):
    ax.plot(
        [boundary, boundary],
        [ax.get_ylim()[0], ax.get_ylim()[1]],
        ls=":",
        lw=1.25,
        color="grey",
        zorder=2
    )
    # Add an annotation for the season
    ax.annotate(
        xy=(boundary, ax.get_ylim()[1] * 0.1),
        xytext=(15, 10),
        textcoords="offset points",
        text=f"{years_list[i]}",
        fontweight="bold",
        size=4,
        color="grey",
        ha="right"
    )



# Comment out this block optional and specific to some analysis:
"""

# Add a line to mark the division between seasons
ax.plot(
    [12,12], # 12th game into the season was Xavi's first game
    [ax.get_ylim()[0], ax.get_ylim()[1]],
    ls = ":",
    lw = 1.25,
    color = "grey",
    zorder = 2
)

# Annotation with data coordinates and offset points.
ax.annotate(
    xy = (12, .55),
    xytext = (20, 70),
    textcoords = "offset points",
    text = "Xavi's arrival",
    fontweight="bold",
    zorder = 5,
    size = 6,
    color = "grey",
    arrowprops=dict(
        arrowstyle="->", shrinkA=0, shrinkB=5, color="grey", linewidth=0.75,
        connectionstyle="angle3,angleA=100,angleB=-30"
    ) # Arrow to connect annotation
)
"""
# Add referencing text to el clasico
# Add text to the plot
"""
ax.annotate("Real Madrid 0-4 Barcelona", xy=(27, 0), xytext=(27, 4.2), color='black',
            fontsize=4.5, ha='center', va='bottom', 
            #bbox=dict(facecolor='none', edgecolor='grey', boxstyle='round'),
            verticalalignment='top')

"""

# Fill between
ax.fill_between(
    X, 
    Y_ag,
    Y_for, 
    where = Y_for >= Y_ag, 
    interpolate = True,
    alpha = 0.85,
    zorder = 3,
    color = line_1[0].get_color()
)

ax.fill_between(
    X, 
    Y_ag,
    Y_for, 
    where = Y_ag > Y_for, 
    interpolate = True,
    alpha = 0.85,
    color = line_2[0].get_color()
)

# Customize the ticks to match spine color and adjust label size.
ax.tick_params(
    color = "grey", 
    length = 5, 
    which = "major", 
    labelsize = 6,
    labelcolor = "grey",
    zorder = 3
)

# Set x-axis major tick positions to only 10 game multiples.
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
# Set y-axis major tick positions to only 0.5 xG multiples.
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

# Title and subtitle for the legend
fig_text(
    x=0.12, y=1.1,
    s = f'{team}',
    color = "black",
    weight = "bold",
    size = 10,
    #family = "DM Sans", #This is a custom font !!
    annotationbbox_kw={"xycoords": "figure fraction"}
)

fig_text(
    x=0.12, y=1.02,
    s = f"Expected goals <created> and <conceded> | 3-match rolling average\nBy Enzo Villafuerte",
    highlight_textprops = [
        {"color": line_1[0].get_color(), "weight": "bold"},
        {"color": line_2[0].get_color(), "weight": "bold"}
    ],
    color = "black",
    size = 6,
    annotationbbox_kw={"xycoords": "figure fraction"}
)


# Adjust the image programatically - Pull the logo from Figures folder, similar to post_game_viz.py
if '_' in team:
    # Code to replace the '_' for a ' '
    team = team.replace("_", " ")


logo_ax = fig.add_axes([0.75, .99, 0.13, 0.13], zorder=1)
club_icon = mpimg.imread(f'team_logos/{team}.png')
logo_ax.imshow(club_icon)
logo_ax.axis("off")

"""
UNCOMMENT THE FOLLOWING IF NEED TO PULL LOGO FROM FOTMOB

fotmob_url = "https://images.fotmob.com/image_resources/logo/teamlogo/"

logo_ax = fig.add_axes([0.75, .99, 0.13, 0.13], zorder=1)
club_icon = Image.open(urllib.request.urlopen(f"{fotmob_url}8634.png"))
logo_ax.imshow(club_icon)
logo_ax.axis("off")
"""

fig.text(0.87, 0.93, 'Inspiration: @sonofacorner', fontsize=4, color='black', ha='right', va='bottom', alpha=0.7)

####

plt.savefig(f'xg_moving_average/images/{team}_flowchart.png', bbox_inches="tight")
#plt.show()

print(df)
print('Success')

# df columns -> Index(['id', 'isResult', 'side', 'datetime', 'result', 'h_id', 'h_title',
       #'h_short_title', 'a_id', 'a_title', 'a_short_title', 'goals_h',
       #'goals_a', 'xG_h', 'xG_a', 'forecast_w', 'forecast_d', 'forecast_l'],
      #dtype='object')

# Still need: Make it into a Streamlit app

# To run:
# python xg_moving_average/xg_moving_average.py
# python xg_moving_average/xg_moving_average.py Barcelona
# python xg_moving_average/xg_moving_average.py Sevilla
# python xg_moving_average/xg_moving_average.py Arsenal
# python xg_moving_average/xg_moving_average.py Girona
# python xg_moving_average/xg_moving_average.py Real_Madrid
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import json
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
from highlight_text import fig_text
from PIL import Image

# Streamlit App
st.title("xG Rolling Average Visualization")
st.sidebar.header("Team Selection")

# Instead of definining teams manually, planning on using one the files of xG model_v21, filter based on soure (ud) and df['team'].unique()
# Define teams available on Understat
teams_list = ['Liverpool', 'Chelsea', 'Arsenal', 'Nottingham_Forest', 'Manchester_City', 'Aston_Villa', 
                'Bournemouth', 'Fulham', 'Brighton', 'Brentford', 'Newcastle_United', 'Tottenham', 
                'Manchester_United', 'West_Ham', 'Crystal_Palace', 'Everton', 'Leicester', 'Ipswich', 
                'Wolverhampton_Wanderers', 'Southampton', 'Atletico_Madrid', 'Barcelona', 'Real_Madrid', 
                'Athletic_Club', 'Mallorca', 'Villarreal', 'Real_Sociedad', 'Osasuna', 'Real_Betis', 
                'Girona', 'Sevilla', 'Celta_Vigo', 'Rayo_Vallecano', 'Las_Palmas', 'Leganes', 'Getafe', 
                'Alaves', 'Espanyol', 'Real_Valladolid', 'Valencia', 'Bayern_Munich', 'Bayer_Leverkusen', 
                'RasenBallsport_Leipzig', 'Eintracht_Frankfurt', 'Freiburg', 'VfB_Stuttgart', 'Werder_Bremen', 
                'Mainz_05', 'Borussia_Dortmund', 'Borussia_M.Gladbach', 'Wolfsburg', 'Union_Berlin', 'Augsburg', 
                'Hoffenheim', 'St._Pauli', 'FC_Heidenheim', 'Holstein_Kiel', 'Bochum', 'Paris_Saint_Germain', 'Monaco', 
                'Marseille', 'Lille', 'Lyon', 'Lens', 'Nice', 'Auxerre', 'Toulouse', 'Reims', 'Brest', 'Strasbourg', 
                'Rennes', 'Nantes', 'Saint-Etienne', 'Angers', 'Le_Havre', 'Montpellier', 'Atalanta', 'Napoli', 'Inter',
                 'Lazio', 'Fiorentina', 'Juventus', 'Bologna', 'AC_Milan', 'Udinese', 'Empoli', 'Torino', 'Genoa', 'Roma',
                  'Lecce', 'Parma_Calcio_1913', 'Verona', 'Como', 'Cagliari', 'Venezia', 'Monza']


selected_team = st.sidebar.selectbox("Select a team:", teams_list)

years_list = ['2021', '2022', '2023', '2024']

@st.cache_data
def season_scraper(url):
    """Scrapes data from the given Understat URL."""
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'lxml')
        scripts = soup.find_all('script')
        strings = scripts[1].string
        ind_start = strings.index("('") + 2
        ind_end = strings.index("')")
        json_data = strings[ind_start:ind_end]
        json_data = json_data.encode('utf8').decode('unicode_escape')
        data = json.loads(json_data)
        df = pd.json_normalize(
            data,
            sep='_',
            meta=['id', 'isResult', 'side', 'datetime', 'result'],
            record_prefix=None
        )
        return df
    except Exception as e:
        st.error(f"Failed to scrape data: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_team_data(team, years):
    """Fetch data for a team across multiple seasons."""
    all_data = []
    for year in years:
        url = f'https://understat.com/team/{team}/{year}'
        df = season_scraper(url)
        if not df.empty:
            df['year'] = year
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# Scrape and visualize data
if selected_team:
    with st.spinner(f"Fetching data for {selected_team}..."):
        df = fetch_team_data(selected_team, years_list)

    if df.empty:
        st.error(f"No data available for {selected_team}.")
    else:
        df = df[df['isResult'] == True]
        df['team_xG'] = df.apply(
            lambda row: row['xG_h'] if row['h_title'] == selected_team else row['xG_a'], axis=1
        )
        df['team_xGA'] = df.apply(
            lambda row: row['xG_a'] if row['h_title'] == selected_team else row['xG_h'], axis=1
        )

        # Rolling
        Y_for = df['team_xG'].rolling(window=1, min_periods=0).mean()
        Y_ag = df['team_xGA'].rolling(window=1, min_periods=0).mean()
        X = pd.Series(range(len(Y_for)))

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4), dpi=200, facecolor="#EFE9E6")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("grey")
        ax.grid(visible=True, lw=0.75, ls=":", color="lightgrey")

        line_1 = ax.plot(X, Y_for, color="#004D98", zorder=4, label="xG For")
        line_2 = ax.plot(X, Y_ag, color="#A50044", zorder=4, label="xG Against")

        ax.fill_between(X, Y_ag, Y_for, where=Y_for >= Y_ag, interpolate=True, alpha=0.85, color=line_1[0].get_color())
        ax.fill_between(X, Y_ag, Y_for, where=Y_ag > Y_for, interpolate=True, alpha=0.85, color=line_2[0].get_color())
        ax.set_title(f"{selected_team} xG Rolling Average (6 Matches)", fontsize=12)
        ax.legend()

        st.pyplot(fig)

        if '_' in selected_team:
            # Code to replace the '_' for a ' '
            selected_team = selected_team.replace("_", " ")

        # Add a placeholder for team logo
        st.sidebar.image(f"team_logos/{selected_team}.png", caption=selected_team, use_container_width=True)


# streamlit run xg_moving_average/streamlit/app.py
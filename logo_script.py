import pandas as pd
import numpy as np
import os
import requests 



fotmob_url = "https://images.fotmob.com/image_resources/logo/teamlogo/"

dict_teams = {'Bayern Munich':'9823', 
              'Borussia Dortmund':'9789',
              'Bayer Leverkusen':'8178',
             'VfB Stuttgart':'10269',
             'RasenBallsport Leipzig':'178475',
             'Hoffenheim': 8226,
             'Wolfsburg':8721,
             'Eintracht Frankfurt': 9810,
             'Freiburg': 8358,
             'Union Berlin': 8149,
             'Werder Bremen': 8697,
             'Augsburg': 8406,
             'FC Heidenheim': 94937,
             'Bochum': 9911,
             'Borussia M.Gladbach': 9788,
             'FC Cologne': 8722,
             'Darmstadt': 8262,
             'Mainz 05': 9905,
             'Holstein Kiel': 8150,
             'St. Pauli': 8152,
            'Manchester City': 8456,
             'Liverpool': 8650,
             'Brighton': 10204,
             'Tottenham': 8586,
             'Arsenal': 9825,
             'Aston Villa': 10252,
             'West Ham': 8654,
             'Newcastle United': 10261,
             'Manchester United': 10260,
             'Crystal Palace': 9826,
             'Fulham': 9879,
             'Nottingham Forest': 10203,
             'Brentford': 9937,
             'Chelsea': 8455,
             'Everton': 8668,
             'Wolverhampton Wanderers': 8602,
             'Bournemouth': 8678,
             'Luton': 8346,
             'Burnley': 8191,
             'Sheffield United': 8657,
             'Leicester': 8197,
             'Southampton': 8466,
             'Ipswich': 9902,
             'Girona': 7732,
             'Real Madrid': 8633,
             'Barcelona': 8634,
             'Athletic Club': 8315,
             'Atletico Madrid': 9906,
             'Real Sociedad': 8560,
             'Rayo Vallecano': 8370,
             'Valencia': 10267,
             'Cadiz': 8385,
             'Real Betis': 8603,
             'Getafe': 8305,
             'Sevilla': 8302,
             'Villarreal': 10205,
             'Osasuna': 8371,
             'Alaves': 9866,
             'Mallorca': 8661,
             'Celta Vigo': 9910,
             'Las Palmas': 8306,
             'Granada': 7878,
             'Almeria': 9865,
             'Espanyol': 8558,
             'Real Valladolid': 10281,
             'Leganes': 7854, 
              'Brest': 8521,
             'Nice': 9831,
             'Paris Saint Germain': 9847,
             'Monaco': 9829,
             'Reims': 9837,
             'Strasbourg': 9848,
             'Le Havre': 9746,
             'Marseille': 8592,
             'Rennes': 9851,
             'Nantes': 9830,
             'Lille': 8639,
             'Metz': 8550,
             'Montpellier': 10249,
             'Lorient': 8689,
             'Toulouse': 9941,
             'Lens': 8588,
             'Lyon': 9748,
             'Clermont Foot': 8311,
             'Angers': 8121,
             'Auxerre': 8583,
             'Saint-Etienne': 9853,
              'Inter': 8636,
             'AC Milan': 8564,
             'Juventus': 9885,
             'Atalanta': 8524,
             'Napoli': 9875,
             'Lecce': 9888,
             'Fiorentina': 8535,
             'Frosinone': 9891,
             'Sassuolo': 7943,
             'Torino': 9804,
             'Genoa': 10233,
             'Lazio': 8543,
             'Bologna': 9857,
             'Verona': 9876,
             'Monza': 6504,
             'Roma': 8686,
             'Salernitana': 6480,
             'Udinese': 8600,
             'Empoli': 8534,
             'Cagliari': 8529,
             'Como': 10171,
             'Parma Calcio 1913': 10167
             # Need to add new teams since this is coming from a couple of seasons ago
             }

# set up download directory
download_directory = 'team_logos/'

# create the directory if it doesn't exist
os.makedirs(download_directory, exist_ok=True)

# download images and save in directory
for team, team_id in dict_teams.items():
    url = f"{fotmob_url}{team_id}.png"
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(download_directory, f"{team}.png"), 'wb') as file:
            file.write(response.content)
            print(f"Downloaded: {team}")
    else:
            print(f"Failed to download: {team}")


# To run: python logo_script.py
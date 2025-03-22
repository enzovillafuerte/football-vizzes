team_dict = {
    
    'Universitario': 4409,
    'Junior FC': 2254,
    'Alianza Lima': 6398,
    'Sporting Cristal': 1848,
    'River Plate': 10076,
    'Independiente del Valle': 192875,
    'Barcelona SC': 6453,
    'Sao Paulo': 10277,
    'Libertad': 6620,
    'Talleres': 10101,
    'Palmeiras': 10283,
    'Bolivar': 5983,
    'Cerro Porteno': 6295
}

# Download the PNGs
import requests
import os


fotmob_url = 'https://images.fotmob.com/image_resources/logo/teamlogo/'

download_directory = 'whoscored-vizzes/teams_png'

os.makedirs(download_directory, exist_ok=True)



for team, team_id in team_dict.items():

    url = f'{fotmob_url}{team_id}.png'

    response = requests.get(url)

    if response.status_code == 200:

        with open(os.path.join(download_directory, f"{team}.png"), 'wb') as file:

            file.write(response.content)
            print(f'Downloading: {team}')

    else: 
        print(f"Unable to download {team}")

# python whoscored-vizzes/teams_png/script.py
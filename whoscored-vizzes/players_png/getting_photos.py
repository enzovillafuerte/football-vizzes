player_dict = {

    # Universitario
    'S. Britos' : 343431, 
    'Aldo Corzo': 118213,
    'William Riveros': 807783,
    'Di Benedetto': 725300,
    'Dulanto': 661777,
    # 'R. Guzmán':  ,
    'P. Reyna': 1028040,
    'C. Inga': 1415328,
    'Andy Polo': 263619,
    'Ancajima': 1432790,
    'R. Ureña': 304875,
    'Murrugarra': 921734,
    'Calcaterra': 146810,
    'J. Concha': 988208,
    'Perez Guedes': 280141,
    'J. Velez': 526949,
    'Gabriel Costa': 389888,
    'E. Flores': 315926,
    'J. Rivera': 847129,
    'D. Churín': 113608,
    'A. Valera': 1133998,
    'Carabalí': 1130861,


    # Sporting Cristal
    # https://www.clubsportingcristal.pe/index.php/plantel-profesional
    'D. Enríquez': 1432728 ,
    'R. Solís': 834699,
    'A. Duarte': 930546,
    #'C. Bautista': ,
    'F. Alcedo': 1254926,
    'G. Chávez': 840182,
    'R. Lutiger': 1049675,
    'L. Sosa': 581722,
    'J. Lora': 1188651,
    'F. Romero': 690324 ,
    'N. Pasquini': 414440,
    #'A. Pósito':  ,
    'L. Díaz': 1274397,
    #'F. Cassiano': ,
    #'F. Lora': ,
    'J. Pretell': 922898,
    'C. Gonzales': 428274,
    'Y. Yotún': 116911,
    #'Maxloren Castro': ,
    'M. Távara': 834703 ,
    #'Ian Wisdom': ,
    'Jostin Alarcón': 1254284,
    #'Y. Del Valle': ,
    #'S. Sánchez': ,
    #'A. Beltrán': ,
    'G. Cazonatti': 872394,
    'Catriel Cabellos': 1418741,
    'S. González': 892828,
    'Cauteruccio': 115871,
    'I. Avila': 116937,
    'L. Iberico': 547724,
    #'M. Sosa': ,
    'F. Pacheco': 844386 ,




    # Alianza Lima
    # https://clubalianzalima.com.pe/#/plantilla/ficha/Ricardo-Lagos

    'G. Vizcarra': 493691,
    'J. Castillo': 964199,
    'R. Garcés': 751527,
    'M. Huamán': 1248844,
    'Ricardo Lagos': 1086550,
    "Jhamir D'Arrigo": 1181405,
    # 'De La Cruz': 
    'Eryc Castillo': 458861,
    'A. Campos': 427217,
    'M. Succar': 982363,
    'Archimbaud': 547854,
    'Gaibor': 200257,
    'Trauco': 248704,
    'Lavandeira': 127268,
    #'R. Alarcón': ,
    #'J. Velásquez': ,
    'Ceppelini': 166121,
    'H. Barcos': 18694,
    #'C. Gómez': ,
    'A. Cantero': 1098460,
    #'J. Delgado': ,
    'G. Enrique': 1312034,
    #'S. Peralta': ,
    #'J. Navea': ,
    'K. Quevedo': 831684


}

# Download the PNGs
import requests
import os


fotmob_url = 'https://images.fotmob.com/image_resources/playerimages/'

download_directory = 'whoscored-vizzes/players_png'

os.makedirs(download_directory, exist_ok=True)



for player, player_id in player_dict.items():

    url = f'{fotmob_url}{player_id}.png'

    response = requests.get(url)

    if response.status_code == 200:

        with open(os.path.join(download_directory, f"{player}.png"), 'wb') as file:

            file.write(response.content)
            print(f'Downloading: {player}')

    else: 
        print(f"Unable to download {player}")


# python whoscored-vizzes/players_png/getting_photos.py
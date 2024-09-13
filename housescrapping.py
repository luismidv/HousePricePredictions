import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from selenium import webdriver
from requests_html import HTMLSession
import random


provinces = ['Alava','Albacete','Alicante','Almería','Asturias','Avila','Badajoz','Barcelona','Burgos','Cáceres',
'Cádiz','Cantabria','Castellón','Ciudad Real','Córdoba','La Coruña','Cuenca','Gerona','Granada','Guadalajara',
'Guipúzcoa','Huelva','Huesca','Islas Baleares','Jaén','León','Lérida','Lugo','Madrid','Málaga','Murcia','Navarra',
'Orense','Palencia','Las Palmas','Pontevedra','La Rioja','Salamanca','Segovia','Sevilla','Soria','Tarragona',
'Santa Cruz de Tenerife','Teruel','Toledo','Valencia','Valladolid','Vizcaya','Zamora','Zaragoza']

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15'
]

def build_link_search():
    """THIS FUNCTION ASKS FOR THE CITY AND THE ZONE YOU ARE SEARCHING FOR AND RETURNS THE LINK FOR THAT CITY/ZONE

    Returns:
        _type_: _description_
    """
    link = "https://www.fotocasa.es/es/"
    capital = input("¿What city are you searching for? ")
    zone = input("¿Are you searching for an specific zone? ")
    if zone != "":
        if capital.capitalize() in provinces:
            link = "https://www.fotocasa.es/es/comprar/viviendas/" +capital+ "-capital" + "/" + zone + "/l"
        else:
            link = "https://www.fotocasa.es/es/comprar/viviendas/" +capital+ "/" + zone + "/l"  
    else:
        link = "https://www.fotocasa.es/es/comprar/viviendas/" +capital+ "/todas-las-zonas/l"
    return link

def get_font_code_webdriver(link):
    """THIS FUNCTIONS RECEIVES THE PAGE'S LINK AND GETS IT'S FONT CODE"""
    try:
        driver = webdriver.Firefox()
        driver.get(link)
        font_code = driver.page_source
        
    except:
        print("Could not get fotocasa's font code")

    return font_code

def get_font_code(link):
    try:
        session = HTMLSession()
        response = session.get(link)
        font_code = response.html.html
        print(response)
        return font_code
    except:
        print("Error at getting html")

def get_font_code_re(link):
    new_int = random.randint(0,6)
    new_user_agent = user_agents[new_int]
    print(new_user_agent)
    new_header = requests.utils.default_headers()
    new_header.update({'User-Agent': new_user_agent})
    response = requests.get(link,headers=new_header)
    print(response.status_code)
    return response
    

    

def page_finding_fotocasa(font_code):
    fotocasa_soup = BeautifulSoup(font_code.text, 'html.parser')
    all_houses = fotocasa_soup.find_all('div', class_='re-CardPackPremium-info')
    print(len(all_houses))
    
    for house in all_houses:
        print("Entramos al all houses")
        price = house.find('span', class_='re-CardPrice')
        price = price.text.strip()
            
        rooms = house.find('span', class_= 're-CardFeaturesWithIcons-feature-icon re-CardFeaturesWithIcons-feature-icon--rooms')
        rooms = rooms.text.strip()
            
        bathrooms = house.find('span', class_= 're-CardFeaturesWithIcons-feature-icon re-CardFeaturesWithIcons-feature-icon--bathrooms')
        bathrooms = bathrooms.text.strip()
            
        metters = house.find('span', class_= 're-CardFeaturesWithIcons-feature-icon re-CardFeaturesWithIcons-feature-icon--surface')
        metters = metters.text.strip()
            
        floor = house.find('span', class_= 're-CardFeaturesWithIcons-feature-icon re-CardFeaturesWithIcons-feature-icon--floor')
        floor = floor.text.strip()
        
        new_dataframe = pd.DataFrame({'price':[price], 
            'rooms':[rooms], 
            'bathrooms':[bathrooms], 
            'metters':[metters],
            'floor': [floor]})
        concat_dataframe(new_dataframe)
            
        
def build_csv():
    data = pd.DataFrame(columns=['price', 'rooms', 'bathrooms', 'metters', 'floor'])
    return data

def concat_dataframe(newdata):
    concated_data = pd.concat([data,newdata],ignore_index=False)
    print(concated_data)
             

def call_print(lista):
    print("Imprimimos la lista")
    for l in lista:
        print(l)



link = build_link_search()
print(link)
font_code = get_font_code_re(link)
data = build_csv()
page_finding_fotocasa(font_code)




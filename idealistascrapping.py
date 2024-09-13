import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from selenium import webdriver
from bs4 import BeautifulSoup
import re


def link_build():
    """FUNCTION THAT BUILD SEARCH LINK"""
    main_link = 'https://www.idealista.com/'
    zona = input("In which city are you searching for a house: ")
    final_link = 'https://www.idealista.com/alquiler-viviendas/' + zona + "-" + zona + "/"

    driver = webdriver.Chrome()
    driver.get(final_link)
    font_code = driver.page_source

    # font_code = requests.get(final_link)
    print(font_code)
    return font_code


def get_font_code(font_code):
    """FUNCTION GETS THE FONT CODE OF THE SEARCHED PAGE"""
    house_soup = BeautifulSoup(font_code, 'html.parser')
    all_houses = house_soup.find_all('div', class_='item-info-container')

    for house in all_houses:
        houses_list = []

        price = house.find('span', class_='item-price h2-simulated')
        price = price.text.strip()
        price = replace_price(price)
        houses_list.append(price)

        all_houses_spans = house.find_all('span', class_='item-detail')
        for item in all_houses_spans:
            houses_list.append(item.text.strip())

        houses_list[1] = replace_price(str(houses_list[1]))
        houses_list[2] = replace_metters(str(houses_list[2]))
        
        build_dataframe_csv(houses_list)

def build_dataframe_csv(houses_list):
    """FUNCTION TO CONSTRUCT THE DATAFRAME FOR THE MODEL"""
    with open('data/luismi_dataframe.csv', 'a') as f:
        position = 1
        for i in range(3):
            if position < 3:
                value1 = houses_list[i] + ","
                f.write(value1)
                position += 1
            else:
                lastvalue = houses_list[i] + "\n"
                f.write(lastvalue)

def generate_house_test_dataframe():
    rooms = np.random.randint(1,7,60)
    metters = np.random.randint(75,126,60)
    dicthouses = {'Floor': rooms, 'Metters': metters}
    test_dataframe = pd.DataFrame(dicthouses)
    test_dataframe.to_csv('test_dataframe_houses.csv', index=False)
    """with open('/data/test_dataframe.csv' , 'a') as f:
        position = 1
        for i in range(len(rooms)):
            if position == 1:
                f.write(rooms[i] + ",")
                position +=1
            else:
                f.write(metters[i] + "\n")"""


def replace_price(string):
    return re.sub(r'[^0-9]', '', string)


def replace_metters(string):
    return re.sub(r'[^0-9]', '', string)


def replace_elevator(string):
    return re.sub(r'[0-9a-zA-Z]','',string)


#font_code = link_build()
#get_font_code(font_code)
generate_house_test_dataframe()
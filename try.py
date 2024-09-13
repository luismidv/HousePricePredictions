import pandas as pd
import numpy as np

df = pd.DataFrame({'year' : [1900,2000], 
                   'month': [20001,21323], 
                   'intnum':[10000,209109]})

df2 = pd.DataFrame({'year': [1990,2000],
                   'month': [2000,2000],
                   'intnum': [2001,2000]})

#new_dataframe = pd.concat([df,df2], ignore_index=False)
list = [2000,3000,4000]
#new_dataframe1 = pd.concat([new_dataframe,list], ignore_index=False)
#print(new_dataframe)
string = 'Planta 7 con ascensor'
string1 = 'Planta 7 sin ascensor'
"""
def ascensor(string):
    if 'con ascensor' in string:
        string = 'Si'
        print(string)
    else:
        string = 'No'
        print(string)

ascensor(string)
ascensor(string1)
"""
"""
rooms = np.random.randint(2,5,60)
metters = np.random.randint(75,126,60)
dict = {'rooms': rooms, 'metters': metters}
test_dataframe = pd.DataFrame(dict = {'rooms' : rooms, 'metters' : metters})
print(test_dataframe)"""

datatest = pd.read_csv('data/test_dataframe_houses.csv')
features = ['Rooms', 'Metters']
xtest = datatest[features].copy()
print(xtest)
print(datatest)
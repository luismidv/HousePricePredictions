import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree

def read_csv_info():
    """READ DATA FROM CSV AND DESCRIBE COLUMNS STATS

    Returns:
        Dataframe: Contains CSV data
    """
    data = pd.read_csv('data/Housing.csv')
    #print(data.describe())
    return data

def prepare_csv_data():
    """Prepare the csv data in order to make the predictions
    """
    data_labels = data['price']
    data_features = data[['area', 'bedrooms','bathrooms','parking']]
    algorithm = input("Do you want to use DecisionTreeRegressor(1) or RandomForest(2): ")
    print(algorithm)
    if algorithm == str(1):
        all_leaf_nodes = [5,25,50,100,250,500]
        score = {leaf_size: model_prediction(data_labels, data_features,algorithm,leaf_size,) for leaf_size in all_leaf_nodes}
        minmae = min(score, key = score.get)
        print(score)
        print(min(score.values()))
        print(min(score,key = score.get))
    else:
        mae = model_prediction(data_labels,data_features,algorithm)
        print(mae)
    

def model_prediction(data_labels,data_features, algorithm,leaf_node = []):
    
    train_X, val_X, train_y, val_y = train_test_split(data_features, data_labels, random_state=0)
    #print(train_X)
    #print(train_y)
    
    if algorithm == str(1):
        print("Regressor")
        model_regressor = DecisionTreeRegressor(max_leaf_nodes=leaf_node,random_state = 0)
        model_regressor.fit(train_X,train_y)
        val_predictions = model_regressor.predict(val_X)
        #print(val_predictions)
        mae = mean_absolute_error(val_y, val_predictions)
        return mae
    else:
        print("Forest")
        model_forest = RandomForestRegressor(random_state=1)
        model_forest.fit(train_X, train_y) 
        val_predictions = model_forest.predict(val_X)
        mae = mean_absolute_error(val_y, val_predictions)
        return mae

read_csv_info()
data = read_csv_info()
prepare_csv_data()
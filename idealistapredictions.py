import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import shap
#import eli5
#from eli5 import PermutationImportance
from sklearn import tree
import graphviz


def description():
    """GET DATA AND PRINT DESCRIPTION IF UNCOMMENTED"""
    data = pd.read_csv('data/luismi_dataframe.csv')
    print(data.describe())
    return data

def description_test():
    """GET DATA TEST AND PRINT DESCRIPTION IF UNCOMMENTED"""
    datatest = pd.read_csv('data/test_dataframe_houses.csv')
    print(datatest.describe())
    return datatest

def fill_dataframe(data):
    """FILL NULL VALUES ON DATAFRAME WITH MEANS"""
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].mean())
    
    return data

def build_data_and_predict(data):
    """BUILD DATAFRAMES WITH FEATURES AND LABELS, CREATE RANDOM FOREST MODELS AND CHECK  """
    data.columns = ['Price', 'Floor', 'Metters']
    features = data[['Floor', 'Metters']]
    labels = data['Price']
    train_X, val_X, train_y, val_y = train_test_split(features,labels, random_state=0)
    model_regressor1 = RandomForestRegressor(random_state=0, n_estimators=200,criterion="absolute_error")
    model_regressor2 = RandomForestRegressor(random_state=0, n_estimators=100,criterion="absolute_error")
    model_regressor3 = RandomForestRegressor(random_state=0, max_depth=8, n_estimators=100,min_samples_split=20)
    model_regressor4 = RandomForestRegressor(random_state=0, max_depth=9, n_estimators=100,min_samples_split=15)
    model_regressor5 = RandomForestRegressor(random_state=0, max_depth=10, n_estimators=100,min_samples_split=12)
    models = [model_regressor1,model_regressor2,model_regressor3,model_regressor4,model_regressor5]
    #model_regressor = model_regressor.fit(train_X, train_y)
    #val_predictions = model_regressor.predict(val_X)
    #permutation(model_regressor,val_X, val_y,features)
    #tree_graph = tree.export_graphviz(model_regressor, out_file = None, feature_names = features)
    #graphviz.Source(tree_graph)
    #partialdependance(model_regressor,val_X,'Metters','Floor')
    #shapsummary(model_regressor,val_X)
    """CHECK MODELS MAE UNCOMMENT IF NEEDED"""
    #measure_model_quality(models,features,labels,train_X, val_X, train_y, val_y)
    make_prediction(model_regressor5, features,labels)

def make_prediction(model_regressor5,features,labels):
    """FUNCTION TO MAKE THE PREDICTIONS WITH THE TEST DATASET"""
    print(features)
    print(labels)
    
    model_regressor5.fit(features,labels)
    preds_test = model_regressor5.predict(datatest)
    build_new_dataframe_results(preds_test)

def build_new_dataframe_results(preds_test):
    output = pd.DataFrame({'Floor': datatest.Floor,
                           'Metters':datatest.Metters,
                           'MonthlyPrice' : preds_test})
    print(output)


def measure_model_quality(models,features,labels,train_X, val_X, train_y, val_y):
    """CHECK MODEL WITH THE BEST QUALITY"""
    pos = 1
    maelist = []
    for model in models:
        
        model.fit(train_X, train_y)
        preds = model.predict(val_X)
        mae = mean_absolute_error(val_y, preds)
        print("Model " + str(pos) + " mae calculado = " + str(mae))
        pos += 1
        maelist.append(mae)
    print(min(maelist))
    
def partialdependance(model_regressor,val_X, feat_name,second_feat_name):
    """PARTIAL DEPENDENCE GRAPH FOR SPECIFIC FEATURE"""
    PartialDependenceDisplay.from_estimator(model_regressor, val_X, [feat_name])
    PartialDependenceDisplay.from_estimator(model_regressor, val_X, [second_feat_name])
    plt.show()

def shapvalues(model_regressor,val_X):
    """SHAP VALUES GRAPH FOR SPECIFIC FEATURE"""
    data_to_shap = val_X.iloc[5]
    print(data_to_shap)
    data_for_prediction_array = data_to_shap.values.reshape(1,-1)
    model_regressor.predict_proba(data_for_prediction_array)
    explainer = shap.TreeExplainer(model_regressor)
    data_to_shap = val_X.iloc[5]
    shap_values = explainer.shap_values(data_to_shap)
    print(len(shap_values))
    print(len(data_to_shap))
    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shap_values[1], data_to_shap)



def shapsummary(model_regressor,val_X):
    """SHAP SUMMARY GRAPH FOR SPECIFIC FEATURE"""
    explainer = shap.TreeExplainer(model_regressor)
    shap_values = explainer.shap_values(val_X)
    shap.summary_plot(shap_values[1], val_X)

def cross_validation_applying(data):
    """USE CROSS VALIDATION SO WE KNOW WHICH PART OF OUR DATASET IS BETTER TO USE AS TEST/TRAIN
       IT'S PRETTY USEFULL WHEN YOU DON'T HAVE A LARGE DATASET, WHAT IN GENERAL GIVES YOU A BETTER ACCURACY"""
    #data.columns = ['Price', 'Floor', 'Metters']
    data.columns = [['Price', 'Floor', 'Metters']]
    features = data[['Floor', 'Metters']]
    labels = data['Price']
    rand_forest = RandomForestRegressor(n_estimators=200, random_state=0)
    scores = -1* cross_val_score(rand_forest, features, labels.values.ravel(), cv = 5, scoring='neg_mean_absolute_error')
    print("MAE scores\n", scores)

def xgboost_applying(data):
    """APPLYING XGBOOST TO GET THE MEAN ABSOLUTE ERROR, NOTICE THAT WE ARE NOT USING N_JOBS, THAT'S BECAUSE IT'S A SHORT DATASET,
       IN CASES WHERE WE HAVE A LARGER DATASET N_JOB IS A GOOD IDEA"""
    data.columns = ['Price', 'Floors', 'Metters']
    features = data[['Floors', 'Metters']]
    labels = data['Price']
    X_train, X_valid, y_train, y_valid = train_test_split(features,labels, random_state=0)
    xgb_model = XGBRegressor(n_estimators = 50, learning_rate = 0.05, random_state = 0, early_stopping_rounds = 5,)
    xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose = False)
    predictions = xgb_model.predict(X_valid)
    mae = mean_absolute_error(predictions, y_valid)
    print(mae)

data = description()
datatest = description_test()
data = fill_dataframe(data)
print(data)
#build_data_and_predict(data)
#cross_validation_applying(data)
xgboost_applying(data)




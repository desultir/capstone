# -*- coding: utf-8 -*-
"""
Created on Sun May  6 07:56:30 2018

@author: u107939
"""
from __future__ import print_function, division
from collections import defaultdict, Counter
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from datetime import datetime as dt
from sklearn import metrics
import pandas as pd
import json
import numpy as np
import os
import random
import re
import nltk
from string import punctuation
from nltk.corpus import stopwords # Import the stop word list
from sklearn.metrics import confusion_matrix
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import gensim
from gensim import corpora
from datetime import datetime, time
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from math import sqrt
pd.options.mode.chained_assignment = None
from sklearn import linear_model
import pandas as pd
from textblob import TextBlob
from datetime import datetime, time
import numpy as np

import warnings
warnings.filterwarnings('ignore')


ROOT='C:/Users/u107939/Capstone'
os.chdir(ROOT)
ROOT=os.getcwd()
print(os.getcwd())

df27=pd.read_csv("C:/Users/u107939/Capstone/twitter_senti_features.csv"\
                 ,header=0)
df27=df27.iloc[:,1:38]


#df27.to_csv("twitter_senti_features.csv")



df5=pd.read_csv("C:/Users/u107939/Capstone/DataSet/project_data/cleaned_target.csv",header=0)
df5.shape
df5=df5.loc[df5['Year'] == 2016]
df5=df5.loc[df5['Type'] == 'Rate per 100,000 population']



for i in range (4,5): #31
    print(i)
    df6=df5.iloc[:,0:i]
    colname  = df6.columns[i-1]
    df6= df6.dropna(subset=[colname]) # delete lga with no crime data
    
    df7 = pd.merge(df27, df6, left_on='lga', right_on='LGA')

    df7.drop([ 'Year','Type','LGA',\
              ], axis=1,inplace=True)
    
    df7=df7.sample(frac=1)
    lgalist = df7['lga'].tolist()
    
    df7.drop([ 'lga'], axis=1,inplace=True)
    train_features = df7.iloc[:,0:24] #15316
    train_labels = df7[colname]
    
    
    
    '''
    # for feature selection
    loo = LeaveOneOut()
    loo.get_n_splits(train_features)
    
    
    for train_index, test_index in loo.split(train_features):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = train_features.iloc[train_index], train_features.iloc[test_index]
        y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]
        #print(X_train, X_test, y_train, y_test)
        base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
        base_model.fit(X_train, y_train)
        predictions = base_model.predict(X_test)
        print(predictions,y_test)
       
    '''
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10)
    kf.get_n_splits(train_features)
    #train_features= train_features.iloc[:,base_model.feature_importances_.argsort()[::-1][:12]]
    
    pv=[]
    av=[]
    pv1=[]
    av1=[]
    rmse_rf_cv = []
    rmse_lr_cv = []
    for train_index, test_index in kf.split(train_features):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = train_features.iloc[train_index], train_features.iloc[test_index]
        y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]
        #print(X_train, X_test, y_train, y_test)
        base_model1 = RandomForestRegressor(n_estimators = 10, random_state = 42)
        base_model1.fit(X_train, y_train)
        predictions = base_model1.predict(X_test)
      
        pv.extend(predictions.tolist())
        av.extend(y_test.tolist())
        lm = linear_model.Lasso()
        model = lm.fit(X_train, y_train)
        lm_predictions = lm.predict(X_test)
        #lm.score(X_test,y_test)
        lm.coef_
        pv1.extend(lm_predictions.tolist())
        av1.extend(y_test.tolist())
        rmse_lr_cv.append(sqrt(mean_squared_error(y_test, lm_predictions)))
        rmse_rf_cv.append(sqrt(mean_squared_error(y_test, predictions)))
    
    #print(list(zip(model.coef_, X_test.columns)))   
    
    print('for :',colname)
    print('Rsq_lm_lasso: ',metrics.r2_score(av1,pv1))
    print('Rsq_rf: ',metrics.r2_score(av,pv))
    rms1 = sqrt(mean_squared_error(av1, pv1))
    print('RMSE_lm_lasso: ',rms1)
    rms = sqrt(mean_squared_error(av, pv))
    print('RMSE_rf: ',rms)
    print('Var_RMSE_LR',np.var(rmse_lr_cv))
    print('Var_RMSE_RF',np.var(rmse_rf_cv))


    
    #VISUALISATION
    
    import matplotlib.pyplot as plt
    pos=list(enumerate(pv))
    pos1=list(enumerate(av))
    pos2=list(enumerate(lgalist))
    fig = plt.figure(figsize=(12,10))
    plt.pNlot([l[0] for l in pos],[l[1] for l in pos],label='pred', lw=2, color='r')
    plt.plot([l[0] for l in pos1],[l[1] for l in pos1],label='actual', lw=1, color='g')
   # plt.xticks(pos,pos2)
    
    plt.legend(loc='upper right')
    
    
    
  
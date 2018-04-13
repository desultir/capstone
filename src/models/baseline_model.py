# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
#nltk.download()
 
#clear variable
#%reset


#DATA
ROOT='C:/Users/u107939/Capstone'
os.chdir(ROOT)
ROOT=os.getcwd()
print(os.getcwd())

df1=pd.read_csv("C:/Users/u107939/Capstone/DataSet/project_data/2016_demographics.csv",header=0)
df1.drop(df1.columns[[0]],axis=1,inplace=True)

df1.head(1)
df1.shape
df1.iloc[:,15536]

#import lga_Area dat
df10=pd.read_csv("C:/Users/u107939/Capstone/DataSet/project_data/LGA_Area.csv",header=0)
df10=df10.iloc[:,[1,4]]

df11 = pd.merge(df1, df10, left_on='LGA_CODE_2016', right_on='Census_Code_2016')



df2=pd.read_csv("C:/Users/u107939/Capstone/DataSet/project_data/cleaned_target.csv",header=0)
df2.shape
df3=df2.iloc[:,0:4]
df3=df3.loc[df3['Year'] == 2016]
df3=df3.loc[df3['Type'] == 'Rate per 100,000 population']
df3= df3.dropna(subset=['Assault - domestic violence']) # delete lga with no crime data




df4 = pd.merge(df11, df3, left_on='clean_name', right_on='LGA')
df4.drop(['clean_name', 'Year','Type','LGA','LGA_CODE_2016',\
          'Census_Code_2016'], axis=1,inplace=True)

df4=df4.replace('..', np.nan, regex=True)

df4=df4.dropna(axis=1, how='any')
#df4.shape


#FEATURES
#select 12 features from Roman's model
df5= df4.loc[:, df4.columns.str.startswith(('M_Tot_Separated','M_Tot_Tot_G05'\
                                            ,'Percent_Unem_loyment_P'\
                                            ,'Tot_P_P','Area sqkm'\
                                            ,'Median_tot_hhd_inc_weekly'\
                                            ,'Median_mortgage_repay_monthly'\
                                            ,'Median_rent_weekly'\
                                            ,'SB_OSB_NRA_Tot_P','Tot_P_G14'\
                                            ,'Median_age_persons'\
                                            ,'P_Elsewhere_Tot','P_Tot_Tot_G09'\
                                            ,'Lang_spoken_home_Eng_only_P'\
                                            ,'Tec_Furt_Educ_inst_Tot_P','Tot_P_G15'\
                                            ,'P_LonePnt_Tot','P_Tot_Tot_G23'\
                                            ,'Assault - domestic violence'))]


df5['Num_sep_males']=df5['M_Tot_Separated_G05']/df5['M_Tot_Tot_G05']
df5['Pop_density']=df5['Tot_P_P_G01']/df5['Area sqkm']
df5['Per_no_relg']=df5['SB_OSB_NRA_Tot_P_G14']/df5['Tot_P_G14']
df5['Per_immi']=df5['P_Elsewhere_Tot_G09H']/df5['P_Tot_Tot_G09H']
df5['Per_english']=df5['Lang_spoken_home_Eng_only_P_G01']/df5['Tot_P_P_G01']
df5['Per_Voc']=df5['Tec_Furt_Educ_inst_Tot_P_G15']/df5['Tot_P_G15']
df5['Per_lone_par']=df5['P_LonePnt_Tot_G23B']/df5['P_LonePnt_Tot_G23B']

df5.drop(['M_Tot_Separated_G05', 'M_Tot_Tot_G05','Tot_P_P_G01',\
          'Area sqkm','SB_OSB_NRA_Tot_P_G14','Tot_P_G14',\
          'P_Elsewhere_Tot_G09H','P_Tot_Tot_G09H',\
          'Lang_spoken_home_Eng_only_P_G01','Tot_P_P_G01',\
          'Tec_Furt_Educ_inst_Tot_P_G15','Tot_P_G15',\
          'P_LonePnt_Tot_G23B','P_LonePnt_Tot_G23B',\
          'P_Tot_Tot_G23B'], axis=1,inplace=True)

df5=df5[['Percent_Unem_loyment_P_G40', 'Median_age_persons_G02',\
       'Median_mortgage_repay_monthly_G02', 'Median_rent_weekly_G02',\
       'Median_tot_hhd_inc_weekly_G02','Per_lone_par', \
       'Num_sep_males', 'Pop_density', 'Per_no_relg', 'Per_immi',\
       'Per_english', 'Per_Voc','Assault - domestic violence']]

# MODELS
train_features = df5.iloc[:,0:12] #15316
train_labels = df4['Assault - domestic violence']




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
   


from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
kf.get_n_splits(train_features)
train_features= train_features.iloc[:,base_model.feature_importances_.argsort()[::-1][:12]]
er=[]
ma=[]
pv=[]
av=[]
for train_index, test_index in kf.split(train_features):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = train_features.iloc[train_index], train_features.iloc[test_index]
    y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]
    #print(X_train, X_test, y_train, y_test)
    base_model1 = RandomForestRegressor(n_estimators = 10, random_state = 42)
    base_model1.fit(X_train, y_train)
    predictions = base_model1.predict(X_test)
    print(predictions,y_test)
    errors = abs(predictions - y_test)
    rmse=abs(predictions - y_test)
    er.extend(errors.tolist())
    mape=100 * (errors / y_test)
    ma.extend(mape.tolist())
    pv.extend(predictions.tolist())
    av.extend(y_test.tolist())
   

rms = sqrt(mean_squared_error(av, pv))
print('RMSE: ',rms)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')



#VISUALISATION

import matplotlib.pyplot as plt
pos=list(enumerate(pv))
pos1=list(enumerate(av))
fig = plt.figure(figsize=(12,10))
plt.plot([l[0] for l in pos],[l[1] for l in pos],label='pred', lw=2, color='r')
plt.plot([l[0] for l in pos1],[l[1] for l in pos1],label='actual', lw=1, color='g')
plt.legend(loc='upper right')



#do code to support model
#"data" is the X dataframe and model is the SKlearn object

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(train_features.columns, base_model.feature_importances_):
    feats[feature] = importance #add the name/value pair 

df9 = pd.DataFrame([feats], columns=feats.keys())
df9.to_csv("out.csv")

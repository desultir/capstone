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
from sklearn import linear_model
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

#df1.head(1)
#df1.shape
#df1.iloc[:,15318]

#import lga_Area dat
#df10=pd.read_csv("C:/Users/u107939/Capstone/DataSet/project_data/LGA_Area.csv",header=0)
#df10=df10.iloc[:,[1,4]]

#df11 = pd.merge(df1, df10, left_on='LGA_CODE_2016', right_on='Census_Code_2016')



df2=pd.read_csv("C:/Users/u107939/Capstone/DataSet/project_data/cleaned_target.csv",header=0)
df2.shape
df2=df2.loc[df2['Year'] == 2016]
df2=df2.loc[df2['Type'] == 'Rate per 100,000 population']


'**************twitter data processing************************************'
df21=pd.read_csv("C:/Users/u107939/Capstone/DataSet/project_data/tweets_w_lga.csv"\
                 ,header=0)

#type(df1)
#n_rows=df1.shape[0]
#list(df1)

#df1['lang'].value_counts().to_csv("out.csv")
df21=df21.loc[df21['lang'] == "en"]

df21['mydates']=pd.to_datetime(df21['created_at'], format='%Y-%m-%d %H:%M:%S')
df21['day_of_week']=df21['mydates'].dt.weekday_name

#df21.head(5)
#df21['mydates'].dt.time

df21['time_of_day'] = np.where((df21['mydates'].dt.time >= time(4,00)) &\
  (df21['mydates'].dt.time < time(10,00)), 'Morning', \
 np.where((df21['mydates'].dt.time >= time(10,00)) &\
  (df21['mydates'].dt.time < time(16,00)), 'Afternoon', \
 np.where((df21['mydates'].dt.time >= time(16,00)) &\
  (df21['mydates'].dt.time < time(22,00)), 'Evening', \
 np.where((df21['mydates'].dt.time >= time(22,00)) |\
  (df21['mydates'].dt.time < time(4,00)), 'Night', \
  '') )))


df22=pd.crosstab(df21.lga, df21.time_of_day)
df22.reset_index(inplace=True)





for i in range (4,7): #31
    print(i)
    df3=df2.iloc[:,0:i]
    colname  = df3.columns[i-1]
    df3= df3.dropna(subset=[colname]) # delete lga with no crime data
    
    
    
    
    df4 = pd.merge(df1, df3, left_on='clean_name', right_on='LGA')
    df4= pd.merge(df4, df22, left_on='LGA', right_on='lga',how='left')
    df4.drop(['clean_name', 'Year','Type','LGA',\
              'lga_x','lga_y'], axis=1,inplace=True)
    df4.update(df4[['Afternoon','Evening','Morning','Night']].fillna(0))
    df4=df4.replace('..', np.nan, regex=True)
    
    df4=df4.dropna(axis=1, how='any')
    #df4.shape
    
    
    #FEATURES
    #select 12 features from Roman's model
    df5= df4.loc[:, df4.columns.str.startswith(('M_Tot_Separated','M_Tot_Tot_G05'\
                                                ,'Percent_Unem_loyment_P'\
                                                ,'Tot_P_P','area'\
                                                ,'Median_tot_hhd_inc_weekly'\
                                                ,'Median_mortgage_repay_monthly'\
                                                ,'Median_rent_weekly'\
                                                ,'SB_OSB_NRA_Tot_P','Tot_P_G14'\
                                                ,'Median_age_persons'\
                                                ,'P_Elsewhere_Tot','P_Tot_Tot_G09'\
                                                ,'Lang_spoken_home_Eng_only_P'\
                                                ,'Tec_Furt_Educ_inst_Tot_P','Tot_P_G15'\
                                                ,'P_LonePnt_Tot','P_Tot_Tot_G23'\
                                                ,'Afternoon','Evening'\
                                                ,'Morning','Night'\
                                                ,colname))]
    
    
    df5['Num_sep_males']=df5['M_Tot_Separated_G05']/df5['M_Tot_Tot_G05']
    df5['Pop_density']=df5['Tot_P_P_G01']/df5['area']
    df5['Per_no_relg']=df5['SB_OSB_NRA_Tot_P_G14']/df5['Tot_P_G14']
    df5['Per_immi']=df5['P_Elsewhere_Tot_G09H']/df5['P_Tot_Tot_G09H']
    df5['Per_english']=df5['Lang_spoken_home_Eng_only_P_G01']/df5['Tot_P_P_G01']
    df5['Per_Voc']=df5['Tec_Furt_Educ_inst_Tot_P_G15']/df5['Tot_P_G15']
    df5['Per_lone_par']=df5['P_LonePnt_Tot_G23B']/df5['P_LonePnt_Tot_G23B']
    
    df5.drop(['M_Tot_Separated_G05', 'M_Tot_Tot_G05','Tot_P_P_G01',\
              'area','SB_OSB_NRA_Tot_P_G14','Tot_P_G14',\
              'P_Elsewhere_Tot_G09H','P_Tot_Tot_G09H',\
              'Lang_spoken_home_Eng_only_P_G01','Tot_P_P_G01',\
              'Tec_Furt_Educ_inst_Tot_P_G15','Tot_P_G15',\
              'P_LonePnt_Tot_G23B','P_LonePnt_Tot_G23B',\
              'P_Tot_Tot_G23B'], axis=1,inplace=True)
    

    
    df5=df5[['Percent_Unem_loyment_P_G40', 'Median_age_persons_G02',\
           'Median_mortgage_repay_monthly_G02', 'Median_rent_weekly_G02',\
           'Median_tot_hhd_inc_weekly_G02','Per_lone_par', \
           'Num_sep_males', 'Pop_density', 'Per_no_relg', 'Per_immi',\
           'Per_english', 'Per_Voc','Afternoon','Evening',\
           'Morning','Night',colname]]
    

    
    
    # MODELS
    train_features = df5.iloc[:,0:12] #15316
    train_labels = df5[colname]
    
    
    
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

    
'''
    
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
    for feature, importance in zip(train_features.columns, base_model1.feature_importances_):
        feats[feature] = importance #add the name/value pair 
    
    df9 = pd.DataFrame([feats], columns=feats.keys())
    df9.to_csv("out.csv")
'''

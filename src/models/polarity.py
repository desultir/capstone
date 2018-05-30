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


ROOT='C:/Users/u107939/Capstone'
os.chdir(ROOT)
ROOT=os.getcwd()
print(os.getcwd())

df2=pd.read_csv("C:/Users/u107939/Capstone/DataSet/project_data/loctrisma2016.csv"\
                 ,header=0)

df2=df2.dropna(subset=['lga'])

df21=df2#.head(10000)

df3=pd.read_csv("C:/Users/u107939/Capstone/DataSet/project_data/derived_features_2016.csv"\
                 ,header=0)

df31=df3.iloc[:,0:2]

df4=pd.read_csv("C:/Users/u107939/Capstone/DataSet/project_data/trisma2016_w_emotion_file.csv"\
                 ,header=0)

df41 = df4.loc[:,['lga','Emotion']]

df42=pd.crosstab(df41.lga, df41.Emotion,margins=False)
df42.reset_index(inplace=True)




#df21=df21.loc[df21['lang'] == "en"]
#df21=df21.dropna(subset=['lat'])



for i, row in df21.iterrows():
    #print (i,df21.loc[i,'text'])
    df21.loc[i,'Polr'] = TextBlob(df21.loc[i,'text']).polarity
    df21.loc[i,'Subj'] = TextBlob(df21.loc[i,'text']).subjectivity
    if i%10000==0:
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print(i)
    

#df21.to_csv("out.csv")
df21['mydates']=pd.to_datetime(df21['created_at'], format='%Y-%m-%d %H:%M:%S')

df21['time_of_day'] = np.where((df21['mydates'].dt.time >= time(6,00)) &\
  (df21['mydates'].dt.time < time(12,00)), 'Morning', \
 np.where((df21['mydates'].dt.time >= time(12,00)) &\
  (df21['mydates'].dt.time < time(18,00)), 'Afternoon', \
 np.where((df21['mydates'].dt.time >= time(18,00)) &\
  (df21['mydates'].dt.time <= time(23,59)), 'Evening', \
 np.where((df21['mydates'].dt.time >= time(0,00)) |\
  (df21['mydates'].dt.time < time(6,00)), 'Night', \
  '') )))

df21['Polr_abs'] = df21['Polr'].abs()
df21['Polr_neg'] = np.where(df21['Polr']>0, 0, df21['Polr'])

df22=pd.crosstab(df21.lga, df21.time_of_day,margins=False)
df22.reset_index(inplace=True)

df23=df21.groupby(['lga', 'time_of_day']).Polr_neg.sum().unstack(fill_value=0)
df23.reset_index(inplace=True)
df23.columns=['lga','Polr_Neg_A','Polr_Neg_E','Polr_Neg_M','Polr_Neg_N']



df24=df21.groupby(['lga']).agg({'Polr':'sum','Subj':'sum','Polr_abs':'sum'\
            ,'Polr_neg':'sum'})
df24.reset_index(inplace=True)

df25= pd.merge(df22, df23, left_on='lga', right_on='lga')
df25= pd.merge(df25, df24, left_on='lga', right_on='lga')

df26= pd.merge(df25, df31, left_on='lga', right_on='lga',how='left')

df26['Afternoon_pcapita']=df26['Afternoon']/df26['population']
df26['Evening_pcapita']=df26['Evening']/df26['population']
df26['Morning_pcapita']=df26['Morning']/df26['population']
df26['Night_pcapita']=df26['Night']/df26['population']
df26['Polr_Neg_A_pcapita']=df26['Polr_Neg_A']/df26['population']
df26['Polr_Neg_E_pcapita']=df26['Polr_Neg_E']/df26['population']
df26['Polr_Neg_M_pcapita']=df26['Polr_Neg_M']/df26['population']
df26['Polr_Neg_N_pcapita']=df26['Polr_Neg_N']/df26['population']
df26['Polr_pcapita']=df26['Polr']/df26['population']
df26['Subj_pcapita']=df26['Subj']/df26['population']
df26['Polr_abs_pcapita']=df26['Polr_abs']/df26['population']
df26['Polr_neg_pcapita']=df26['Polr_neg']/df26['population']

df42= pd.merge(df42, df31, left_on='lga', right_on='lga',how='left')

df42['Anger_pcapita'] = df42['Anger']/df42['population']
df42['Disgust_pcapita'] = df42['Disgust']/df42['population']
df42['Fear_pcapita'] = df42['Fear']/df42['population']
df42['Joy_pcapita'] = df42['Joy']/df42['population']
df42['Sadness_pcapita'] = df42['Sadness']/df42['population']
df42['Surprise_pcapita'] = df42['Surprise']/df42['population']

df27= pd.merge(df26, df42, left_on='lga', right_on='lga',how='inner')
df27.drop(['population_x','population_y'], axis=1,inplace=True)

df27.to_csv("twitter_senti_features.csv")


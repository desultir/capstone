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

arrBad = [
'2g1c',
#'2 girls 1 cup',
'acrotomophilia',
'anal',
'anilingus',
'anus',
'arsehole',
'ass',
'asshole',
'assmunch',
'auto erotic',
'autoerotic',
'babeland',
#'baby batter',
#'ball gag',
#'ball gravy',
#'ball kicking',
#'ball licking',
#'ball sack',
#'ball sucking',
'bangbros',
'bareback',
#'barely legal',
'barenaked',
'bastardo',
'bastinado',
'bbw',
'bdsm',
#'beaver cleaver',
#'beaver lips',
'bestiality',
#'bi curious',
#'big black',
#'big breasts',
#'big knockers',
#'big tits',
'bimbos',
'birdlock',
'bitch',
#'black cock',
#'blonde action',
#'blonde on blonde action',
#'blow j',
#'blow your l',
#'blue waffle',
'blumpkin',
'bollocks',
'bondage',
'boner',
'boob',
'boobs',
#'booty call',
#'brown showers',
#'brunette action',
'bukkake',
'bulldyke',
#'bullet vibe',
#'bung hole',
'bunghole',
'busty',
'butt',
'buttcheeks',
'butthole',
#'camel toe',
'camgirl',
'camslut',
'camwhore',
#'carpet muncher',
'carpetmuncher',
#'chocolate rosebuds',
'circlejerk',
#'cleveland steamer',
'clit',
'clitoris',
#'clover clamps',
'clusterfuck',
'cock',
'cocks',
'coprolagnia',
'coprophilia',
'cornhole',
'cum',
'cumming',
'cunnilingus',
'cunt',
'darkie',
#'date rape',
'daterape',
#'deep throat',
'deepthroat',
'dick',
'dildo',
#'dirty pillows',
#'dirty sanchez',
#'dog style',
#'doggie style',
'doggiestyle',
#'doggy style',
'doggystyle',
'dolcett',
'domination',
'dominatrix',
'dommes',
#'donkey punch',
#'double dong',
#'double penetration',
#'dp action',
#'eat my ass',
'ecchi',
'ejaculation',
'erotic',
'erotism',
'escort',
#'ethical slut',
'eunuch',
'faggot',
'fecal',
'felch',
'fellatio',
'feltch',
#'female squirting',
'femdom',
'figging',
'fingering',
'fisting',
#'foot fetish',
'footjob',
'frotting',
'fuck',
'fucking',
#'fuck buttons',
#'fudge packer',
'fudgepacker',
'futanari',
'g-spot',
#'gang bang',
#'gay sex',
'genitals',
#'giant cock',
#'girl on',
#'girl on top',
#'girls gone wild',
'goatcx',
'goatse',
'gokkun',
#'golden shower',
#'goo girl',
'goodpoop',
'goregasm',
'grope',
#'group sex',
'guro',
#'hand job',
'handjob',
#'hard core',
'hardcore',
'hentai',
'homoerotic',
'honkey',
'hooker',
#'hot chick',
#'how to kill',
#'how to murder',
#'huge fat',
'humping',
'incest',
'intercourse',
#'jack off',
#'jail bait',
'jailbait',
#'jerk off',
'jigaboo',
'jiggaboo',
'jiggerboo',
'jizz',
'juggs',
'kike',
'kinbaku',
'kinkster',
'kinky',
'knobbing',
#'leather restraint',
#'leather straight jacket',
#'lemon party',
'lolita',
'lovemaking',
#'make me come',
#'male squirting',
'masturbate',
#'menage a trois',
'milf',
#'missionary position',
'motherfucker',
#'mound of venus',
#'mr hands',
#'muff diver',
'muffdiving',
'nambla',
'nawashi',
'negro',
'neonazi',
#'nig nog',
'nigga',
'nigger',
'nimphomania',
'nipple',
'nipples',
#'nsfw images',
'nude',
'nudity',
'nympho',
'nymphomania',
'octopussy',
'omorashi',
#'one cup two girls',
#'one guy one jar',
'orgasm',
'orgy',
'paedophile',
'panties',
'panty',
'pedobear',
'pedophile',
'pegging',
'penis',
#'phone sex',
#'piece of shit',
#'piss pig',
'pissing',
'pisspig',
'playboy',
#'pleasure chest',
#'pole smoker',
'ponyplay',
'poof',
#'poop chute',
'poopchute',
'porn',
'porno',
'pornography',
#'prince albert piercing',
'pthc',
'pubes',
'pussy',
'queaf',
'raghead',
#'raging boner',
'rape',
'raping',
'rapist',
'rectum',
#'reverse cowgirl',
'rimjob',
'rimming',
#'rosy palm',
#'rosy palm and her 5 sisters',
#'rusty trombone',
's&m',
'sadism',
'scat',
'schlong',
'scissoring',
'semen',
'sex',
'sexo',
'sexy',
#'shaved beaver',
#'shaved pussy',
'shemale',
'shibari',
'shit',
'shota',
'shrimping',
'slanteye',
'slut',
'smut',
'snatch',
'snowballing',
'sodomize',
'sodomy',
'spic',
'spooge',
#'spread legs',
#'strap on',
'strapon',
'strappado',
#'strip club',
#'style doggy',
'suck',
'sucks',
#'suicide girls',
#'sultry women',
'swastika',
'swinger',
#'tainted love',
#'taste my',
#'tea bagging',
'threesome',
'throating',
#'tied up',
#'tight white',
'tit',
'tits',
'titties',
'titty',
#'tongue in a',
'topless',
'tosser',
'towelhead',
'tranny',
'tribadism',
#'tub girl',
'tubgirl',
'tushy',
'twat',
'twink',
'twinkie',
#'two girls one cup',
'undressing',
'upskirt',
#'urethra play',
'urophilia',
'vagina',
#'venus mound',
'vibrator',
#'violet blue',
#'violet wand',
'vorarephilia',
'voyeur',
'vulva',
'wank',
#'wet dream',
'wetback',
#'white power',
#'women rapping',
#'wrapping men',
#'wrinkled starfish',
#'xx',
'xxx',
'yaoi',
#'yellow showers',
'yiffy',
'zoophilia']

def profanityFilter(text):

    brokenStr1 = text.split()
    #badWordMask = '!@#$%!@#$%^~!@%^~@#$%!@#$%^~!'
    i = 0
    for word in brokenStr1:
        if word in arrBad:
            i=i+1
            #print (text , 'count' , i)
            #text = text.replace(word,badWordMask[:len(word)])
            #print new
    return i

#print (profanityFilter("this thing sucks sucks sucks fucking stuff"))



ROOT='C:/Users/u107939/Capstone'
os.chdir(ROOT)
ROOT=os.getcwd()
print(os.getcwd())

df2=pd.read_csv("C:/Users/u107939/Capstone/DataSet/project_data/loctrisma2016.csv"\
                 ,header=0)

df2=df2.dropna(subset=['lga'])


print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
df21=df2 #.head(20000)


for i, row in df21.iterrows():
    #print (i,df21.loc[i,'text'])
    df21.loc[i,'Predictions'] = profanityFilter(df21.loc[i,'text'])
     
    if i%10000==0:
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print(i)

df21.to_csv("out.csv")

import pandas as pd

from emotion_predictor import EmotionPredictor

# Pandas presentation options
pd.options.display.max_colwidth = 150   # show whole tweet's content
pd.options.display.width = 200          # don't break columns
# pd.options.display.max_columns = 7      # maximal number of columns


# Predictor for Ekman's emotions in multiclass setting.
model = EmotionPredictor(classification='ekman', setting='mc', use_unison_model=True)
'''
tweets = [
    "Watching the sopranos again from start to finish!",
    "Finding out i have to go to the  dentist tomorrow",
    "Sun in my eyes but I don't mind, what a beautiful day we've had in New York today!",
    "Feels like someone's stabbed me in my hope",
    "Do people have no Respect for themselves or you know others peoples homes",
    "I want to go outside and chalk but I have no chalk",
    "I hate coming to the doctors when I feel as if I might have a big problem",
    "My mom wasn't mad",
    "You don't indicate once I'm already in the road THEN rev and honk at me you stupid bitch #learnhowtodrive #bitch",
    "Come home from work and this is on my doorstep. I guess he has a secret admirer",
    "The 'egyption hot models' facebook page is pathetic... simply photos of obese horny women.",
    "I HATE PAPERS AH #AH #HATE",
]

predictions = model.predict_classes(tweets)
print(predictions, '\n')

#probabilities = model.predict_probabilities(tweets)
#print(probabilities, '\n')

embeddings = model.embedd(tweets)
print(embeddings, '\n')
'''

df21=pd.read_csv("C:/Users/u107939/Capstone/DataSet/project_data/tweets_w_lga.csv"\
                 ,header=0)

#type(df1)
#n_rows=df1.shape[0]
#list(df1)

#df1['lang'].value_counts().to_csv("out.csv")
df21=df21.loc[df21['lang'] == "en"]
df21=df21.dropna(subset=['lat'])

tweets= df21.text.tolist()
tweets=tweets[:50]
predictions = model.predict_classes(tweets)
print(predictions, '\n')
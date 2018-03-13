# -*- coding: utf-8 -*-
import os, re
import click
import logging
import pandas
from string import punctuation
from nltk.corpus import stopwords # Import the stop word list
from dotenv import find_dotenv, load_dotenv
import gensim
from gensim import corpora

@click.command()
@click.argument('filename', type=click.Path(), default="poster_docs.csv")
@click.argument('input_filepath', type=click.Path(exists=True), default="./data/interim")
@click.argument('output_filepath', type=click.Path(), default="./data/processed")
def run_lda(filename, input_filepath, output_filepath):

    df2= pandas.read_csv(os.path.join(input_filepath, filename))
    df_text=df2.loc[:,'text']
    n_row=df_text.shape[0]
    for i in range(n_row):
        df_text[i]=clean_data(df_text[i])

    df_list = [df_l.split() for df_l in df_text.values]

    # Creating the term dictionary of our courpus, where every unique term is assigned an index. 
    dictionary = corpora.Dictionary(df_list)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in df_list]

    Lda = gensim.models.ldamodel.LdaModel

    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=50)

    print(ldamodel.print_topics(num_topics=10, num_words=10))

    #TODO- Actually write out tweetdocs/topic affiliation

def clean_data(data):
    #This function inputs a text desc and returns various types of feature vectors
    data=re.sub(r"http\S+", "", data)
    data=data.strip(punctuation)
    #remove stop words and numbers
    data = re.sub("[^a-zA-Z]", " ", data) #letters only
    stops = set(stopwords.words("english"))
    data  = [word for word in data.split() if word.lower() not in stops]
    data = ' '.join(data)
    data=data.lower()

    return data




if __name__ == '__main__':
    run_lda()

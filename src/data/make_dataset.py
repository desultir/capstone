# -*- coding: utf-8 -*-
import os
import click
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
import logging
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath='../data/raw', output_filepath='../data/processed'):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df1=pd.read_csv(os.path.join(input_filepath, "tweet.csv"),header=0)

    #df1=df1.loc[df1['lang'] == "en"]

    df = df1[['from_user_name','text']]

    df2=df.groupby('from_user_name', as_index=False).agg({'text':lambda x:' '.join(x)})

    df2.to_csv(os.path.join(output_filepath, "poster_docs.csv"))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

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
import time
import random
import re
import nltk
import logging
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.option('--by-location/--no-location', default=False)
@click.argument('input_filepath', type=click.Path(exists=True), default='./data/raw')
@click.argument('output_filepath', type=click.Path(), default='./data/processed')
def main(input_filepath='../data/raw', output_filepath='../data/processed', by_location=False):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
        check
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


    if not by_location:
        df1=pd.read_csv(os.path.join(input_filepath, "tweet.csv"),header=0)
        df = df1[['from_user_name','text']]
        df2=df.groupby('from_user_name', as_index=False).agg({'text':lambda x:' '.join(x)})

        df2.to_csv(os.path.join(output_filepath, "poster_docs.csv"))
    else:
        #geolocate
        df1=pd.read_csv(os.path.join(input_filepath, "geocoded-tweet.csv"),header=0)

        df2=df.groupby('lga', as_index=False).agg({'text':lambda x:' '.join(x)})

        df2.to_csv(os.path.join(output_filepath, "location_docs.csv"))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

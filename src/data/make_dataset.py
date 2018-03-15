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
@click.option('--geocode/--no-geocode', default=False)
@click.argument('input_filepath', type=click.Path(exists=True), default='./data/raw')
@click.argument('output_filepath', type=click.Path(), default='./data/processed')
def main(input_filepath='../data/raw', output_filepath='../data/processed', by_location=False, geocode=False):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
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
        from geolocation.main import GoogleMaps
        if geocode:
            df1=pd.read_csv(os.path.join(input_filepath, "tweet.csv"),header=0)
            gm = GoogleMaps(api_key='#FILLMEOUT')
        
            df = df1[['location', 'lat','lng','text']]
            i = 0
            def geocode(df):
                if i < 2400 and pd.notnull(df['lat']) and pd.notnull(df['lng']):
                    i += 1
                    location = gm.search(lat=df['lat'], lng=df['lng'])).first()
                    for area in location.administrative_area:
                        if area.area_type =='administrative_area_level_2':
                            df['lga'] = area.name

                    df['postal_code'] = location.postal_code
                    df['city'] = location.city
                    if i % 50 == 0:
                        #ratelimit
                        print(".")
                        time.sleep(1)
            df['location'] = df.apply(geocode, axis=1)
        else:
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

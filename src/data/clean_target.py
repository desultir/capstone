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
@click.argument('input_filepath', type=click.Path(exists=True), default='./data/raw')
@click.argument('output_filepath', type=click.Path(), default='./data/processed')
def main(input_filepath='../data/raw', output_filepath='../data/processed'):
    """ Runs data processing scripts to turn raw target data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final target set from raw data')
    with open(os.path.join(input_filepath,"LgaRankings_27_Offences.xlsx"), "rb") as f:
        df_dict = pd.read_excel(f, sheet_name=None, encoding='latin-1', skiprows=0, header=[4,5], na_values=['nc'], skip_footer=7)

    cleaned_list = []

    for key in df_dict:
        unstacked = pd.DataFrame(df_dict[key].unstack())

        unstacked.index = unstacked.index.rename(["Year","Type", "LGA"])
        unstacked.columns = [key]
        cleaned_list.append(unstacked)
        
    final_df = cleaned_list[0]
    for i in range(1,len(cleaned_list)):
        final_df = pd.merge(final_df, cleaned_list[i], how='inner', left_index=True, right_index=True)

    final_df.to_csv(os.path.join(output_filepath, "cleaned_target.csv"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

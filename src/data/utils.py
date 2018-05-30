###
# Util functions for use across modules
#
import pandas as pd
import numpy as np
import sys, os

def load_target(base = "../../data/clean", filename="filtered_nonlog_target.csv"):
    nswdf_target = pd.read_csv(os.path.join(base, filename), index_col=0)
    return nswdf_target

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

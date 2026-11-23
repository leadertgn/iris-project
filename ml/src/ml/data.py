import pandas as pd

from . import config


def load_data() : 
    df = pd.read_csv(config.DATA_PATH)
    X = df.drop(columns=config.TARGET)
    y = df[config.TARGET]
    return X,y

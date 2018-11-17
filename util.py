import numpy as np 
import pandas as pd

DATA_PATH = "./movie_data.csv"

def load_data():
	df = pd.read_csv(DATA_PATH, sep=',', header=None)
	print(df)
	return df.values
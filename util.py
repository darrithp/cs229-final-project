import numpy as np 

DATA_PATH = "./movie_data.csv"

def load_data():
	data_arr = np.genfromtxt(DATA_PATH, delimiter=',')
	return data_arr
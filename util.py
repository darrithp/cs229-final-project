import numpy as np 
import pandas as pd

DATA_PATH = "./movie_data.csv"

def load_data():
	df = (pd.read_csv(DATA_PATH, encoding='latin1', sep=',', header=None)).values
	# 0='imdbId', 1='Imdb Link', 2='Title', 3='IMDB Score', 4='Genre', 5='Poster'
	imgs = {img_id: img_link for img_id, img_link in df[1:, [0, 5]] if type(img_link) is np.str}
	genres = {img_id: genre_list.split('|') for img_id, genre_list in df[1:, [0, 4]] if type(genre_list) is np.str and img_id in imgs}
	return imgs, genres

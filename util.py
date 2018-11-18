import numpy as np 
import pandas as pd
import urllib

DATA_PATH = "./movie_data.csv"
IMG_PATH = "./imgs/"
ID_PATH = "./updated_ids.txt"

# Returns dictionaries of image urls and genres indexed by imdb IDs. The
# output dictionaries only include values which are non-null. 
def load_data():
	df = (pd.read_csv(DATA_PATH, encoding='latin1', sep=',', header=None)).values
	# 0='imdbId', 1='Imdb Link', 2='Title', 3='IMDB Score', 4='Genre', 5='Poster'
	imgs = {img_id: img_link for img_id, img_link in df[1:, [0, 5]] if type(img_link) is np.str}
	genres = {img_id: genre_list.split('|') for img_id, genre_list in df[1:, [0, 4]] if type(genre_list) is np.str and img_id in imgs}
	return imgs, genres

# Downloads the images for the imdb IDs with non-null urls
def load_imgs(imgs):
	updated_ids = list(imgs.keys())
	ntotal = len(updated_ids)
	print("Downloading poster images...")
	ndownloaded = 0
	nprocessed = 0
	for img_id in imgs:
		try:
			urllib.request.urlretrieve(imgs[img_id], IMG_PATH + img_id + ".png")
			ndownloaded += 1
		except:
			updated_ids.remove(img_id)
		nprocessed += 1
		if nprocessed % 1000 == 0:
			print("Processed " + str(nprocessed) + " images...")
	print(str(ndownloaded) + " of " + str(ntotal) + " images successfully downloaded!")
	with open(ID_PATH, 'w') as f:
		for item in updated_ids:
			f.write("%s\n" % item)

# THIS IS THE FUNCTION you use to get the correctly indexed genres. 
# The return value has all the clean imdb IDs. 
def get_genres(genres):
	updated_ids = open(ID_PATH, 'r').read().split('\n')[:-1]
	return {img_id: genres[img_id] for img_id in genres if img_id in updated_ids}
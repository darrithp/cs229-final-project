import numpy as np
import pandas as pd
import urllib
import ast
import os
import pickle
import copy

DATA_PATH = "./data/movie_data.csv"
IMG_PATH = "./data/imgs/"
LABELS_PATH = "./data/genre_labels.txt"


def unpickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def repickle(obj, out_path):
    with open(out_path, 'wb') as f:
        pickle.dump(obj, f, protocol=2)

def create_dir(dir_path):
    try:
        os.stat(dir_path)
    except:
        os.mkdir(dir_path)
# Returns dictionaries of image urls and genres indexed by imdb IDs. The
# output dictionaries only include values which are non-null strings. 
def load_data():
        df = (pd.read_csv(DATA_PATH, encoding='latin1', sep=',', header=None)).values
        # 0='imdbId', 1='Imdb Link', 2='Title', 3='IMDB Score', 4='Genre', 5='Poster'
        pre_imgs = {img_id: img_link for img_id, img_link in df[1:, [0, 5]] if type(img_link) is np.str}
        genres = {img_id: genre_list.split('|') for img_id, genre_list in df[1:, [0, 4]] if type(genre_list) is np.str and img_id in pre_imgs}
        imgs = {img_id: pre_imgs[img_id] for img_id in genres}
        return imgs, genres

def generate_pkl_files():
        df = (pd.read_csv(DATA_PATH, encoding='latin1', sep=',', header=None)).values
        # 0='imdbId', 1='Imdb Link', 2='Title', 3='IMDB Score', 4='Genre', 5='Poster'
        pre_imgs = {img_id: img_link for img_id, img_link in df[1:, [0, 5]] if type(img_link) is np.str}
        genres = {img_id: genre_list.split('|') for img_id, genre_list in df[1:, [0, 4]] if type(genre_list) is np.str and img_id in pre_imgs}
        class_list = []
        for key in genres:
                class_list = class_list + genres[key]
        class_list_set = set(class_list)
        class_list = list(class_list_set)

        repickle(class_list, os.path.join("data", "class_indeces.pkl"))
        repickle(genres, os.path.join("data", "genre_multi_labels.pkl"))


        multi_genres = copy.deepcopy(genres)

        for key in genres:
                genres[key] = class_list.index(genres[key][0])
                new_list = []
                for genre in multi_genres[key]:
                    new_list.append(class_list.index(genre))
                multi_genres[key] = new_list

        repickle(genres, os.path.join("data", "movies_multi_index.pkl"))
        repickle(genres, os.path.join("data", "movies_single_index.pkl"))

# Downloads the images for the imdb IDs with valid non-null urls.
# Returns an updated version of the genres dictionary which contains
# entries for only those movies whose posters were downloaded successfully.
# (Note that the imgs dict is no longer needed since we have the 
# images themselves after calling this function)
def load_imgs(imgs, genres):
        ntotal = len(imgs)
        print("Downloading poster images...")
        ndownloaded = 0
        nprocessed = 0
        for img_id in imgs:
                try:
                        urllib.request.urlretrieve(imgs[img_id], IMG_PATH + img_id + ".png")
                        ndownloaded += 1
                except:
                        del genres[img_id]
                nprocessed += 1
                if nprocessed % 10 == 0:
                        print("Processed " + str(nprocessed) + " images...")
        print(str(ndownloaded) + " of " + str(ntotal) + " images successfully downloaded!")
        # write genre labels to txt file for easy access later
        with open(LABELS_PATH, 'w') as f:
                f.write(str(genres))
        return genres

# Reads the genres dict in from LABELS_PATH
def get_genres():
        return ast.literal_eval(open(LABELS_PATH, 'r').read())

def main():
        (imgs, genres) = load_data()
        magic_genres = load_imgs(imgs, genres)

if __name__ == '__main__':
        main()

import util as ut
import os
import numpy as np
import subprocess
import sys
import random

SPLIT = [0.6, 0.2, 0.2]

# Squeezes image to [0, 1], and also makes sure it's in [C, H, W] format
def preprocess_img(img):
    return np.array(np.transpose(img, (2, 0, 1)), dtype=np.float32) / 255.0

def resize_crop_img(img_filename):
    img = Image.open(img_filename).convert('RGB')
    return img.resize(128, 256)


def generate_set(ids, movie_dict, data_path, dataset_type):
	dataset = {}
	for movie_id in ids:
		sample = {}
		sample['class'] = movie_dict[movie_id]
		img_name = movie_id + ".png"
    	image_path = os.path.join("imgs", img_name)
		img = resize_crop_img(image_path) 
		sample['preprocess_img'] = preprocess_img(img)
		dataset[movie_id] = sample

	filename = dataset_type + 'data.pkl'
	ut.repickle({'data': dataset}, os.path.join(data_path, filename))

def gen_dataset(N, data_path):
    data_path = os.path.abspath(data_path)
    util.create_dir(data_path)

    movie_dict = ut.unpickle(os.path.join("data","movies_single_index"))

    full_ids = list(movie_dict.keys())
    movie_ids = random.sample(full_ids, N)
    ids_l = len(movie_ids)
    train_ids = movie_ids[:np.floor(ids_l*SPLIT[0])]
    val_ids = movie_ids[np.floor(ids_l*SPLIT[0]):np.floor(ids_l*(SPLIT[0]+SPLIT[1]))]
    test_ids = movie_ids[np.floor(ids_l*(SPLIT[0]+SPLIT[1]):]

    generate_set(train_ids, movie_dict, data_path, "train")
    generate_set(train_ids, movie_dict, data_path, "val")
    generate_set(train_ids, movie_dict, data_path, "test")


def main():
    if len(sys.argv) < 3:
        print('Usage: python3 gen_dataset.py [N] [output_path]')
        exit()

    gen_dataset(int(sys.argv[1]), sys.argv[2])

if __name__ == '__main__':
    main()
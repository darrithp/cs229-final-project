import util as ut
import os
import numpy as np
import subprocess
import sys
import random
from PIL import Image

SPLIT = [0.8, 0.1, 0.1]

# Squeezes image to [0, 1], and also makes sure it's in [C, H, W] format
def preprocess_img(img):
    return np.array(np.transpose(img, (2, 0, 1)), dtype=np.float32) / 255.0

def resize_crop_img(img_filename):
    img = Image.open(img_filename).convert('RGB')
    return img.resize((128, 256))


def generate_set(ids, movie_dict, data_path, dataset_type, isMC):
    dataset = {}
    testlist = []
    counter = 0
    for movie_id in ids:
        counter += 1
        if counter % 50  == 0:
            print("Finished adding %d samples." % (counter))
        sample = {}
        one_hot = [0] * 28
        if isMC:
            one_hot[movie_dict[movie_id]] = 1
        else:
            genre_list = movie_dict[movie_id]
            for genre_index in genre_list:
                one_hot[genre_index] = 1
                
        sample['class'] = one_hot
        img_name = movie_id + ".png"
        image_path = os.path.join("data/imgs", img_name)
        try:
            img = resize_crop_img(image_path)
        except:
            continue
        sample['preprocess_img'] = preprocess_img(img)
        testlist.append(sample['class'])
        dataset[movie_id] = sample

    filename = dataset_type + 'data.pkl'
    ut.repickle({'data': dataset}, os.path.join(data_path, filename))

def gen_dataset(N, data_path, isMC = True):
    data_path = os.path.abspath(data_path)
    ut.create_dir(data_path)

    if (isMC):
        movie_dict = ut.unpickle(os.path.join("data","movies_single_index.pkl"))
    else:
        movie_dict = ut.unpickle(os.path.join("data","movies_multi_index.pkl")) #Need to make this new index thing

    full_ids = list(movie_dict.keys())
    movie_ids = random.sample(full_ids, N)

    ids_l = len(movie_ids)
    train_ids = movie_ids[:int(ids_l*SPLIT[0])]
    val_ids = movie_ids[int(ids_l*SPLIT[0]):int(ids_l*(SPLIT[0]+SPLIT[1]))]
    test_ids = movie_ids[int(ids_l*(SPLIT[0]+SPLIT[1])):-1]

    print("Generating Training Set...")
    generate_set(train_ids, movie_dict, data_path, "train", isMC)
    print("Generating Validation Set...")
    generate_set(val_ids, movie_dict, data_path, "val", isMC)
    print("Generating Test Set...")
    generate_set(test_ids, movie_dict, data_path, "test", isMC)


def main():
    if len(sys.argv) < 3:
        print('Usage: python3 gen_dataset.py [MC|ML] [N]')
        exit()

    isMultiClassifier = True
    if sys.argv[1] == "ML":
        isMultiClassifier = False
    num_samples = int(sys.argv[2])
    data_path = os.path.join("data", "temp", (sys.argv[1] + str(num_samples) + "dataset"))
    gen_dataset(num_samples, data_path, isMultiClassifier)

if __name__ == '__main__':
    main()

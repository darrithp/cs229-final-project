
import torch.utils.data as data
import util
import numpy as np

class MovieDataset(data.Dataset):

    def __init__(self, data_path):
        dataset = util.unpickle(data_path)
        self.data = dataset['data']
        
    def __getitem__(self, index):
        item = self.data[index]
        img = item['preprocessed_img']
        label = item['class']
        return img, label
    
    def __len__(self):
        return len(self.data.keys())


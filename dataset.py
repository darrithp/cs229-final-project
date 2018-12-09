
import torch.utils.data as data
import util
import numpy as np

class MovieDataset(data.Dataset):

    def __init__(self, data_path):
        dataset = util.unpickle(data_path)
        self.data = dataset['data']
        self.ids = list(self.data.keys())
        
    def __getitem__(self, index):
        sample = self.data[self.ids[index]]
        img = sample['preprocess_img']
        label = sample['class']
        return img, label
    
    def __len__(self):
        return len(self.data.keys())


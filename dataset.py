from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
import os
from config import Config
config = Config()
data_config = json.load(open(config.DATA_CONFIG))

'''
For dataset visit kaggle for https://www.kaggle.com/datasets/dysonsphere90/animegan or use kaggle api as described in the notebooks
'''

class PhotoToAnimeDataset(Dataset):
    def __init__(self,creator:str = None, train:bool = True,transform:bool = True)->None:
        if train:
            if creator not in {'Hayao','Paprika','Shinkai','SummerWar'}:
                raise ValueError("creator must be one of 'Hayao','Paprika','Shinkai','SummerWar'")
            self.photo_list = data_config['train']['train_photo']
            self.anime_list = data_config['train'][creator]

        else:
            self.photo_list = data_config['val']['photo']
            self.anime_list = data_config['val']['anime']
        self.transform = transform
    def __len__(self)->int:
        return max(len(self.photo_list),len(self.anime_list))
    def __getitem__(self,idx)->tuple:
        photo_img = Image.open(self.photo_list[idx%len(self.photo_list)]).convert('RGB')
        anime_img = Image.open(self.anime_list[idx%len(self.anime_list)]).convert('RGB')
        if self.transform:
            photo_img = config.preprocess(photo_img)
            anime_img = config.preprocess(anime_img)
        return photo_img,anime_img
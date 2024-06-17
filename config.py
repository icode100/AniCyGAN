from torchvision import transforms
import os
from typing import Callable
import torch
if not os.path.isdir("AnimeGAN/checkpoints"):
    os.mkdir("AnimeGAN/checkpoints")
if not os.path.isdir("AnimeGAN/gen_photo"):
    os.mkdir("AnimeGAN/gen_photo")
if not os.path.isdir("AnimeGAN/gen_anime"):
    os.mkdir("AnimeGAN/gen_anime")

class Config:
    LEARNING_RATE:float = 0.0002
    BETA_1:float = 0.5
    BETA_2:float = 0.999
    LAMBDA_CYCLE:int = 10
    LAMBDA_IDENTITY:int = 5
    NUM_EPOCHS:int = 20
    DATA_CONFIG:str = "AnimeGAN/data_config.json"
    BATCH_SIZE:int = 1
    NUM_WORKERS:int = 4
    SAVE_MODEL:bool = True
    LOAD_MODEL:bool = False
    CHECKPOINT_GEN_ANIMATION:Callable[..., str] = lambda self,creator:f'AnimeGAN/checkpoints/{creator}/gen_animation.pth'
    CHECKPOINT_DIS_ANIMATION:Callable[..., str] = lambda self,creator:f'AnimeGAN/checkpoints/{creator}/dis_animation.pth'
    CHECKPOINT_GEN_PHOTO:Callable[..., str] = lambda self,creator:f'AnimeGAN/checkpoints/{creator}/gen_photo.pth'
    CHECKPOINT_DIS_PHOTO:Callable[..., str] = lambda self,creator:f'AnimeGAN/checkpoints/{creator}/dis_photo.pth'
    DEVICE:str = "cuda" if torch.cuda.is_available() else "cpu"
    ANIME_SAVED_IMAGES:str = "AnimeGAN/gen_anime"
    PHOTO_SAVED_IMAGES:str = "AnimeGAN/gen_photo"
    preprocess:transforms.transforms.Compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
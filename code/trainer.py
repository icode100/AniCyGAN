import torch
from Generator import Generator
from Discriminator import Discriminator
from dataset import PhotoToAnimeDataset
from utils import Utils
from config import Config
utils = Utils()
config = Config()
from tqdm import tqdm
import torch.nn as nn
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os

'''
For training results check the notebooks and the loss curves
'''

class Trainer:
    def train_epoch(
            self,config = Config(),
            disc_anime = None,
            gen_anime = None,
            disc_photo = None,
            gen_photo = None,
            trainloader = None,
            valloader = None,
            opt_disc = None,
            opt_gen = None,
            l1_loss = None,
            bce_loss = None,
            d_scaler = torch.cuda.amp.GradScaler(),
            g_scaler = torch.cuda.amp.GradScaler(),
        ):
        discriminator_loss_epoch = 0
        generator_loss_epoch = 0

        loader = tqdm(trainloader,colour='blue')
        for idx,(photo_img,anime_img) in enumerate(loader):
            photo_img = photo_img.to(config.DEVICE)
            anime_img = anime_img.to(config.DEVICE)

            #train discriminator
            with torch.cuda.amp.autocast():
                # loss for generating anime from photo
                fake_anime = gen_anime(photo_img)
                disc_fake_anime,disc_real_anime = disc_anime(fake_anime.detach()),disc_anime(anime_img)
                disc_anime_loss_fake = bce_loss(disc_fake_anime,torch.zeros_like(disc_fake_anime))
                disc_anime_loss_real = bce_loss(disc_real_anime,torch.ones_like(disc_real_anime))
                disc_anime_loss = disc_anime_loss_fake + disc_anime_loss_real

                # loss for generating photo from anime
                fake_photo = gen_photo(anime_img)
                disc_fake_photo, disc_real_photo = disc_photo(fake_photo.detach()),disc_photo(photo_img)
                disc_photo_loss_fake = bce_loss(disc_fake_photo,torch.zeros_like(disc_fake_photo))
                disc_photo_loss_real = bce_loss(disc_real_photo,torch.ones_like(disc_real_photo))
                disc_photo_loss = disc_photo_loss_fake + disc_photo_loss_real
                disc_loss = (disc_anime_loss + disc_photo_loss)/2
                discriminator_loss_epoch += disc_loss.item()
            opt_disc.zero_grad()
            d_scaler.scale(disc_loss).backward(retain_graph = True)
            d_scaler.step(opt_disc)
            d_scaler.update()

            # train generator
            with torch.cuda.amp.autocast():
                # generator loss for minimization of log(1-D(G(x)))
                disc_gen_fake_anime,disc_gen_fake_photo = disc_anime(fake_anime),disc_anime(fake_photo)
                gen_loss_anime = bce_loss(disc_gen_fake_anime,torch.ones_like(disc_gen_fake_anime))
                gen_loss_photo = bce_loss(disc_gen_fake_photo,torch.ones_like(disc_gen_fake_photo))

                # cycle loss
                cycle_anime_loss = l1_loss(anime_img,gen_anime(fake_photo))
                cycle_photo_loss = l1_loss(photo_img,gen_photo(fake_anime))

                # identity loss
                identity_anime_loss = l1_loss(anime_img,gen_anime(anime_img))
                identity_photo_loss = l1_loss(photo_img,gen_photo(photo_img))

                gen_loss = gen_loss_anime + gen_loss_photo + (cycle_anime_loss + cycle_photo_loss)*config.LAMBDA_CYCLE + (identity_anime_loss + identity_photo_loss)*config.LAMBDA_IDENTITY
                generator_loss_epoch += gen_loss.item()
            opt_gen.zero_grad()
            g_scaler.scale(gen_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()
            if not idx%200:
                save_image(fake_anime*0.5+0.5,f"{config.ANIME_SAVED_IMAGES}/{idx}.png")
                save_image(fake_photo*0.5+0.5,f"{config.PHOTO_SAVED_IMAGES}/{idx}.png")
            loader.set_postfix(
                disc_loss = f"{disc_loss.item():.4f}",
                gen_loss = f"{gen_loss.item():.4f}"
            )
        return discriminator_loss_epoch/len(trainloader),generator_loss_epoch/len(trainloader)


    def train(self,creator:str=None):
        disc_anime,disc_photo = Discriminator(in_channels=3).to(config.DEVICE),Discriminator(in_channels=3).to(config.DEVICE)
        gen_anime,gen_photo = Generator(in_channels=3).to(config.DEVICE),Generator(in_channels=3).to(config.DEVICE)
        opt_disc = torch.optim.Adam(
            list(disc_anime.parameters()) + list(disc_photo.parameters()),
            lr = config.LEARNING_RATE,
            betas = (config.BETA_1,config.BETA_2)
        )
        opt_gen = torch.optim.Adam(
            list(gen_anime.parameters()) + list(gen_photo.parameters()),
            lr = config.LEARNING_RATE,
            betas = (config.BETA_1,config.BETA_2)
        )
        l1_loss = nn.L1Loss()
        bce_loss = nn.BCEWithLogitsLoss()
        if config.LOAD_MODEL:
            utils.load(
                disc_anime,
                opt_disc,
                config.CHECKPOINT_DIS_ANIMATION(creator)
            )
            utils.load(
                disc_photo,
                opt_disc,
                config.CHECKPOINT_DIS_PHOTO(creator)
            )
            utils.load(
                gen_anime,
                opt_gen,
                config.CHECKPOINT_GEN_ANIMATION(creator)
            )
            utils.load(
                gen_photo,
                opt_gen,
                config.CHECKPOINT_GEN_PHOTO(creator)
            )
        try:
            trainset = PhotoToAnimeDataset(creator = creator,transform=True)
            trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size = config.BATCH_SIZE,
                shuffle = True,
                num_workers = config.NUM_WORKERS
            )
        except ValueError as e:
            print(e)
            return
        valset = PhotoToAnimeDataset(train=False,transform=True)
        valloader = torch.utils.data.DataLoader(
            valset,
            batch_size = config.BATCH_SIZE,
            shuffle = False,
            num_workers = config.NUM_WORKERS
        )
        generator_loss = list()
        discriminator_loss = list()
        print('__Training__')
        for epoch in range(config.NUM_EPOCHS):
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
            gen_loss,disc_loss = self.train_epoch(
                config = config,
                disc_anime = disc_anime,
                gen_anime = gen_anime,
                disc_photo = disc_photo,
                gen_photo = gen_photo,
                trainloader = trainloader,
                valloader = valloader,
                opt_disc = opt_disc,
                opt_gen = opt_gen,
                l1_loss = l1_loss,
                bce_loss = bce_loss
            )
            generator_loss.append(gen_loss)
            discriminator_loss.append(disc_loss)
            if config.SAVE_MODEL:
                if not os.path.isdir(f'AnimeGAN/checkpoints/{creator}'):
                    os.mkdir(f'AnimeGAN/checkpoints/{creator}')
                utils.save(
                    disc_anime,
                    opt_disc,
                    config.CHECKPOINT_DIS_ANIMATION(creator)
                )
                utils.save(
                    disc_photo,
                    opt_disc,
                    config.CHECKPOINT_DIS_PHOTO(creator)
                )
                utils.save(
                    gen_anime,
                    opt_gen,
                    config.CHECKPOINT_GEN_ANIMATION(creator)
                )
                utils.save(
                    gen_photo,
                    opt_gen,
                    config.CHECKPOINT_GEN_PHOTO(creator)
                )
        print("__Training Complete__")
        # plotting
        print("__plotting loss curves__")
        plt.figure(figsize=(30,30))
        plt.plot(generator_loss,color="red")
        plt.plot(discriminator_loss,color='blue')
        plt.legend(['gen_loss','disc_loss'])
        plt.title('LOSS vs EPOCH',fontdict={'fontsize':10})
        plt.xlabel('EPOCH')
        plt.ylabel('LOSS')
        plt.xticks(range(0, config.NUM_EPOCHS+1 , 1),fontsize=10)
        plt.yticks(fontsize=10)
        plt.show()


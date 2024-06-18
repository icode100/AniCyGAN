# Anime Style Transfer with Cycle GAN

This project uses a Cycle GAN model with identity loss to translate natural images into anime-style images. The model can generate images in the styles of two renowned anime creators, Hayao Miyazaki and Makoto Shinkai.

<table align="left">
  <td>
    <a href="https://www.kaggle.com/code/mightywarrior001/anicygan/" target="_parent"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Colab"/></a>
  </td>
</table>

<br>

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Sample Generations](#sample)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Loss Curves](#charts)


## Introduction

The goal of this project is to create a model that can translate natural images into anime-style images. The model is trained to generate images in the unique styles of Hayao Miyazaki and Makoto Shinkai, two iconic figures in the anime industry. 

Cycle GANs are used for this task, which consist of two sets of generators and discriminators that learn to translate images from one domain to another and back, ensuring the generated images remain realistic and faithful to the target style.

## Features

- **Style Transfer:** Convert natural images into anime-style images.
- **Dual Styles:** Generate images in the styles of both Hayao Miyazaki and Makoto Shinkai.
- **Cycle Consistency:** Ensure the translated images can be mapped back to the original images, maintaining content consistency.
- **Identity Loss:** Preserve the identity of the images to enhance style transfer quality.

## Sample

![phto sample](__results__\combined.png)
<i>The above results are generated from sample phtographs taken by [Mahathi Bodela](https://github.com/mahathibodela) the first row shows the real images taken from camera the second row shows the generated anime images in Hayao Miyazaki's style and the 3rd row shows the images taken in Makoto Shinkai's style<i>
#### PC: [Mahathi Bodela](https://github.com/mahathibodela)


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/icode100/anicygan.git
   cd anime-style-transfer
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Pretrained Model

Download the pretrained model from this link for [Hayao Miyazaki's](https://github.com/icode100/AniCyGAN/blob/main/AnimeGAN/checkpoints/Hayao/gen_animation.pth) style and this for [Makoto Shinkai's](https://github.com/icode100/AniCyGAN/blob/main/AnimeGAN/checkpoints/Shinkai/gen_animation.pth) style.

### Generate Anime Style Images

1. Visit my [Application](https://anicygan.streamlit.app/).
2. On the Home page upload the image
3. In the side bar choose the style you would like to use while generating 
4. On the preferred page click on generate button 

## Dataset

The dataset is taken from kaggle named [AnimeGAN](https://www.kaggle.com/datasets/dysonsphere90/animegan) *courtesy [dyanospehere90](https://www.kaggle.com/dysonsphere90/)*. Though we have trained our generators on only Miyazaki's and Shinkai's style, the dataset consist of wide variety of styles and works.

## Training

To look for the training references look up for the `code` directory and the jupyter notebooks in `training_notebooks` directory. 


## Charts

![](__results__\shinkai_loss_curve.png)
*Loss curve for Makoto Shinkai's style GAN the <span style='color:red'>red</span> line depicts generator loss and <span style='color:blue'>blue</span> line depicts discriminator loss*

![](__results__\miyazaki_loss_curve.png)
*Loss curve for Hayao Miyazaki's style GAN the <span style='color:red'>red</span> line depicts generator loss and <span style='color:blue'>blue</span> line depicts discriminator loss*
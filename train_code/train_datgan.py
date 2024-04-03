# -*- coding: utf-8 -*-

import  sys
import os
import torch

# import important files
root_path = os.path.abspath('.')
sys.path.append(root_path)
from opt import opt
from architecture.dat import DAT    
from architecture.discriminator import UNetDiscriminatorSN, MultiScaleDiscriminator
from train_code.train_master import train_master



class train_datgan(train_master):
    def __init__(self, options, args) -> None:
        super().__init__(options, args, "datgan", True)     # Pass a model name unique code


    def loss_init(self):

        # prepare pixel loss (Generator)
        self.pixel_loss_load()

        # prepare perceptual loss
        self.GAN_loss_load()


    def call_model(self):

        # Generator: DAT light
        if opt['model_size'] == "light":
            # DAT model 
            self.generator = DAT(
                upscale=4,
                in_chans=3,
                img_size=64,
                img_range=1.,
                depth=[18],
                embed_dim=60,
                num_heads=[6],
                expansion_factor=2,
                resi_connection='3conv',
                split_size=[8,32],
                upsampler='pixelshuffledirect',
            ).cuda()

        elif opt['model_size'] == "small":
            # DAT small model 11.21M param
            self.generator = DAT(
                upscale=4,
                in_chans=3,
                img_size=64,
                img_range=1.,
                depth=[6,6,6,6,6,6],
                embed_dim=180,
                num_heads=[6,6,6,6,6,6],
                expansion_factor=2,
                resi_connection='1conv',
                split_size=[8,16],
                upsampler='pixelshuffledirect',
            ).cuda()
            
        else:
            raise NotImplementedError("We don't support such model size in DAT model")

        # Discriminator
        if opt['discriminator_type'] == "PatchDiscriminator":
            self.discriminator = MultiScaleDiscriminator(3).cuda()
        elif opt['discriminator_type'] == "UNetDiscriminator":
            self.discriminator = UNetDiscriminatorSN(3).cuda()
        
        self.generator.train(); self.discriminator.train()


    def run(self):
        self.master_run()
                        

    def calculate_loss(self, gen_hr, imgs_hr):

        ######################  We have 3 losses on Generator  ######################
        # Generator Pixel loss (l1 loss):  generated vs. GT
        l_g_pix = self.cri_pix(gen_hr, imgs_hr)
        self.generator_loss += l_g_pix
        self.weight_store["pixel_loss"] = l_g_pix


        # Generator perceptual loss:        generated vs. perceptual
        l_g_percep_danbooru = self.cri_danbooru_perceptual(gen_hr, imgs_hr)
        l_g_percep_vgg = self.cri_vgg_perceptual(gen_hr, imgs_hr)
        l_g_percep = l_g_percep_danbooru + l_g_percep_vgg 
        self.generator_loss += l_g_percep
        self.weight_store["perceptual_loss"] = l_g_percep


        # Generator GAN loss               label correction
        fake_g_preds = self.discriminator(gen_hr)
        l_g_gan = self.cri_gan(fake_g_preds, True, is_disc=False) # loss_weight (self.gan_loss_weight) is included
        self.generator_loss += l_g_gan
        self.weight_store["gan_loss"] = l_g_gan # Already with gan_loss_weight (0.2/1)


    def tensorboard_report(self, iteration):
        self.writer.add_scalar('Loss/train-Generator_Loss-Iteration', self.generator_loss, iteration)
        self.writer.add_scalar('Loss/train-Pixel_Loss-Iteration', self.weight_store["pixel_loss"], iteration)
        self.writer.add_scalar('Loss/train-Perceptual_Loss-Iteration', self.weight_store["perceptual_loss"], iteration)
        self.writer.add_scalar('Loss/train-Discriminator_Loss-Iteration', self.weight_store["gan_loss"], iteration)


# -*- coding: utf-8 -*-

import  sys
import os
import torch

# import important files
root_path = os.path.abspath('.')
sys.path.append(root_path)
from architecture.rrdb import RRDBNet
from architecture.discriminator import UNetDiscriminatorSN
from train_code.train_master import train_master



class train_esrgan(train_master):
    def __init__(self, options, args) -> None:
        super().__init__(options, args, "esrgan", True)     # Pass a model name unique code


    def loss_init(self):

        # prepare pixel loss (Generator)
        self.pixel_loss_load()

        # prepare perceptual loss
        self.GAN_loss_load()


    def call_model(self):
        # Generator
        self.generator = RRDBNet(3, 3, scale=self.options['scale'], num_block=self.options['ESR_blocks_num']).cuda()
        # self.generator = torch.compile(self.generator).cuda()
        self.discriminator = UNetDiscriminatorSN(3).cuda()
        # self.discriminator = torch.compile(self.discriminator).cuda()
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


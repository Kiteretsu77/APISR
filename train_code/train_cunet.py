# -*- coding: utf-8 -*-
import sys
import os
import torch


# Import important files
root_path = os.path.abspath('.')
sys.path.append(root_path)
from architecture.cunet import UNet_Full       # This place need to adjust for different models
from train_code.train_master import train_master



# Mixed precision training
scaler = torch.cuda.amp.GradScaler()


class train_cunet(train_master):
    def __init__(self, options, args) -> None:
        super().__init__(options, args, "cunet")    # Pass a model name unique code


    def loss_init(self):
        # Prepare pixel loss
        self.pixel_loss_load()
        

    def call_model(self):
        # Generator Prepare (Don't formet torch.compile if needed)
        self.generator = UNet_Full().cuda()     # Cunet only support 2x SR
        # self.generator = torch.compile(self.generator).cuda()
        self.generator.train()

    
    def run(self):
        self.master_run()
                        

    
    def calculate_loss(self, gen_hr, imgs_hr):

        # Generator pixel loss (l1 loss):  generated vs. GT
        l_g_pix = self.cri_pix(gen_hr, imgs_hr, self.batch_idx)
        self.weight_store["pixel_loss"] = l_g_pix
        self.generator_loss += l_g_pix


    def tensorboard_report(self, iteration):
        # self.writer.add_scalar('Loss/train-Generator_Loss-Iteration', self.generator_loss, iteration)
        self.writer.add_scalar('Loss/train-Pixel_Loss-Iteration', self.weight_store["pixel_loss"], iteration)

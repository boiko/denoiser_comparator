from __future__ import division
from __future__ import print_function
import os, time, scipy.io, shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import re
import cv2

from skimage import img_as_float32, img_as_ubyte

from .utils import *
from .model import CBDNet as CBDNetModel
from .. import Denoiser


class CBDNet(Denoiser):
    """
    Toward Convolutional blind denoising of real photographs
    Guo, Shi and Yan, Zifei and Zhang, Kai and Zuo

    Based on the code from here:
    https://github.com/IDKiro/CBDNet-pytorch

    paper: https://arxiv.org/pdf/1807.04686v2.pdf
    """

    name = "cbdnet"
    description = "Convolutional Blind Denoising of Real Photographs"

    current_dir = os.path.dirname(__file__)

    def __init__(self, weights="all", use_gpu=False):
        self._weights = weights
        self._use_gpu = use_gpu
        self._model = CBDNetModel()

        if self._use_gpu:
            print('Using GPU!')
            self._model.cuda()
        else:
            print('Using CPU!')

        self._load_checkpoint()

    def _load_checkpoint(self):
        # load pre-trained data
        checkpoint = os.path.join(self.current_dir, "checkpoints", self._weights, 'checkpoint.pth.tar')
        print(checkpoint)
        if os.path.exists(checkpoint):
            # load existing model
            self._model_info = torch.load(checkpoint)
            print('==> loading existing model: {}'.format(checkpoint))

            self._model.load_state_dict(self._model_info['state_dict'])
        else:
            raise (ValueError('Error: No trained model detected!'))

    def denoise(self, image):
        self._model.eval()
        with torch.no_grad():
            noisy_img = self.swap_bgr_rgb(img_as_float32(image))

            temp_noisy_img = noisy_img
            temp_noisy_img_chw = hwc_to_chw(temp_noisy_img)

            input_var = torch.from_numpy(temp_noisy_img_chw.copy()).type(torch.FloatTensor).unsqueeze(0)
            if self._use_gpu:
                input_var = input_var.cuda()
            _, output = self._model(input_var)

            output_np = output.squeeze().cpu().detach().numpy()
            output_np = chw_to_hwc(np.clip(output_np, 0, 1))


            return img_as_ubyte(self.swap_bgr_rgb(output_np))
        



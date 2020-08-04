from collections import OrderedDict
from google_drive_downloader import GoogleDriveDownloader as gdd
from skimage import img_as_float32, img_as_ubyte
import numpy as np
import os
import torch
from .. import Denoiser
from .networks.denoising_rgb import DenoiseNet

class CycleISP(Denoiser):
    """
    CycleISP: Real Image Restoration via Improved Data Synthesis
    Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao

    Based on the code from here:
    https://github.com/swz30/CycleISP.git

    Paper: https://arxiv.org/abs/2003.07761
    """

    name = "cycleisp"
    description = "Real Image Restoration via Improved Data Synthesis"

    def __init__(self, weights="dnd", use_gpu=False):
        """
        Initialize a new instance of this class using the given weights
        :param weights: The pre-trained weights to be used. One of "sidd" or "dnd"
        """

        self._weights = weights
        self._gpu_available = use_gpu and torch.cuda.is_available()
        self._device = torch.device("cuda" if self._gpu_available else "cpu")
        if not self._gpu_available:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # load the network
        self._model = DenoiseNet()
        # and the weights
        self._load_checkpoint()
        # move data to cpu/gpu
        self._model.to(self._device)
        # enable parallelism
        self._model = torch.nn.DataParallel(self._model)
        # last but not least, set the model in evaluation mode
        self._model.eval()

    def _download_weights(self, target_file):
        """
        Download the pre-trained weights from the specified location
        """
        file_ids = {
            "sidd": "1sraG9JKmp0ieLjntRL7Jj2FXBrPr-YVp",
            "dnd": "1740sYH7bG-c-jL5wc3e1_NOpxwGTXS9c",
        }
        gdd.download_file_from_google_drive(file_id = file_ids[self._weights],
                                            dest_path=target_file, unzip=True)

    def _load_checkpoint(self):
        filename = os.path.join(os.path.dirname(__file__), "{}_rgb.pth".format(self._weights))
        if not os.path.exists(filename):
            self._download_weights(filename)

        # in case we don't have a GPU to use, map the device to CPU
        if self._gpu_available:
            checkpoint = torch.load(filename)
        else:
            checkpoint = torch.load(filename, map_location=torch.device('cpu'))

        try:
            self._model.load_state_dict(checkpoint["state_dict"])
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self._model.load_state_dict(new_state_dict)

    def denoise(self, image):
        with torch.no_grad():
            img = np.array([img_as_float32(self.swap_bgr_rgb(image), True)])
            img = torch.from_numpy(img).to(self._device)
            img = img.permute(0, 3,1,2)
            rgb_restored = self._model(img)
            rgb_restored = torch.clamp(rgb_restored,0,1)
            output = rgb_restored.permute(0, 2, 3, 1)[0].to(self._device).detach().numpy()
            return img_as_ubyte(self.swap_bgr_rgb(output), True)

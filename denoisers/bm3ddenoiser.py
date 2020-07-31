from . import Denoiser
from skimage import img_as_float, img_as_ubyte
import bm3d
import numpy as np

class BM3DDenoiser(Denoiser):
    """
    BM3D is an algorithm for attenuation of additive spatially correlated
    stationary (aka colored) Gaussian noise

    Based on the following paper:
    Y. Mäkinen, L. Azzari, A. Foi, 2019.
    Exact Transform-Domain Noise Variance for Collaborative Filtering of Stationary Correlated Noise.
    In IEEE International Conference on Image Processing (ICIP), pp. 185-189
    """

    name = "bm3d"
    description = "Exact Transform-Domain Noise Variance for Collaborative Filtering of Stationary Correlated Noise"

    def denoise(self, image):
        # BM3d works on float RGB, so swap the input BGR into RGB, convert to float and then back
        img = self.swap_bgr_rgb(image)
        result = self.swap_bgr_rgb(bm3d.bm3d(img, sigma_psd=25.4,
                                             stage_arg=bm3d.BM3DStages.ALL_STAGES))
        return result.astype("uint8")
from . import Denoiser
from scipy.stats import uniform
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

    # best according to param_search.py
    #sigma_psd = 91.02

    sigma_psd = 30.0

    param_grid = {
        "sigma_psd": uniform(loc=0., scale=255.)
    }

    def denoise(self, image):
        # BM3d works on  RGB, so swap the input BGR into RGB and then back
        img = self.swap_bgr_rgb(image)
        result = self.swap_bgr_rgb(bm3d.bm3d_rgb(img, sigma_psd=self.sigma_psd))
        return np.clip(result, 0, 255).astype("uint8")

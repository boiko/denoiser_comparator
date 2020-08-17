from . import Denoiser
from abc import abstractmethod
from scipy.stats import uniform
from skimage import restoration
from skimage import img_as_float, img_as_ubyte
import numpy as np
import pywt

class SKImageDenoiser(Denoiser):
    """  In order to run SKImage denoisers we need to tweak the image
        format. This class wraps the @ref denoise() call to do the image conversion
        accordingly """

    @abstractmethod
    def _denoise(self, image):
        pass

    def denoise(self, image):
        # openCV images are BGR and skimage uses RGB, so invert the last and convert to float
        img = img_as_float(self.swap_bgr_rgb(image), force_copy=True)

        # and invert back to BGR for comparing
        result = self.swap_bgr_rgb(self._denoise(img))

        # not sure why but on some images there are values outside the range -1 and 1
        if result.min() < -1. or result.max() > 1:
            print("WARNING: image has values outside of range -1, 1, clipping:")
            print("         min: {} max: {}".format(result.min(), result.max()))
            result = np.clip(result, -1, 1)

        # convert back to ubyte for comparing
        return img_as_ubyte(result)

class BilateralDenoiser(SKImageDenoiser):
    """ Bilateral denoiser

    This denoiser is implemented in scikit-image and is based on the following reference:
    Reference: https://doi.org/10.1109/ICCV.1998.710815
    """
    name = "bilateral"
    description = "Bilateral Denoiser (skimage)"

    win_size = 3
    sigma_color = 0.1
    sigma_spatial = 1
    bins = 10000

    param_grid = {
        "win_size": range(3, 31, 2),
        "sigma_color": uniform(),
        "sigma_spatial": range(1, 30),
        "bins": range(5000, 50000)
    }

    def _denoise(self, image):
        return restoration.denoise_bilateral(image, win_size=self.win_size,
                                             sigma_color=self.sigma_color,
                                             sigma_spatial=self.sigma_spatial,
                                             bins=self.bins, multichannel=True)


class NLMeansDenoiser(SKImageDenoiser):
    """ Non-Linear Means denoiser

    This denoiser is implemented in scikit-image and is based on the following reference:
    Reference: Original: https://doi.org/10.1109/CVPR.2005.38
               Fast: https://doi.org/10.1109/ISBI.2008.4541250
    """
    name = "nlmeans"
    description = "Non-Linear Means Denoiser (skimage)"

    patch_size = 3
    patch_distance = 17
    h = 0.51

    param_grid = {
        "patch_size": range(3, 16, 2),
        "patch_distance": range(3, 30, 2),
        "h": uniform(),
    }

    def _denoise(self, image):
        sigma = restoration.estimate_sigma(image, average_sigmas=True, multichannel=True)
        return restoration.denoise_nl_means(image, sigma=sigma, fast_mode=False, multichannel=True)


class TVChambolleDenoiser(SKImageDenoiser):
    """  Total Variation Denoiser

    This denoiser is implemented in scikit-image and is based on the following reference:
    Reference: https://link.springer.com/article/10.1023/B:JMIV.0000011325.36760.1e
    """

    name = "tvchambolle"
    description = "Total Variation Denoiser (skimage)"

    weight = 0.1
    eps = 0.0002
    n_iter_max = 200

    param_grid = {
        "weight": uniform(),
        "eps": uniform(0., 0.01),
        "n_iter_max": range(100, 1000),
    }

    def _denoise(self, image):
        return restoration.denoise_tv_chambolle(image, multichannel=True, weight=self.weight,
                                                eps=self.eps, n_iter_max=self.n_iter_max)


class WaveletDenoiser(SKImageDenoiser):
    """ Wavelet denoiser

    This denoiser is implemented in scikit-image and is based on the following reference:
    References: https://doi.org/10.1109/83.862633
                https://doi.org/10.1093/biomet/81.3.425
    """

    name = "wavelet"
    description = "Wavelet Denoiser (skimage)"

    sigma = 0.80
    wavelet = "db6"
    mode = "soft"
    convert2ycbcr = False

    param_grid = {
        "sigma": uniform(),
        "wavelet": pywt.wavelist(kind="discrete"),
        "convert2ycbcr": (True, False),
    }

    def _denoise(self, image):
        return restoration.denoise_wavelet(image, multichannel=True, rescale_sigma=False,
                                           sigma=self.sigma, wavelet=self.wavelet, mode=self.mode,
                                           convert2ycbcr=self.convert2ycbcr)

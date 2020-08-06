from . import Denoiser
from skimage import restoration
from skimage import img_as_float, img_as_ubyte
import numpy as np
from abc import abstractmethod

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
    def _denoise(self, image):
        return restoration.denoise_bilateral(image, multichannel=True)


class NLMeansDenoiser(SKImageDenoiser):
    """ Non-Linear Means denoiser

    This denoiser is implemented in scikit-image and is based on the following reference:
    Reference: Original: https://doi.org/10.1109/CVPR.2005.38
               Fast: https://doi.org/10.1109/ISBI.2008.4541250
    """
    name = "nlmeans"
    description = "Non-Linear Means Denoiser (skimage)"

    def _denoise(self, image):
        #sigma = np.mean(restoration.estimate_sigma(image, multichannel=True))
        return restoration.denoise_nl_means(image, fast_mode=True, multichannel=True)


class TVChambolleDenoiser(SKImageDenoiser):
    """  Total Variation Denoiser

    This denoiser is implemented in scikit-image and is based on the following reference:
    Reference: https://link.springer.com/article/10.1023/B:JMIV.0000011325.36760.1e
    """

    name = "tvchambolle"
    description = "Total Variation Denoiser (skimage)"

    def _denoise(self, image):
        return restoration.denoise_tv_chambolle(image, multichannel=True)


class WaveletDenoiser(SKImageDenoiser):
    """ Wavelet denoiser

    This denoiser is implemented in scikit-image and is based on the following reference:
    References: https://doi.org/10.1109/83.862633
                https://doi.org/10.1093/biomet/81.3.425
    """

    name = "wavelet"
    description = "Wavelet Denoiser (skimage)"

    def _denoise(self, image):
        return restoration.denoise_wavelet(image, multichannel=True, rescale_sigma=True)

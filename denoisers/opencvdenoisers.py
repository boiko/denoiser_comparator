from . import Denoiser
import cv2
from scipy.stats import uniform


class FastNLMeansDenoiser(Denoiser):
    """ Non-Linear Means denoiser

    This denoiser is implemented in opencv and is based on the following reference:
    Reference: https://doi.org/10.1109/CVPR.2005.38
    """
    name = "fastnlmeans"
    description = "Fast Non-Linear Means Denoiser (OpenCV)"

    h = 14
    h_color = 29
    template_window_size = 3
    search_window_size = 23

    param_grid = {
        "h": range(3, 15),
        "h_color": range(9,30),
        "template_window_size": range(3, 18, 2),
        "search_window_size": range(7, 26, 2),
    }
    def denoise(self, image):
        return cv2.fastNlMeansDenoisingColored(image, h=self.h, hColor=self.h_color)

class BlurDenoiser(Denoiser):
    """ Regular blur filter

    This denoiser is implemented in opencv
    """
    name = "blur"
    description = "Blur smoothing filter"

    # the one and only param
    kernel_size = 11
    param_grid = {
        "kernel_size": range(3, 22, 2),
    }

    def denoise(self, image):
        return cv2.blur(image, ksize=(self.kernel_size,self.kernel_size))

class GaussianBlurDenoiser(Denoiser):
    """ Gaussian blur filter

    This denoiser is implemented in opencv
    """
    name = "gaussianblur"
    description = "Gaussian blur filter"

    # params
    kernel_size = 11
    sigma_x = 3.90
    sigma_y = 2.418

    param_grid = {
        "kernel_size": range(3, 28, 2),
        "sigma_x": uniform(loc=0, scale=30),
        "sigma_y": uniform(loc=0, scale=30),
    }

    def denoise(self, image):
        return cv2.GaussianBlur(image, ksize=(self.kernel_size, self.kernel_size),
                                sigmaX=self.sigma_x, sigmaY=self.sigma_y)

class MedianBlurDenoiser(Denoiser):
    """Median blur filter

    This denoiser is implemented in opencv
    """
    name = "medianblur"
    description = "Median blur filter"

    kernel_size = 15

    param_grid = {
        "kernel_size": range(3, 22, 2),
    }

    def denoise(self, image):
        return cv2.medianBlur(image, ksize=self.kernel_size)

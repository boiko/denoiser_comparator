from . import Denoiser
import cv2


class FastNLMeansDenoiser(Denoiser):
    """ Non-Linear Means denoiser

    This denoiser is implemented in scikit-image and is based on the following reference:
    Reference: https://doi.org/10.1109/CVPR.2005.38
    """
    name = "fastnlmeans"
    description = "Fast Non-Linear Means Denoiser (OpenCV)"

    def denoise(self, image):
        return cv2.fastNlMeansDenoisingColored(image)

class BlurDenoiser(Denoiser):
    """ Regular blur filter

    This denoiser is implemented in opencv
    """
    name = "blur"
    description = "Blur smoothing filter"

    def denoise(self, image):
        return cv2.blur(image, ksize=(5,5))

class GaussianBlurDenoiser(Denoiser):
    """ Gaussian blur filter

    This denoiser is implemented in opencv
    """
    name = "gaussianblur"
    description = "Gaussian blur filter"

    def denoise(self, image):
        return cv2.GaussianBlur(image, ksize=(5,5), sigmaX=0)

class MedianBlurDenoiser(Denoiser):
    """Median blur filter

    This denoiser is implemented in opencv
    """
    name = "medianblur"
    description = "Median blur filter"

    def denoise(self, image):
        return cv2.medianBlur(image, )
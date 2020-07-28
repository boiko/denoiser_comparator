from . import Noiser
from skimage.util import random_noise
from skimage import img_as_float, img_as_ubyte
import numpy as np

class SKImageNoiser(Noiser):
    """
    Synthetic noise creators implemented in skimage
    """

    description = "Implementation of SKImage noise generators"

    def __init__(self, noise):
        """
        initialize the class with the given noise type
        :param noise: the noise to be used. see :ref: skimage.util.random_noise
        """
        self.name = noise
        self._noise = noise

    def noise(self, image):
        # openCV images are BGR and skimage uses RGB, so invert the last and convert to float
        img = img_as_float(image[:, :, ::-1], force_copy=True)

        # and invert back to BGR for comparing
        result = random_noise(img, mode=self._noise, clip=True)[:, :, ::-1]

        # not sure why but on some images there are values outside the range -1 and 1
        if result.min() < -1. or result.max() > 1:
            print("WARNING: image has values outside of range -1, 1, clipping:")
            print("         min: {} max: {}".format(result.min(), result.max()))
            result = np.clip(result, -1, 1)

        # convert back to ubyte for comparing
        return img_as_ubyte(result)
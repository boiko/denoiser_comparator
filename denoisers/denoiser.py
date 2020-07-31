# Denoiser: base class for image denoisers

from abc import ABC, abstractmethod

class Denoiser(ABC):
    """ Base class for all denoiser objects

        Sub-classes need to implement the @ref denoise() """

    name = "Denoiser"
    description = "Base class for denoisers"

    @abstractmethod
    def denoise(self, image):
        """ Process the given @ref image and returns the processed image.
            Sub-classes need to implement this method.
            :type image: ndarray"""
        pass

    @classmethod
    def swap_bgr_rgb(cls, image):
        """
        Swap RGB and BGR images. This function does not check which is which, just do the swap
        :param image: Image to be swapped
        :return: swapped image
        """
        return image[:, :, ::-1]

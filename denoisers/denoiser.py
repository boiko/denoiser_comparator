# Denoiser: base class for image denoisers
# TODO: add copyright

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

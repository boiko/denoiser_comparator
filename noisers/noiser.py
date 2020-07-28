"""
Noiser: base class for syntehtic noise implementations
"""

from abc import ABC, abstractmethod

class Noiser(ABC):
    """ Base class for all noiser objects

        Sub-classes need to implement the @ref noise() """

    name = "Noiser"
    description = "Base class for noisers"

    @abstractmethod
    def noise(self, image):
        """ Add artificial noise to the given @ref image and returns the processed image.
            Sub-classes need to implement this method.
            :type image: ndarray"""
        pass
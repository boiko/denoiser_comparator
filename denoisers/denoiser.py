# Denoiser: base class for image denoisers

from abc import ABC, abstractmethod
import numpy as np
from metrics import default_metric

class Denoiser(ABC):
    """
    Base class for all denoiser objects

    Sub-classes need to implement the @ref denoise()
    """

    name = "Denoiser"
    description = "Base class for denoisers"

    param_grid = {}

    parallel = True

    def __init__(self, **kwargs):
        # if used from sklearn (via score) use a default metric
        self._metric = None
        self.set_params(**kwargs)

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

    # implement Scikit-learn estimator interface to make it easier to do grid search on denoisers
    def get_params(self, deep=False):
        return {p: getattr(self, p) for p in self.param_grid.keys()}

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def fit(self, noisy_images, ref_images):
        return self

    def predict(self, noisy_images):
        return np.array([self.denoise(noisy) for noisy in noisy_images])

    def score(self, noisy_images, ref_images):
        results = self.predict(noisy_images)

        if not self._metric:
            self._metric = default_metric()

        # return the average value of the default metric for the denoised images
        return np.array([self._metric.compare(ref, res) for ref, res in zip(ref_images, results)]).mean()

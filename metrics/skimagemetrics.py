from metrics import Metric
from skimage import metrics

import cv2
import numpy as np

class MeanSquaredError(Metric):
    """ Mean squared error metric

    This metric is implemented in scikit-image
    """
    name = "msqe"
    description = "Mean Squared Error Metric"

    def compare(self, imgref, imgtest):
        return metrics.mean_squared_error(imgref, imgtest)

class NormalizedRootMSE(Metric):
    """ Normalized Root Mean squared error metric

    This metric is implemented in scikit-image
    """
    name = "nrmse"
    description = "Normalized Root Mean Squared Error Metric"

    def compare(self, imgref, imgtest):
        return metrics.normalized_root_mse(imgref, imgtest)


class PeakSignalNoiseRatio(Metric):
    """ Peak signal noise ratio metric

    This metric is implemented in scikit-image
    """
    name = "psnr"
    description = "Peak Signal Noise Ratio"

    def compare(self, imgref, imgtest):
        return metrics.peak_signal_noise_ratio(imgref, imgtest)

class StructuralSimilarity(Metric):
    """ Structural Similarity metric

    This metric is implemented in scikit-image
    """
    name = "ssim"
    description = "Structural Similarity"

    def compare(self, imgref, imgtest):
        return metrics.structural_similarity(imgref, imgtest, multichannel=True)
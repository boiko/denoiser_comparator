# Metric: base class for image metrics
# TODO: add copyright

from abc import ABC, abstractmethod

class Metric(ABC):
    """ Base class for all metric objects

        Sub-classes need to implement the @ref compare() """

    name = "Metric"
    description = "Base class for metrics"

    @abstractmethod
    def compare(self, imgref, imgtest):
        """ Compare @ref image1 and @ref image2 and returns a summarized metric (a value)
            :type image1: ndarray
            :type image2: ndarray """
        pass
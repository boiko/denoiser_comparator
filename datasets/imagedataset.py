# ImageDataset: base class for image datasets
# TODO: add copyright

from abc import ABC, abstractmethod
import cv2
import os
import requests

class CropWindow(object):
    CROP_TOP = 1
    CROP_BOTTOM = 2
    CROP_LEFT = 4
    CROP_RIGHT = 8
    CROP_VCENTER = 16
    CROP_HCENTER = 32
    CROP_CENTER = CROP_VCENTER | CROP_HCENTER

    def __init__(self, width, height, position):
        self.width = width
        self.height = height
        self.position = position

    def crop_image(self, image):
        height, width, channels = image.shape

        x = y = 0
        if self.position & self.CROP_TOP:
            y = 0
        if self.position & self.CROP_BOTTOM:
            y = height - self.height
        if self.position & self.CROP_VCENTER:
            y = int((height - self.height) / 2)
        if self.position & self.CROP_LEFT:
            x = 0
        if self.position & self.CROP_RIGHT:
            x = width - self.width
        if self.position & self.CROP_HCENTER:
            x = int((width - self.width) / 2)

        return image[y:y+self.height, x:x+self.width]

class ImageDataset(ABC):
    """ Base class for all dataset objects

        Sub-classes need to implement the @ref image_triplets() """

    def __init__(self):
        super().__init__()

        self.crop_window = None

        # load the image triplets from sub-class
        self.triplets = self.image_triplets()

    @abstractmethod
    def image_triplets(self):
        """ Returns the triplet of (name, reference, noisy) for each image in the dataset.
            Sub-classes need to implement this method. """
        pass

    @abstractmethod
    def fetch(self, path):
        pass

    def download(self, url, local_path):
        """
        Downloads the given file to a local path
        :param url: the remote URL
        :param local_path: the local file path
        """
        print("Downloading {}".format(url))
        response = requests.get(url)
        with open(local_path, "wb") as local_file:
            local_file.write(response.content)

    def load_image(self, path):
        if not os.path.exists(path):
            # try to fetch it
            self.fetch(path)

        image = cv2.imread(path)
        if self.crop_window:
            image = self.crop_window.crop_image(image)
        return image

    def crop(self, width, height, position):
        self.crop_window = CropWindow(width, height, position)

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current < len(self.triplets):
            name, noisy, ref = self.triplets[self.current]
            self.current += 1
            return (name, self.load_image(noisy), self.load_image(ref))
        else:
            raise StopIteration


    def __len__(self):
        return len(self.triplets)
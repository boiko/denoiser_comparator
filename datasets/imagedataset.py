# ImageDataset: base class for image datasets
# TODO: add copyright

from abc import ABC, abstractmethod
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import os
import requests
import noisers
import glob
import pandas as pd

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

        If the images from the dataset are not local, the sub-class need to implement the
        :ref: fetch() method. All images will be downloading when creating an instance of the
        dataset in order to be able to perform sanity checks on them.

        Sub-classes need to implement the @ref image_triplets()

    """

    name = "Image dataset"
    description = "Base class for all image datasets"

    def __init__(self):
        super().__init__()
        self._noiser = None

        self.crop_window = None
        self.all_of_same_size = True
        self.metadata = None

        # load the image triplets from sub-class
        self._triplets = self.image_triplets()
        self._load_metadata()

    def _load_metadata(self):
        """
        Load metadata of all images and calculate metrics out of them
        """

        metadata = {
            "name": [],
            "width": [],
            "height": [],
        }
        for name, ref, noisy in self._triplets:
            if not os.path.exists(ref):
                # try to fetch it
                self.fetch(ref)

            ref_img = Image.open(ref)
            metadata["name"].append(name)
            metadata["width"].append(ref_img.width)
            metadata["height"].append(ref_img.height)

            if noisy:
                if not os.path.exists(noisy):
                    self.fetch(noisy)

                noisy_img = Image.open(noisy)
                if noisy_img.width != ref_img.width or noisy_img.height != noisy_img.height:
                    raise ValueError("""Reference and noisy images are not of same size:
                        * Ref: {}
                        * Noisy: {}""". format(ref, noisy))

        self.metadata = pd.DataFrame(metadata)

        # now check if all of the images have the same size
        if self.metadata.width.nunique() > 1 or self.metadata.height.nunique() > 1:
            print("Warning: Dataset has images of different size. Cropping to ({}x{}) for consistency" \
                  .format(self.metadata.width.min(), self.metadata.height.min()))
            self.crop(self.metadata.width.min(), self.metadata.height.min(), CropWindow.CROP_CENTER)

    @abstractmethod
    def image_triplets(self):
        """ Returns the triplet of (name, reference, noisy) for each image in the dataset.
            Sub-classes need to implement this method. """
        pass

    def fetch(self, path):
        """
        Fetches the image for the given local path. The default implementation does nothing.
        :param path: the path to the local file that is missing
        """
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
        if width > self.metadata.width.min() or height > self.metadata.height.min():
            print("WARNING: Dataset has images that are smaller than the requested crop window ({}x{})"\
                  .format(width, height))
            width = min(width, self.metadata.width.min())
            height = min(height, self.metadata.height.min())
            print("Using the following crop sizes instead: ({}x{})".format(width, height))
        self.crop_window = CropWindow(width, height, position)

    def __iter__(self):
        self._current = 0
        return self

    def __next__(self):
        if self._current < len(self._triplets):
            item =  self.__getitem__(self._current)
            self._current += 1
            return item
        else:
            raise StopIteration

    def __len__(self):
        return len(self._triplets)

    def __getitem__(self, item):
        if item < 0 or item >= len(self._triplets):
            raise ValueError("Out of range: {}".format(item))

        name, ref, noisy = self._triplets[item]

        # if the given dataset does not provide noisy images, use a synthetic noise generator
        if not self._noiser and not noisy:
            self._noiser = noisers.default_noiser()
            print("The given dataset does not provide noisy images. Using default noiser ({})" \
                  .format(self._noiser.name))

        ref_image = self.load_image(ref)
        # in case we are using a noiser, ignore the noisy path
        if self._noiser:
            noisy_image = self._noiser.noise(ref_image)
        else:
            noisy_image = self.load_image(noisy)

        return (name, noisy_image, ref_image)

    def set_noiser(self, noiser):
        """
        Use the given noiser to generate synthetic noise in the images.
        Even if the dataset is a symetric one (with both noise and reference) the noisy
        image will be generated using the given noiser.
        :param noiser: the :ref: Noiser to be used
        """
        self._noiser = noiser


class BasicImageDataset(ImageDataset):
    """
    Basic image dataset implementation that loads image from a given directory and use them as a
    dataset. Datasets of this kind are not symmetric as they don't provide noisy images. A
    synthetic noise generator needs to be used.
    """

    name = "basic_image_dataset"
    description = "Filesystem based asymetric image dataset"

    def __init__(self, path):
        """
        Creates an instance of this dataset collecting images from the given path
        :param path:
        """
        self._path = path
        self.name = os.path.basename(path)
        self._load_entries()
        super().__init__()

    def _load_entries(self):
        supported = [".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".webp",
                     ".pbm", ".pgm", ".ppm", ".pxm", ".pnm", ".pfm", ".sr", ".ras",
                     ".tiff", ".tif", ".exr", ".hdr", ".pic"]

        self._entries = []
        for entry in glob.glob(os.path.join(self._path, "*")):
            name, ext = os.path.splitext(entry)
            if not os.path.isfile(entry) or ext.lower() not in supported:
                continue

            self._entries.append((os.path.basename(name), entry, None))

    def image_triplets(self):
        return self._entries

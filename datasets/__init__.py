from .imagedataset import ImageDataset, CropWindow
from .natural_images import NaturalImageDataset

dataset_map = {
    "natural_images": NaturalImageDataset,
}

def list_datasets():
    return list(dataset_map.keys())

def create(dataset):
    if dataset not in dataset_map:
        raise ValueError(dataset)

    return dataset_map[dataset]()

from .imagedataset import ImageDataset, CropWindow
import os
import glob
from importlib import import_module
from inspect import isclass

dataset_map = {}

def dataset_factory(subdir):
    basename = os.path.basename(subdir)
    class_factory = None

    if not os.path.isdir(subdir) or basename.startswith("__py"):
        return (basename, class_factory)

    # now try to find a suitable class (inhering from ImageDataset) on the given path
    module = import_module(".{}".format(basename), "datasets")
    for name in dir(module):
        item = getattr(module, name, None)
        if item and isclass(item) and issubclass(item, ImageDataset):
            class_factory = item
            break
    return (basename, class_factory)

def list_datasets():
    # check for cached values
    if dataset_map:
        return list(dataset_map.keys())

    # check all subdirs for valid datasets
    for entry in glob.glob(os.path.join(os.path.dirname(__file__), "*")):
        name, class_factory = dataset_factory(entry)
        if class_factory is not None:
            dataset_map[name] = class_factory

    return list(dataset_map.keys())

def create(dataset):
    if dataset not in dataset_map:
        raise ValueError(dataset)

    return dataset_map[dataset]()

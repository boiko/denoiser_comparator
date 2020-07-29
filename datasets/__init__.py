from .imagedataset import ImageDataset, BasicImageDataset, CropWindow
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

    if not class_factory:
        # use a basic dataset for this dir
        class_factory = lambda: BasicImageDataset(subdir)

    return (basename, class_factory)

def list_datasets(with_description=False):
    if not dataset_map:
        # check all subdirs for valid datasets
        for entry in glob.glob(os.path.join(os.path.dirname(__file__), "*")):
            name, class_factory = dataset_factory(entry)
            if class_factory is not None:
                dataset_map[name] = {
                    "factory": class_factory,
                    "description": getattr(class_factory, "description",
                                           "Directory based dataset at {}".format(entry)),
                }

    if with_description:
        return [(k, v["description"]) for k, v in dataset_map.items()]

    return list(dataset_map.keys())

def create(dataset):
    if dataset not in dataset_map:
        raise ValueError(dataset)

    return dataset_map[dataset]["factory"]()

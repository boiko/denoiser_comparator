#!/usr/bin/env python3

from argparse import ArgumentParser
from matplotlib import pyplot as plt
from denoise_comparator import check_invalid, print_available
import datasets

def inspect_dataset(dataset):
    # just for the fun of it, plot the image size distribution to the
    axes = plt.subplot(2, 1, 1)
    dataset.metadata.plot(x="width", y="height", kind="scatter", ax=axes)
    plt.title("Image sizes")

    axes = plt.subplot(2, 2, 3)
    dataset.metadata.width.plot(kind="box", ax=axes)
    plt.title("Width boxplot")

    axes = plt.subplot(2, 2, 4)
    dataset.metadata.height.plot(kind="box", ax=axes)
    plt.title("Height boxplot")

    plt.tight_layout(pad=1.0)
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()

    # store the name of the default dataset
    default_dataset = datasets.list_datasets()[0]
    parser.add_argument("--list", action="store_true",
                        help="List available datasets")
    parser.add_argument("--dataset", action="store", default=default_dataset,
                        help="Dataset to be used (default: {})".format(default_dataset))
    options = parser.parse_args()

    if options.list:
        print_available("Available datasets:", datasets.list_datasets(with_description=True))
        exit(0)

    # sanity check if no wrong values were given
    the_datasets = check_invalid("datasets", [options.dataset], datasets.list_datasets())
    if not the_datasets:
        exit(1)

    the_dataset = datasets.create(the_datasets[0])
    inspect_dataset(the_dataset)
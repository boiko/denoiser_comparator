#!/usr/bin/env python3

from argparse import ArgumentParser
from matplotlib import pyplot as plt
from tqdm import tqdm
from denoise_comparator import check_invalid, print_available
import pandas as pd
import datasets
import metrics
import noisers

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

def inspect_metrics(dataset, the_metrics):
    # dict to be converted to dataframe
    results = {metric.name: [] for metric in the_metrics}
    results["name"] = []

    print("Comparing images to get metrics (this may take a few minutes)")
    for name, ref, noisy in tqdm(dataset, "Comparing noisy and ref"):
        results["name"].append(name)
        for metric in the_metrics:
            results[metric.name].append(metric.compare(ref, noisy))
        print(".", end="", flush=True)

    metric_df = pd.DataFrame(results)

    # add a main title
    plt.suptitle("Histogram of metrics between noisy ({}) and reference" \
                 .format(dataset.noiser.name if dataset.noiser else "from dataset"))

    # now generate a subplot with the histogram of all metrics
    rows = int(len(the_metrics)/4) + (1 if len(the_metrics) % 4 else 0)
    for metric, index in zip(the_metrics, range(0, len(the_metrics))):
        ax = plt.subplot(rows, 4, (index % 4) + 1)
        metric_df[metric.name].hist(ax=ax)
        plt.title(metric.name.upper())

if __name__ == "__main__":
    parser = ArgumentParser()

    # store the name of the default dataset
    default_dataset = datasets.list_datasets()[0]
    parser.add_argument("--list", action="store_true",
                        help="List available datasets, metrics and noisers")
    parser.add_argument("--dataset", action="store", default=default_dataset,
                        help="Dataset to be used (default: {})".format(default_dataset))
    parser.add_argument("--metrics", action="store", nargs="+", metavar=("METRIC1", "METRIC2"),
                        help="Metrics to be used to compare results (default: all)", default="all")
    parser.add_argument("--noiser", action="store",
                        help="Generate synthetic noise using the given noiser")
    options = parser.parse_args()

    if options.list:
        print_available("Available datasets:", datasets.list_datasets(with_description=True))
        print_available("Available metrics:", metrics.list_metrics(with_description=True) \
                        + [("all", "Use all metrics")])
        print_available("Available noisers:", noisers.list_noisers(with_description=True))
        exit(0)

    # sanity check if no wrong values were given
    the_datasets = check_invalid("datasets", [options.dataset], datasets.list_datasets())
    if not the_datasets:
        exit(1)

    options.metrics = check_invalid("metrics", options.metrics, metrics.list_metrics(), True)
    if not options.metrics:
        exit(1)

    if options.noiser:
        the_noisers = check_invalid("noisers", [options.noiser], noisers.list_noisers())
        if not the_noisers:
            exit(1)

    the_metrics = [metrics.create(m) for m in options.metrics]
    the_dataset = datasets.create(the_datasets[0])
    if options.noiser:
        the_dataset.set_noiser(noisers.create(options.noiser))

    inspect_dataset(the_dataset)

    plt.figure()
    inspect_metrics(the_dataset, the_metrics)

    plt.show()


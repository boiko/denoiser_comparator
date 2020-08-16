#!/usr/bin/env python3

from argparse import ArgumentParser
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from tqdm import tqdm
import denoisers
import datasets
import noisers
from denoise_comparator import check_invalid, print_available

if __name__ == "__main__":
    parser = ArgumentParser()

    # store the name of the default dataset
    default_dataset = datasets.list_datasets()[0]
    parser.add_argument("--list", action="store_true",
                        help="List available denoisers, datasets and noisers")
    parser.add_argument("--denoisers", action="store", nargs="+",
                        metavar=("DENOISER1", "DENOISER2"),
                        help="Choose which denoisers should be used (default: all)", default="all")
    parser.add_argument("--noiser", action="store",
                        help="Generate synthetic noise using the given noiser")
    parser.add_argument("--dataset", action="store", default=default_dataset,
                        help="Dataset to be used (default: {})".format(default_dataset))
    options = parser.parse_args()

    if options.list:
        print_available("Available denoisers:", denoisers.list_denoisers(with_description=True) \
                        + [("all", "Run all the available denoisers")])
        print_available("Available noisers:", noisers.list_noisers(with_description=True))
        print_available("Available datasets:", datasets.list_datasets(with_description=True))
        exit(0)

    # sanity check if no wrong values were given
    options.denoisers = check_invalid("denoisers", options.denoisers, denoisers.list_denoisers(),
                                      True)
    if not options.denoisers:
        exit(1)

    the_datasets = check_invalid("datasets", [options.dataset], datasets.list_datasets())
    if not the_datasets:
        exit(1)

    if options.noiser:
        the_noisers = check_invalid("noisers", [options.noiser], noisers.list_noisers())
        if not the_noisers:
            exit(1)

    the_denoisers = [denoisers.create(d) for d in options.denoisers]
    the_dataset = datasets.create(the_datasets[0])
    if options.noiser:
        the_dataset.set_noiser(noisers.create(options.noiser))

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # split the dataset between train and test and extract one random patch from each image
    the_dataset.crop(400, 400, datasets.CropWindow.CROP_RANDOM)
    train, test = train_test_split(list(tqdm(the_dataset, "Loading data")), train_size=0.7)

    print("Size of train: {} test: {}".format(len(train), len(test)))
    for name, ref, noisy in train:
        X_train.append(noisy)
        y_train.append(ref)

    for name, ref, noisy in test:
        X_test.append(noisy)
        y_test.append(ref)

    results = {}
    for denoiser in the_denoisers:
        if not denoiser.param_grid:
            print("WARNING: Denoiser {} has no parameter values to search".format(denoiser.name))
            continue

        print("Grid searching {} denoiser...".format(denoiser.name))
        grid = RandomizedSearchCV(estimator=denoiser, param_distributions=denoiser.param_grid,
                                n_jobs=-1, cv=3, verbose=1, n_iter=200)

        grid.fit(X_train, y_train)

        # evaluate the best params both on train and on test data
        denoiser.set_params(**grid.best_params_)
        results[denoiser.name] = {
            "params": grid.best_params_,
            "train_score": denoiser.score(X_train, y_train),
            "test_score": denoiser.score(X_test, y_test),
        }

    # print results
    for denoiser, data in results.items():
        print("-----------------------------------------------------------------------------------")
        print("Best parameters for denoiser {}:".format(denoiser))

        for key, value in data["params"].items():
            print("  * {}: {}".format(key, value))

        print("  --")
        print("  Train score: {}".format(data["train_score"]))
        print("  Test score: {}".format(data["test_score"]))



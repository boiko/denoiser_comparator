#!/usr/bin/env python3

import denoisers
import datasets
import metrics
import noisers
import cv2
import time
import pathlib
import json
from results import Results
from argparse import ArgumentParser
from tqdm import tqdm
from joblib import Parallel, delayed

def print_available(message, entries):
    print(message)
    for entry in entries:
        # check if the given entry is a name,description tuple
        if isinstance(entry, tuple):
            name, desc = entry
            print("  * {:15}{}".format(name, desc))
        else:
            print(" * {}".format(entry))
    print("")

def check_invalid(what, informed, available, allow_all=False):
    # if the user passes "all", ignore all the rest
    if allow_all and "all" in informed:
        return available

    invalid = [d for d in informed if d not in available]
    if invalid:
        if allow_all:
            available += ["all"]
        print_available("The specified {} ({}) are invalid. Available ones:".format(what, ", ".join(invalid)),
                        available)
        return []
    return informed

def prepare_output_dir(csv_file):
    csv_path = pathlib.Path(csv_file)
    output_dir = csv_path.parent / csv_path.name.replace(".csv", "")

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    return output_dir

def save_metadata(meta_file, dataset, noiser, denoisers, metrics, crop):
    meta = {}
    meta["dataset"] = dataset.name
    meta["noiser"] = noiser if noiser else "none"
    meta["denoisers"] = {}
    for denoiser in denoisers:
        meta[denoiser.name] = {p: getattr(denoiser, p, None) for p in denoiser.param_grid}
    meta["metrics"] = [m.name for m in metrics]
    meta["crop"] = {"width": crop[0], "height": crop[1]} if crop else None

    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=4)

def generate_batches(ds, size):
    for start in range(0, len(ds), size):
        yield ds[start:start + size]

def run(name, noisy, denoiser):
    tqdm.write("Image: {} denoiser: {}...".format(name, denoiser.name))
    start = time.time()
    denoisy = denoiser.denoise(noisy)
    end = time.time()
    duration = end - start
    return name, denoiser, denoisy, duration

if __name__ == "__main__":

    parser = ArgumentParser()

    # store the name of the default dataset
    default_dataset = datasets.list_datasets()[0]
    parser.add_argument("--list", action="store_true",
                        help="List available denoisers, datasets, noisers and metrics")
    parser.add_argument("--denoisers", action="store", nargs="+", metavar=("DENOISER1", "DENOISER2"),
                        help="Choose which denoisers should be used (default: all)", default="all")
    parser.add_argument("--noiser", action="store",
                        help="Generate synthetic noise using the given noiser")
    parser.add_argument("--dataset", action="store", default=default_dataset,
                        help="Dataset to be used (default: {})".format(default_dataset))
    parser.add_argument("--metrics", action="store", nargs="+", metavar=("METRIC1", "METRIC2"),
                        help="Metrics to be used to compare results (default: all)", default="all")
    parser.add_argument("--output", action="store", default="output.csv",
                        help="Output CSV file to store the results (default: output.csv)")
    parser.add_argument("--crop", nargs=2, metavar=("WIDTH", "HEIGHT"), type=int)
    parser.add_argument("--discard-images", action="store_true",
                        help="By default image results are saved to same folder/name as the output CSV file."
                             " Skip saving.")
    parser.add_argument("--parallel", action="store_true", default=False,
                        help="Run jobs in parallel. This might affect the runtime of the algorithms")
    options = parser.parse_args()

    if options.list:
        print_available("Available denoisers:", denoisers.list_denoisers(with_description=True) \
                        + [("all", "Run all the available denoisers")])
        print_available("Available noisers:", noisers.list_noisers(with_description=True))
        print_available("Available datasets:", datasets.list_datasets(with_description=True))
        print_available("Available metrics:", metrics.list_metrics(with_description=True) \
                        + [("all", "Use all metrics")])
        exit(0)

    # sanity check if no wrong values were given
    options.denoisers = check_invalid("denoisers", options.denoisers, denoisers.list_denoisers(), True)
    if not options.denoisers:
        exit(1)

    options.metrics = check_invalid("metrics", options.metrics, metrics.list_metrics(), True)
    if not options.metrics:
        exit(1)

    the_datasets = check_invalid("datasets", [options.dataset], datasets.list_datasets())
    if not the_datasets:
        exit(1)

    if options.noiser:
        the_noisers = check_invalid("noisers", [options.noiser], noisers.list_noisers())
        if not the_noisers:
            exit(1)

    the_denoisers = [denoisers.create(d) for d in options.denoisers]
    the_metrics = [metrics.create(m) for m in options.metrics]
    the_dataset = datasets.create(the_datasets[0])
    if options.crop:
        # crop at center by default
        the_dataset.crop(options.crop[0], options.crop[1], datasets.CropWindow.CROP_CENTER)

    if options.noiser:
        the_dataset.set_noiser(noisers.create(options.noiser))

    # just in case the user didn't provide the extension, add it
    if not options.output.endswith(".csv"):
        options.output = options.output + ".csv"

    print("Results are being saved to {}".format(options.output))
    output_dir = pathlib.Path(".")

    meta_file = options.output.replace(".csv", "_meta.json")
    print("Metadata will be saved to {}".format(meta_file))

    save_metadata(meta_file, the_dataset, options.noiser, the_denoisers, the_metrics, options.crop)
    if not options.discard_images:
        output_dir = prepare_output_dir(options.output)
        print("Images are being saved to {}".format(output_dir))

    results = Results(options.output)

    batch_size = 8 if options.parallel else 1
    n_jobs = -1 if options.parallel else 1

    pbar = tqdm(total=len(the_dataset) * len(the_denoisers))
    for batch in generate_batches(the_dataset, batch_size):
        result_images = {}
        batch_results = []
        jobs = []
        for name, reference, noisy in batch:
            result_images[name] = {
                "reference": reference,
                "noisy": noisy,
            }

            # store the metric values for the noisy images
            for metric in the_metrics:
                value = metric.compare(reference, noisy)
                results.append(name, None, metric, value, 0)


            sequential_denoisers = [d for d in the_denoisers if not d.parallel]
            parallel_denoisers = [d for d in the_denoisers if d.parallel]

            # for the non-parallel denoisers, just run them
            for denoiser in sequential_denoisers:
                batch_results.append(run(name, noisy, denoiser))
                pbar.update(1)

            for denoiser in parallel_denoisers:
                jobs.append((name, noisy, denoiser))

        batch_results += Parallel(n_jobs=n_jobs)(delayed(run)(name, noisy, denoiser) for name, noisy, denoiser in jobs)

        for name, denoiser, denoisy, duration in batch_results:
            result_images[name][denoiser.name] = denoisy

            for metric in the_metrics:
                value = metric.compare(result_images[name]["reference"], denoisy)
                results.append(name, denoiser, metric, value, duration)

        if not options.discard_images:
            for image_name, data in result_images.items():
                for key, img in data.items():
                    cv2.imwrite(str(output_dir / "{}_{}.png".format(image_name, key)), img)

        # the non-parallel denoisers have been accounted for already
        pbar.update(len(batch) * len(parallel_denoisers))
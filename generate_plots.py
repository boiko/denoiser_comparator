#!/usr/bin/env python3

import pandas as pd
import os
import seaborn as sbn
from matplotlib import pyplot as plt
from argparse import ArgumentParser


def plot_dataset_noise_shape(data, target_dir):
    noisy_metrics = data[data.denoiser == "none"]

    for metric, metric_data in noisy_metrics.groupby("metric"):
        fig = plt.figure()
        metric_data.value.hist(bins=20)
        plt.xlabel(metric.upper())
        plt.savefig(os.path.join(target_dir, "dataset_noise_{}.eps".format(metric)))
        plt.close(fig)

def scatter_denoisers_vs_noisy(data, target_dir):
    for metric, metric_data in data.groupby("metric"):
        # get the denoiser names
        denoisers = [d for d in metric_data.denoiser.unique() if d != "none"]
        pivot = metric_data.pivot_table(index="image", columns="denoiser", values="value")
        for denoiser in denoisers:
            fig = plt.figure()
            sbn.scatterplot(data = pivot, x=denoiser, y="none")
            plt.xlabel(denoiser)
            plt.ylabel("Imagem com ruído")
            plt.savefig(os.path.join(target_dir, "scatter_{}_noisy_vs_{}.eps".format(metric, denoiser)))
            plt.close(fig)

def plot_average_per_metric(data, target_dir):
    for metric, metric_data in data.groupby("metric"):
        metric_data = metric_data.copy()
        metric_data.denoiser = metric_data.denoiser.str.replace("none", "Imagem com ruído")
        pivot = metric_data.pivot_table(index="image", columns="denoiser", values="value")

        mean = pivot.mean()
        fig = plt.figure()
        ax = sbn.barplot(x=mean, y=mean.index)
        ax.set(xlabel=metric.upper(), ylabel="")

        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, "barplot_mean_{}.eps".format(metric)))
        plt.close(fig)

def plot_runtime(data, target_dir):
    # grab the runtime from one of the metrics
    metric = data.metric.unique()[0]
    denoiser_data = data[(data.denoiser != "none") & (data.metric == metric)]

    # now pivot the table to get the average runtime
    pivot = denoiser_data.pivot_table(index=["image", "metric"], columns="denoiser", values="time")
    mean = pivot.mean()
    fig = plt.figure()
    ax = sbn.barplot(x=mean, y=mean.index)
    ax.set(xlabel="Tempo médio (s)", ylabel="")
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, "runtime_average.eps"))
    plt.close(fig)

    # and the total time
    acc = pivot.sum()
    fig = plt.figure()
    ax = sbn.barplot(x=acc, y=acc.index)
    ax.set(xlabel="Tempo total (s)", ylabel="")
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, "runtime_total.eps"))
    plt.close(fig)


def prepare_target_dir(csv_file):
    dirname = csv_file.replace(".csv", "") + "_plots"
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    return dirname

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("csv_file", help="The CSV file with the results that should be plotted")
    options = parser.parse_args()

    # get the data
    data = pd.read_csv(options.csv_file, index_col=0)

    # prepare the target directory to store the plots
    target = prepare_target_dir(options.csv_file)

    # histogram of metrics between noisy and ref
    plot_dataset_noise_shape(data, target)

    # scatter of noisy x denoisers
    scatter_denoisers_vs_noisy(data, target)

    # average measurements of denoisers per metric
    plot_average_per_metric(data, target)

    # and the average runtime
    plot_runtime(data, target)

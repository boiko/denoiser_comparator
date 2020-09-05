#!/usr/bin/env python3

import pandas as pd
import os
import seaborn as sbn
from matplotlib import pyplot as plt
from argparse import ArgumentParser

class Figure:
    def __init__(self, basedir, name, formats):
        self.basedir = basedir
        self.name = name
        self.formats = formats

    def __enter__(self):
        self.fig = plt.figure()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for fmt in self.formats:
            self.fig.savefig(os.path.join(self.basedir, "{}.{}".format(self.name, fmt)))
        plt.close(self.fig)

class ResultPlotter:
    def __init__(self, csv_file, formats):
        self.csv_file = csv_file
        self.formats = formats
        # get the data
        self.data = pd.read_csv(options.csv_file, index_col=0)

        # prepare the target directory to store the plots
        self.target_path = self.prepare_target_dir()

    def figure(self, name):
        return Figure(self.target_path, name, self.formats)

    def plot_all(self):
        # histogram of metrics between noisy and ref
        self.plot_dataset_noise_shape()

        # scatter of noisy x denoisers
        self.scatter_denoisers_vs_noisy()

        # average measurements of denoisers per metric
        self.plot_average_per_metric()

        # the average runtime
        self.plot_runtime()

        # correlation between coeficients
        self.plot_correlation_metrics()

    def plot_dataset_noise_shape(self):
        data = self.data # for short
        noisy_metrics = data[data.denoiser == "none"]

        for metric, metric_data in noisy_metrics.groupby("metric"):
            with self.figure("dataset_noise_{}".format(metric)):
                metric_data.value.hist(bins=20)
                plt.xlabel(metric.upper())

    def scatter_denoisers_vs_noisy(self):
        data = self.data
        for metric, metric_data in data.groupby("metric"):
            # get the denoiser names
            denoisers = [d for d in metric_data.denoiser.unique() if d != "none"]
            pivot = metric_data.pivot_table(index="image", columns="denoiser", values="value")
            for denoiser in denoisers:
                with self.figure("scatter_{}_noisy_vs_{}".format(metric, denoiser)):
                    sbn.scatterplot(data = pivot, x=denoiser, y="none")
                    plt.xlabel(denoiser)
                    plt.ylabel("Imagem com ruído")

    def plot_average_per_metric(self):
        data = self.data
        for metric, metric_data in data.groupby("metric"):
            metric_data = metric_data.copy()
            metric_data.denoiser = metric_data.denoiser.str.replace("none", "Imagem com ruído")
            pivot = metric_data.pivot_table(index="image", columns="denoiser", values="value")

            print(pivot.describe())
            mean = pivot.mean()
            with self.figure("barplot_mean_{}".format(metric)):
                ax = sbn.barplot(x=mean, y=mean.index)
                ax.set(xlabel=metric.upper(), ylabel="")
                plt.tight_layout()

    def plot_runtime(self):
        data = self.data
        # grab the runtime from one of the metrics
        metric = data.metric.unique()[0]
        denoiser_data = data[(data.denoiser != "none") & (data.metric == metric)]

        # now pivot the table to get the average runtime
        pivot = denoiser_data.pivot_table(index=["image", "metric"], columns="denoiser", values="time")
        mean = pivot.mean()

        with self.figure("runtime_average"):
            ax = sbn.barplot(x=mean, y=mean.index)
            ax.set(xlabel="Tempo médio (s)", ylabel="")
            plt.tight_layout()

        # and the total time
        acc = pivot.sum()
        with self.figure("runtime_total"):
            ax = sbn.barplot(x=acc, y=acc.index)
            ax.set(xlabel="Tempo total (s)", ylabel="")
            plt.tight_layout()

        with self.figure("runtime_hist"):
            legend = []
            for denoiser, den_data in data[data.denoiser != "none"].groupby("denoiser"):
                legend.append(denoiser)
                plt.hist(den_data[den_data.metric == "psnr"].time)
            plt.legend(legend, ncol=2)

    def plot_correlation_metrics(self):
        data = self.data
        pivot = data.pivot_table(index=["image", "denoiser"], columns="metric", values="value")

        with self.figure("heatmap_metric_corr"):
            sbn.heatmap(pivot.corr(), annot=True)

    def prepare_target_dir(self):
        dirname = self.csv_file.replace(".csv", "") + "_plots"
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        return dirname

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--format", "-f", action="append")
    parser.add_argument("csv_file", help="The CSV file with the results that should be plotted")
    options = parser.parse_args()

    global formats
    formats = options.format

    if not formats:
        formats = ["eps"]

    plotter = ResultPlotter(options.csv_file, formats)
    plotter.plot_all()
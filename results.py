import pandas as pd

class Results(object):
    """ Helper class for storing result entries """
    def __init__(self, filename, print=False):
        super().__init__()

        self.filename = filename
        self.print = print

        # each result will be added to this
        self.results = {
            "image": [],
            "denoiser": [],
            "metric": [],
            "value": [],
            "time": [],
        }

    def append(self, image, denoiser, metric, value, time):
        self.results["image"].append(image)
        self.results["denoiser"].append(denoiser.name if denoiser else "none")
        self.results["metric"].append(metric.name)
        self.results["value"].append(value)
        self.results["time"].append(time)

        if self.print:
            print("{} {} {}: {} ({})".format(image, denoiser.name if denoiser else "none",
                                        metric.name, value, time))

        self.save()

    def save(self):
        # save partial results, just in case
        dataframe = pd.DataFrame(self.results)
        dataframe.to_csv(self.filename)
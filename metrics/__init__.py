from .metric import Metric
from .skimagemetrics import MeanSquaredError, NormalizedRootMSE, PeakSignalNoiseRatio, StructuralSimilarity


metrics = [
    MeanSquaredError,
    NormalizedRootMSE,
    PeakSignalNoiseRatio,
    StructuralSimilarity,
]

def metric_map():
    return {m.name: {"factory": m, "description": m.description} for m in metrics}

def list_metrics(with_description=False):
    if with_description:
        return [(k, v["description"]) for k, v in metric_map().items()]

    return list(metric_map().keys())

def create(metric):
    if metric not in metric_map():
        raise ValueError(metric)

    return metric_map()[metric]["factory"]()

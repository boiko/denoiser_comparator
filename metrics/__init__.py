from .metric import Metric
from .skimagemetrics import MeanSquaredError, NormalizedRootMSE, PeakSignalNoiseRatio, StructuralSimilarity


metric_map = {
    MeanSquaredError.name: MeanSquaredError,
    NormalizedRootMSE.name: NormalizedRootMSE,
    PeakSignalNoiseRatio.name: PeakSignalNoiseRatio,
    StructuralSimilarity.name: StructuralSimilarity,
}

def list_metrics():
    return list(metric_map.keys())

def create(metric):
    if metric not in metric_map:
        raise ValueError(metric)

    return metric_map[metric]()

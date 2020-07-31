from .denoiser import Denoiser
import importlib

denoiser_map = {
    # skimage ones
    "nlmeans": {
        "factory": (".skimagedenoisers", "NLMeansDenoiser"),
        "description": "Non-local means denoising",
    },
    "bilateral": {
        "factory": (".skimagedenoisers", "BilateralDenoiser"),
        "description": "Denoise image using bilateral filter",
    },
    "tvchambolle": {
        "factory": (".skimagedenoisers", "TVChambolleDenoiser"),
        "description": "Total-variation denoising",
    },
    "wavelet": {
        "factory": (".skimagedenoisers", "WaveletDenoiser"),
        "description": "Perform wavelet denoising on an image",
    },
    # opencv ones
    "fastnlmeans": {
        "factory": (".opencvdenoisers", "FastNLMeansDenoiser"),
        "description": "Fast non-local means denoising",
    },
    "blur": {
        "factory": (".opencvdenoisers", "BlurDenoiser"),
        "description": "Blurs an image using the normalized box filter",
    },
    "gaussianblur": {
        "factory": (".opencvdenoisers", "GaussianBlurDenoiser"),
        "description": "Blurs an image using a Gaussian filter",
    },
    "medianblur": {
        "factory": (".opencvdenoisers", "MedianBlurDenoiser"),
        "description": "Blurs an image using the median filter",
    },
    # other non-CNN denoisers
    "bm3d": {
        "factory": (".bm3ddenoiser", "BM3DDenoiser"),
        "description": "Exact Transform-Domain Noise Variance for Collaborative Filtering of Stationary Correlated Noise",
    },
    # deep learning stuff
    "deepimageprior": {
        "factory": (".deepimageprior", "DeepImagePrior"),
        "description": "Deep Image prior denoiser",
    },
}

def list_denoisers(with_description=False):
    if with_description:
        return [(k, v["description"]) for k, v in denoiser_map.items()]

    return list(denoiser_map.keys())

def create(denoiser):
    if denoiser not in denoiser_map:
        raise ValueError(denoiser)

    modulename, denoiserclass = denoiser_map[denoiser]["factory"]
    module = importlib.import_module(modulename, "denoisers")

    the_class = getattr(module, denoiserclass)
    return the_class()

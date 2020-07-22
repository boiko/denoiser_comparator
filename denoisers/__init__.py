from .denoiser import Denoiser
import importlib

denoiser_map = {
    # skimage ones
    "nlmeans": (".skimagedenoisers", "NLMeansDenoiser"),
    "bilateral": (".skimagedenoisers", "BilateralDenoiser"),
    "tvchambolle": (".skimagedenoisers", "TVChambolleDenoiser"),
    "wavelet": (".skimagedenoisers", "WaveletDenoiser"),
    # opencv ones
    "fastnlmeans": (".opencvdenoisers", "FastNLMeansDenoiser"),
    "blur": (".opencvdenoisers", "BlurDenoiser"),
    "gaussianblur": (".opencvdenoisers", "GaussianBlurDenoiser"),
    "medianblur": (".opencvdenoisers", "MedianBlurDenoiser"),
    # deep learning stuff
    "deepimageprior": (".deepimageprior", "DeepImagePrior"),
}

def list_denoisers():
    return list(denoiser_map.keys())

def create(denoiser):
    if denoiser not in denoiser_map:
        raise ValueError(denoiser)

    modulename, denoiserclass = denoiser_map[denoiser]
    module = importlib.import_module(modulename, "denoisers")

    the_class = getattr(module, denoiserclass)
    return the_class()

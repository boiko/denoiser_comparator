from .noiser import Noiser
from .skimagenoiser import SKImageNoiser


noiser_map = {
    "gaussian": {
        "factory": lambda: SKImageNoiser("gaussian"),
        "description": "Gaussian-distributed additive noise",
    },
    "localvar": {
        "factory": lambda: SKImageNoiser("localvar"),
        "description": "Gaussian-distributed noise, with local variance at each point of image",
    },
    "poisson": {
        "factory": lambda: SKImageNoiser("poisson"),
        "description": "Poisson-distributed noise generated from the data",
    },
    "salt": {
        "factory": lambda: SKImageNoiser("salt"),
        "description": "Replaces random pixels with 1",
    },
    "pepper": {
        "factory": lambda: SKImageNoiser("pepper"),
        "description": "Replaces random pixels with 0 (for unsigned images) or -1 (for signed ones)",
    },
    "snp": {
        "factory": lambda: SKImageNoiser("s&p"),
        "description": "Apply salt and pepper noise to random pixels",
    },
    "speckle": {
        "factory": lambda: SKImageNoiser("speckle"),
        "description": "Multiplicative noise using out = image + n*image",
    },
}

def list_noisers(with_description=False):
    if with_description:
        return [(k, v["description"]) for k, v in noiser_map.items()]

    return list(noiser_map.keys())

def create(noiser):
    if noiser not in noiser_map:
        raise ValueError(noiser)

    return noiser_map[noiser]()

def default_noiser():
    """
    Returns a reasonable synthetic noise genrator by default.
    :return: an instance of the given :ref: Noiser
    """
    return create("gaussian")

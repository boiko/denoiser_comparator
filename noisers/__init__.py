from .noiser import Noiser
from .skimagenoiser import SKImageNoiser


noiser_map = {
    "gaussian": lambda: SKImageNoiser("gaussian"),
    "localvar": lambda: SKImageNoiser("localvar"),
    "poisson": lambda: SKImageNoiser("poisson"),
    "salt": lambda: SKImageNoiser("salt"),
    "pepper": lambda: SKImageNoiser("pepper"),
    "snp": lambda: SKImageNoiser("s&p"),
    "speckle": lambda: SKImageNoiser("speckle"),
}

def list_noisers():
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

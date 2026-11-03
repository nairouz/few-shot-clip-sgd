# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Upstream dataset code: CoOp; MIT (see licenses/CoOp-MIT.txt).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenet import ImageNet
from .imagenet_sketch import ImageNetSketch
from .imagenet_v2     import ImageNetV2
from .imagenet_a      import ImageNetA
from .imagenet_r      import ImageNetR
from typing import Any
import os 

dataset_list = {
                "oxford_pets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "fgvc": FGVCAircraft,
                "food101": Food101,
                "oxford_flowers": OxfordFlowers,
                "stanford_cars": StanfordCars,
                "imagenet": ImageNet,
                "imagenet_sketch": ImageNetSketch,
                "imagenet_v2"    : ImageNetV2,
                "imagenet_a"     : ImageNetA,
                "imagenet_r"     : ImageNetR,
                }

def build_dataset(name, root, shots, preprocess):
    root = os.path.expanduser(root)      # ← add this line
    cls = dataset_list[name]
    try:
        return cls(root, shots, preprocess)
    except TypeError:
        return cls(root, shots)


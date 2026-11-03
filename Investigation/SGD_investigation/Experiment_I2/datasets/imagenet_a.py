# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Upstream dataset code: CoOp; MIT (see licenses/CoOp-MIT.txt).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

# datasets/imagenet_a.py
import os, glob, os.path as osp
from .utils import DatasetBase, Datum, listdir_nohidden
from .imagenet import (imagenet_classes, ImagenetClassname2Idx, get_wnid2idx)

_TEMPLATE = ["a photo of a {}."]

class ImageNetA(DatasetBase):
    dataset_dir = "imagenet-adversarial"          # ~/datasets/imagenet-adversarial
    template    = _TEMPLATE

    def __init__(self, root, num_shots, preprocess, test_preprocess=None):
        self.root = os.path.expanduser(root)
        if test_preprocess is None:
            test_preprocess = preprocess

        train, val, test = self.read_data()
        super().__init__(train, val, test)
        self._test          = test                 # preserve test split
        self.num_shots      = num_shots
        self.preprocess     = preprocess
        self.test_preprocess= test_preprocess
        self._classnames    = imagenet_classes

    # ----------------------------------------------------------
    def read_data(self):
        img_dir  = osp.join(self.root, self.dataset_dir, "imagenet-a")
        if not osp.isdir(img_dir):                 # fallback: data one level up
            img_dir = osp.join(self.root, self.dataset_dir)

        wnid2idx = get_wnid2idx(self.root)         # for WNID folders
        data = []

        for folder in listdir_nohidden(img_dir, sort=True):
            # 1) WNID → direct lookup
            if folder in wnid2idx:
                label = wnid2idx[folder]
                cname = imagenet_classes[label]
            else:
                # 2) try readable classname with underscores
                cname = folder.replace("_", " ")
                if cname not in ImagenetClassname2Idx:
                    continue                       # skip unknown folder
                label = ImagenetClassname2Idx[cname]

            for impath in glob.glob(osp.join(img_dir, folder, "*")):
                data.append(Datum(impath=impath, label=label, classname=cname))

        return data, [], data                     # train, val, test

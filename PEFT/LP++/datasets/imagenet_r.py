# datasets/imagenet_r.py
import os, glob, os.path as osp
from .utils import DatasetBase, Datum, listdir_nohidden
from .imagenet import (imagenet_classes, ImagenetClassname2Idx, get_wnid2idx)          # ← NEW: we also use WNID map

_TEMPLATE = ["a photo of a {}."]

class ImageNetR(DatasetBase):
    dataset_dir = "imagenet-rendition"
    template    = _TEMPLATE

    def __init__(self, root, num_shots, preprocess, test_preprocess=None):
        self.root = os.path.expanduser(root)
        if test_preprocess is None:
            test_preprocess = preprocess

        train, val, test = self.read_data()
        super().__init__(train, val, test)
        self._test           = test            # keep test split
        self.num_shots       = num_shots
        self.preprocess      = preprocess
        self.test_preprocess = test_preprocess
        self._classnames     = imagenet_classes

    # ----------------------------------------------------------
    def read_data(self):
        # accept both layouts
        img_dir = osp.join(self.root, self.dataset_dir, "imagenet-r")
        if not osp.isdir(img_dir):
            img_dir = osp.join(self.root, self.dataset_dir)   # fallback

        wnid2idx = get_wnid2idx(self.root)      # WNID → idx
        data = []

        for folder in listdir_nohidden(img_dir, sort=True):
            # 1) WNID folders (common)
            if folder in wnid2idx:
                label = wnid2idx[folder]
                cname = imagenet_classes[label]
            else:
                # 2) readable class name with underscores
                cname = folder.replace("_", " ")
                if cname not in ImagenetClassname2Idx:
                    continue                    # skip unknown folder
                label = ImagenetClassname2Idx[cname]

            for impath in glob.glob(osp.join(img_dir, folder, "*")):
                data.append(Datum(impath=impath, label=label, classname=cname))
                
        return data, [], data                  # train, val, test

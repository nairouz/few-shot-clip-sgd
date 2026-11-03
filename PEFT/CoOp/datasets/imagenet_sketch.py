# datasets/imagenet_sketch.py
import os.path as osp, glob, os
from .utils import DatasetBase, Datum, listdir_nohidden
from .imagenet import imagenet_classes, get_wnid2idx

_TEMPLATE = ["a photo of a {}."]

class ImageNetSketch(DatasetBase):
    dataset_dir = "imagenet-sketch"
    template    = _TEMPLATE

    def __init__(self, root, num_shots, preprocess, test_preprocess=None):
        self.root = os.path.expanduser(root)     # ~/datasets
        if test_preprocess is None:
            test_preprocess = preprocess

        train, val, test = self.read_data()      # ← WILL return 3 lists
        super().__init__(train, val, test)

        # store attributes AFTER super().__init__
        self._test = test  
        self.num_shots        = num_shots
        self.preprocess       = preprocess
        self.test_preprocess  = test_preprocess
        self._classnames      = imagenet_classes  # private slot

    # ----------------------------------------------------------
    def read_data(self):
        img_dir  = osp.join(self.root, self.dataset_dir, "images")
        wnid2idx = get_wnid2idx(self.root)       # lazy mapping (built once)

        data = []
        for wnid in listdir_nohidden(img_dir, sort=True):
            if wnid not in wnid2idx:             # safety guard
                continue
            label = wnid2idx[wnid]               # 0‑999
            cname = imagenet_classes[label]
            for impath in glob.glob(osp.join(img_dir, wnid, "*")):
                data.append(Datum(impath=impath, label=label, classname=cname))

        # *** NEVER return None ***
        return data, [], data                   # train, val, test

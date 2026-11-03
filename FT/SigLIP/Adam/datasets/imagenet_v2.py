# datasets/imagenet_v2.py
import os.path as osp, glob, os
from .utils import DatasetBase, Datum, listdir_nohidden
from .imagenet import imagenet_classes

_TEMPLATE = ["a photo of a {}."]

class ImageNetV2(DatasetBase):
    dataset_dir = "imagenetv2"
    template    = _TEMPLATE

    def __init__(self, root, num_shots, preprocess, test_preprocess=None):
        self.root = os.path.expanduser(root)
        if test_preprocess is None:
            test_preprocess = preprocess

        train, val, test = self.read_data()
        super().__init__(train, val, test)
        self._test          = test                  # keep test split
        self.num_shots      = num_shots
        self.preprocess     = preprocess
        self.test_preprocess= test_preprocess
        self._classnames    = imagenet_classes      # ← private slot (✓)

    # ----------------------------------------------------------
    def read_data(self):
        img_dir = osp.join(
            self.root, self.dataset_dir,
            "imagenetv2-matched-frequency-format-val"
        )
        data = []
        for cls_id in listdir_nohidden(img_dir, sort=True):   # '0000' … '0999'
            label = int(cls_id)
            cname = imagenet_classes[label]
            for impath in glob.glob(osp.join(img_dir, cls_id, "*")):
                data.append(Datum(impath=impath, label=label, classname=cname))
        return data, [], data           # train, val, test  (val intentionally empty)
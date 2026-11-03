# Dataset preparation

The experiments use the same datasets and predefined train/validation/test splits described in the [SO dataset guide](https://github.com/nairouz/SO/blob/main/DATASETS.md). That guide provides the dataset downloads, annotations, and fixed split files. Image files and annotations are external inputs.

The data root is the parent directory passed through `--root_path`. Some folder names in the supplied loaders differ from the SO guide. The table below maps the SO layout to the paths expected by this repository. Symbolic links allow an existing SO installation to be reused without copying images or changing split files.

| Dataset (`--dataset`) | Folder in the SO guide | Folder expected here | Images within that folder | Fixed split or annotations |
| --- | --- | --- | --- | --- |
| [Caltech101](https://github.com/nairouz/SO/blob/main/DATASETS.md#caltech101) (`caltech101`) | `caltech-101` | `Caltech101` | `101_ObjectCategories/` | `split_zhou_Caltech101.json` |
| [DTD](https://github.com/nairouz/SO/blob/main/DATASETS.md#dtd) (`dtd`) | `dtd` | `DTD` | `images/` | `split_zhou_DescribableTextures.json` |
| [EuroSAT](https://github.com/nairouz/SO/blob/main/DATASETS.md#eurosat) (`eurosat`) | `eurosat` | `eurosat` | `2750/` | `split_zhou_EuroSAT.json` |
| [FGVC-Aircraft](https://github.com/nairouz/SO/blob/main/DATASETS.md#fgvcaircraft) (`fgvc`) | `fgvc_aircraft` | `fgvc_aircraft` | `images/` | `variants.txt`; `images_variant_train.txt`; `images_variant_val.txt`; `images_variant_test.txt` |
| [Food101](https://github.com/nairouz/SO/blob/main/DATASETS.md#food101) (`food101`) | `food-101` | `Food101` | `images/` | `split_zhou_Food101.json` |
| [ImageNet](https://github.com/nairouz/SO/blob/main/DATASETS.md#imagenet) (`imagenet`) | `imagenet` | `imagenet` | `train/<class>/`; `val/<class>/` | Original ImageNet partitions; class subdirectories required by `torchvision.datasets.ImageFolder` |
| [Flowers102](https://github.com/nairouz/SO/blob/main/DATASETS.md#flowers102) (`oxford_flowers`) | `oxford_flowers` | `Flower102` | `jpg/` | `split_zhou_OxfordFlowers.json` |
| [OxfordPets](https://github.com/nairouz/SO/blob/main/DATASETS.md#oxfordpets) (`oxford_pets`) | `oxford_pets` | `OxfordPets` | `images/` | `split_zhou_OxfordPets.json` |
| [StanfordCars](https://github.com/nairouz/SO/blob/main/DATASETS.md#stanfordcars) (`stanford_cars`) | `stanford_cars` | `StanfordCars` | Paths relative to the dataset folder, as specified in the split JSON | `split_zhou_StanfordCars.json` |
| [SUN397](https://github.com/nairouz/SO/blob/main/DATASETS.md#sun397) (`sun397`) | `sun397` | `SUN397` | `SUN397/` | `split_zhou_SUN397.json` |
| [UCF101](https://github.com/nairouz/SO/blob/main/DATASETS.md#ucf101) (`ucf101`) | `ucf101` | `UCF101` | `UCF-101-midframes/` | `split_zhou_UCF101.json` |

## Reusing an SO installation

For a complete SO installation on a case-sensitive Linux filesystem, the following commands create the required aliases. `DATA_ROOT` identifies the existing dataset parent directory. The `-T` option prevents an existing destination directory from being treated as a link container; existing paths are not overwritten.

```bash
DATA_ROOT=/absolute/path/to/datasets

ln -sT caltech-101 "$DATA_ROOT/Caltech101"
ln -sT dtd "$DATA_ROOT/DTD"
ln -sT food-101 "$DATA_ROOT/Food101"
ln -sT oxford_flowers "$DATA_ROOT/Flower102"
ln -sT oxford_pets "$DATA_ROOT/OxfordPets"
ln -sT stanford_cars "$DATA_ROOT/StanfordCars"
ln -sT sun397 "$DATA_ROOT/SUN397"
ln -sT ucf101 "$DATA_ROOT/UCF101"
ln -sT images/train "$DATA_ROOT/imagenet/train"
ln -sT images/val "$DATA_ROOT/imagenet/val"
```

No aliases are needed for EuroSAT or FGVC-Aircraft. An installation that already uses the expected paths needs no additional links. Experiments use the same parent directory through `--root_path "$DATA_ROOT"`.

The ImageNet aliases account for the SO layout, which stores images in `imagenet/images/train/` and `imagenet/images/val/`. This repository's loader reads `imagenet/train/` and `imagenet/val/`. Both directories contain class subfolders. The loader samples the support and small validation subsets from the training partition. The original ImageNet validation partition supplies the test set.

## Splits and few-shot sampling

The supplied `split_zhou_*.json` files retain the fixed train/validation/test partitions from the SO preparation procedure. Their top-level keys are `train`, `val`, and `test`. Each entry is `[relative_image_path, zero_based_label, class_name]`. The relative image path is resolved against the image directory in the table, except StanfordCars, which resolves it against the dataset folder. FGVC-Aircraft uses its supplied variant annotation files.

Few-shot support sets are sampled from the predefined training partition using the shot count and seed specified by each experiment. The [shot-scaling](../Investigation/ShotScaling/README.md) and [SigLIP](../FT/SigLIP/README.md) guides specify their dataset subsets, supported shot counts, and model preprocessing.

Base-to-new evaluation additionally reads `<dataset>_seed_<seed>.json` from `--split_dir`, with `base` and `new` class-index lists. The entry point expects those files to exist. The `make_b2n_splits.py` helper referenced by the error message is not included, so those class-partition files must be prepared separately.

## Domain-generalization datasets

The optional domain-generalization loaders use the SO guide's directory names directly:

| `--dataset` | Folder below the data root | Images within that folder | Preparation guide |
| --- | --- | --- | --- |
| `imagenet_v2` | `imagenetv2` | `imagenetv2-matched-frequency-format-val/` | [ImageNetV2](https://github.com/nairouz/SO/blob/main/DATASETS.md#imagenetv2) |
| `imagenet_sketch` | `imagenet-sketch` | `images/` | [ImageNet-Sketch](https://github.com/nairouz/SO/blob/main/DATASETS.md#imagenet-sketch) |
| `imagenet_a` | `imagenet-adversarial` | `imagenet-a/` | [ImageNet-A](https://github.com/nairouz/SO/blob/main/DATASETS.md#imagenet-a) |
| `imagenet_r` | `imagenet-rendition` | `imagenet-r/` | [ImageNet-R](https://github.com/nairouz/SO/blob/main/DATASETS.md#imagenet-r) |

The corresponding `datasets/imagenet_*.py` files define the loading behavior.

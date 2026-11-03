# Dataset preparation

The data root is the parent directory passed through `--root_path`. The exact folder names are case-sensitive. Image files and annotations are external inputs.

| `--dataset` | Folder below the data root | Images relative to that folder | Required split or annotation files |
| --- | --- | --- | --- |
| `caltech101` | `Caltech101` | `101_ObjectCategories/` | `split_zhou_Caltech101.json` |
| `dtd` | `DTD` | `images/` | `split_zhou_DescribableTextures.json` |
| `eurosat` | `eurosat` | `2750/` | `split_zhou_EuroSAT.json` |
| `fgvc` | `fgvc_aircraft` | `images/` | `variants.txt`; `images_variant_train.txt`; `images_variant_val.txt`; `images_variant_test.txt` |
| `food101` | `Food101` | `images/` | `split_zhou_Food101.json` |
| `imagenet` | `imagenet` | `train/<class>/`; `val/<class>/` | `torchvision.datasets.ImageFolder` layout |
| `oxford_flowers` | `Flower102` | `jpg/` | `split_zhou_OxfordFlowers.json` |
| `oxford_pets` | `OxfordPets` | `images/` | `split_zhou_OxfordPets.json` |
| `stanford_cars` | `StanfordCars` | Paths specified relative to `StanfordCars/` in the JSON | `split_zhou_StanfordCars.json` |
| `sun397` | `SUN397` | `SUN397/` | `split_zhou_SUN397.json` |
| `ucf101` | `UCF101` | `UCF-101-midframes/` | `split_zhou_UCF101.json` |

The JSON split format follows `datasets/oxford_pets.py`: the top-level keys are `train`, `val`, and `test`. Each entry is `[relative_image_path, zero_based_label, class_name]`. The loaders sample few-shot support sets from these predefined splits.

For ImageNet, the loader samples the support and small validation subsets from `imagenet/train/`. The original ImageNet validation images in `imagenet/val/` supply the test set in this protocol.

Base-to-new evaluation additionally reads `<dataset>_seed_<seed>.json` from `--split_dir`, with `base` and `new` class-index lists. The entry point expects those split files to exist. The `make_b2n_splits.py` helper referenced by the error message is not included, so these files must be prepared separately.

Domain-generalization loaders are available for `imagenet_v2`, `imagenet_a`, `imagenet_r`, and `imagenet_sketch`. Their expected directory structures are specified in the corresponding `datasets/imagenet_*.py` files.

The [shot-scaling](../Investigation/ShotScaling/README.md) and [SigLIP](../FT/SigLIP/README.md) experiments use the corresponding dataset subsets and local loaders. Their guides specify the supported dataset names, shot counts, and model preprocessing.

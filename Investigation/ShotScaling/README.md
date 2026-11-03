# CLIP shot scaling

This implementation compares SGD, Adam, and Adam-M0 across six datasets and five shot levels. The augmented grid contains 270 runs: six datasets × five shot counts × three seeds × three optimizers. The optimizer variants are `sgd`, `adam_raw`, and `adam_m0`. A separate no-augmentation protocol covers 54 runs.

Training uses CLIP ViT-B/16, 30 epochs, nominal batch size 64, and the fixed optimizer settings defined by the implementation. Diagnostic measurements use paired class-balanced support/reference batches. The training loop retains the benchmark minibatch protocol.

## Training

The following command runs from this directory:

```bash
python main.py --dataset eurosat --shots 1 --seed 1 \
  --optimizer_variant sgd --protocol augmented --batch_size 64 --epochs 30 \
  --root_path /absolute/path/to/datasets
```

`python main.py --help` describes the remaining options. The dataset grid is EuroSAT, Oxford-Pets, DTD, FGVC-Aircraft, Stanford-Cars, and Caltech101. Shot counts are 1, 2, 4, 8, and 16, with seeds 1–3.

## Analysis

The [`analysis/`](analysis/) scripts aggregate generated training runs and produce shot-scaling plots. Their input and output paths are explicit command-line arguments. The [results guide](../../docs/RESULTS.md) describes the numerical exports in `results/inputs/` and the table calculations in `scripts/analysis/`, both relative to the repository root.

[Dataset preparation](../../docs/DATASETS.md) specifies the image and split layouts. The CLIP loader requires pretrained weights and downloads them when absent from its cache.

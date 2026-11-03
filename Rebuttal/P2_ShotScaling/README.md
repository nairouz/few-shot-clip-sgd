# P2: shot scaling

This package contains the training, diagnostic, and analysis source bundled with the P2 raw-result archive. The augmented comparison has 270 runs: six datasets × five shot counts × three seeds × three optimizers. The three optimizer variants are `sgd`, `adam_raw`, and `adam_m0`. The archive also includes a separate no-augmentation bridge protocol.

Training uses CLIP ViT-B/16, 30 epochs, nominal batch size 64, and the fixed optimizer settings defined by the package. The diagnostic protocol uses paired class-balanced support/reference batches. Full-support training is not substituted for the benchmark minibatch protocol.

From this directory:

```bash
python main.py --dataset eurosat --shots 1 --seed 1 \
  --optimizer_variant sgd --protocol augmented --batch_size 64 --epochs 30 \
  --root_path /absolute/path/to/datasets
```

`python main.py --help` describes the remaining options. The source is preserved without scientific changes. Dataset images and pretrained weights are external inputs. Cluster orchestration and internal deployment documents are omitted. The included numerical exports and camera-ready aggregation scripts are in the parent `results/` and `scripts/` directories.

# SigLIP adaptation

This implementation compares dense SGD, Adam, and Adam-M0 and provides zero-shot evaluation and analysis. It uses `google/siglip-base-patch16-224` at revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`.

| Directory | Condition |
| --- | --- |
| [`SGD/`](SGD/) | Dense transformer-block adaptation with SGD |
| [`Adam/`](Adam/) | Dense transformer-block adaptation with Adam |
| [`AdamM0/`](AdamM0/) | Adam with a fixed multiplier calibrated to SGD's first update |
| [`ZeroShot/`](ZeroShot/) | Zero-shot SigLIP evaluation |
| [`analysis/`](analysis/) | Accuracy, preconditioning, and update-budget aggregation and plots |
| [`checkpoint/`](checkpoint/) | Pinned model metadata and file checksums |

## Environment and checkpoint

The [checkpoint manifest](checkpoint/checkpoint_manifest.json) records transformers 4.52.4, huggingface_hub 0.32.1, and sentencepiece 0.2.2. The remaining dependencies are listed in the repository-level [requirements](../../requirements.txt). These recorded versions can be installed in the experiment environment:

```bash
python -m pip install transformers==4.52.4 huggingface_hub==0.32.1 sentencepiece==0.2.2
```

Each run receives an explicit local model snapshot through `--model_id` and uses offline loading. The checkpoint manifest lists the required files and checksums. From the repository root, the verification command is:

```bash
python FT/SigLIP/SGD/tools/verify_checkpoint.py \
  --model_dir /absolute/path/to/pinned/siglip/snapshot
```

## Training

A dense-SGD run from the repository root uses:

```bash
python FT/SigLIP/SGD/main.py \
  --dataset eurosat --shots 1 --seed 1 --epochs 30 \
  --root_path /absolute/path/to/datasets \
  --model_id /absolute/path/to/pinned/siglip/snapshot \
  --offline --device cuda --output_root /absolute/path/to/output
```

The corresponding entry points for the other conditions are `FT/SigLIP/Adam/main.py`, `FT/SigLIP/AdamM0/main.py`, and `FT/SigLIP/ZeroShot/main.py`. Each parser exposes its options through `--help`. Training uses dense transformer-block adaptation, native SigLIP preprocessing, and fixed epoch-30 evaluation. The nominal batch size is 64.

The accuracy grid covers EuroSAT, FGVC-Aircraft, DTD, and Oxford-Pets at 1, 4, and 16 shots with seeds 1–3. Mechanism diagnostics cover EuroSAT and DTD at 1 and 16 shots. The [results guide](../../docs/RESULTS.md) links the numerical exports and table-generation scripts. [Dataset preparation](../../docs/DATASETS.md) specifies the image and split layouts.

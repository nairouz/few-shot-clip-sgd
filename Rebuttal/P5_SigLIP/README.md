# P5: SigLIP transfer

This package contains the source bundled with the P5 SigLIP result archive. It includes dense SGD, raw Adam, Adam-M0, zero-shot evaluation, and analysis code. The pretrained model is `google/siglip-base-patch16-224` at revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`.

The checkpoint manifest records the required files and checksums. Its original workstation location has been removed; upstream identifiers and file checksums are unchanged. Each run receives an explicit local snapshot through `--model_id` and uses offline loading.

The manifest records transformers 4.52.4, huggingface_hub 0.32.1, and sentencepiece 0.2.2. The other Python dependencies are listed in the repository-level requirements. A separate environment can retain these recorded versions:

```bash
python -m pip install transformers==4.52.4 huggingface_hub==0.32.1 sentencepiece==0.2.2
```

A dense-SGD run from `P5_DenseSGD/` uses:

```bash
python main.py --dataset eurosat --shots 1 --seed 1 --epochs 30 \
  --root_path /absolute/path/to/datasets \
  --model_id /absolute/path/to/pinned/siglip/snapshot \
  --offline --device cuda --output_root /absolute/path/to/output
```

The analogous entry points are `P5_Adam/main.py`, `P5_AdamM0/main.py`, and `P5_ZeroShot/main.py`. Each parser exposes its own options through `--help`. The source preserves the supplied transformer-block trainable scope, native SigLIP preprocessing, and epoch-30 evaluation protocol. The numerical exports distinguish the full accuracy grid from the narrower mechanism-diagnostic subset.

Cluster submission code, internal deployment documents, and hardware-dependent tests are omitted. No new training run was performed while preparing this release.

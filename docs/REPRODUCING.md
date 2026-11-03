# Reproducing the experiments

## Scope of this release

The source follows the supplied experimental implementation. The main benchmark and the diagnostic experiments have different entry points and configurations. In `FT/SGD/train.py` and `FT/Adam/train.py`, `mark_as_trainable` activates attention weight matrices and the two MLP weight matrices of the selected transformer blocks. With `--encoder both --position all`, these matrices are trained in every block of both encoders. Biases, normalization parameters, embeddings, final projection matrices, and logit scale remain frozen. This is the precise implementation of dense transformer-weight adaptation used by these entry points.

The SGD entry point constructs `torch.optim.SGD` with its default zero momentum and zero weight decay. The Adam entry point uses `torch.optim.Adam`. The three AdamW directories encode weight decay in their names. The methods retain their individual argument parsers and training loops.

## Installation and data

`requirements.txt` lists imported third-party packages. It is not a version lockfile. A compatible CUDA PyTorch / torchvision installation is needed by the training and GPU-memory profiling code. Dependencies and pretrained weights must be available before an experiment starts. The local CLIP loader downloads weights when they are absent from its cache.

The command-line `--root_path` option points to the dataset parent directory. `DATASETS_ROOT`, when set, takes precedence in the supplied `resolve_root` helper. `CLIP_DOWNLOAD_ROOT` configures the diagnostic cache loaders and the Adam benchmark loader. The other benchmark loaders retain CLIP's default `~/.cache/clip` location. [DATASETS.md](DATASETS.md) lists the expected directory names and annotations.

## Isolated benchmark runs

The launcher in `scripts/run_experiment.py` invokes the supplied `main.py` with a distinct working directory. This separation matters because original result and checkpoint filenames identify the method, backbone, shot count, and seed, but do not always include the learning rate. Each learning-rate trial therefore needs a different `--output-dir`.

```bash
python scripts/run_experiment.py \
  --method Adam \
  --output-dir results/dtd/adam/vit-b16/1shot/seed1/lr1e-4 \
  --root_path /absolute/path/to/datasets \
  --dataset dtd --backbone ViT-B/16 \
  --shots 1 --seed 1 --lr 1e-4 \
  --max_epochs 30 --batch_size 64 \
  --encoder both --position all --use_aug
```

LoRA has its own optimizer argument and rank:

```bash
python scripts/run_experiment.py \
  --method LoRA \
  --output-dir results/dtd/lora-adam/vit-b16/16shot/seed1/lr1e-4 \
  --root_path /absolute/path/to/datasets \
  --dataset dtd --backbone ViT-B/16 \
  --shots 16 --seed 1 --lr 1e-4 \
  --max_epochs 30 --batch_size 64 \
  --encoder both --position all --optimizer Adam --rank 2 --use_aug
```

The LoRA code is the supplied historical implementation. Its zero-dropout linear path uses temporary weight merging during the forward pass. It has been retained to preserve experiment provenance. Changing its forward computation would constitute a different implementation and would require a separate numerical comparison.

The launcher also supports the remaining methods listed by `python scripts/run_experiment.py --help`. Method-specific options are documented by each method's `main.py --help` after installation. The zero-shot parser does not accept training-only flags such as `--shots`, `--lr`, or `--max_epochs`.

## Main benchmark and longer trajectories

The paper's primary comparison covers seeds 1, 2, and 3; shots 1, 2, 4, 8, and 16; and CLIP ViT-B/16, ViT-B/32, and ViT-L/14. The main comparison uses learning rate `1e-4`. The robustness grid contains `1e-2`, `1e-3`, `1e-4`, and `1e-5`. Main runs use 30 epochs and batch size 64. Longer trajectories use `--max_epochs 300` with the experiment-specific setting reported in the paper.

For non-ImageNet datasets, `--use_aug` enables the supplied random resized crop and horizontal flip. Without it, those loaders use CLIP preprocessing. The supplied ImageNet loader defines its training augmentation internally. Its behavior is therefore not switched by the main script's `--use_aug` flag.

The benchmark scripts report final-checkpoint results. The main SGD and Adam loops set `VALIDATION = False`; they do not perform a validation-based learning-rate or checkpoint search. Result aggregation and any selection used for a particular paper table must follow that table's documented protocol. Test accuracy files alone do not provide an independent validation-selection record.

## Outputs

An isolated run contains `run_command.json`, `console.log`, the implementation's result files, timing/memory files, and any checkpoint it saves. `results_with_aug/` and `results_without_aug/` are the main result subdirectories. Checkpoints default to `models_with_aug/` or `models_without_aug/`.

## Diagnostic experiments

[EXPERIMENTS.md](EXPERIMENTS.md) maps every diagnostic package to its measurement script. Full-support training is an explicit diagnostic option, `--full_support_train`. It is distinct from the minibatch benchmark. Reference-set caps, second-moment estimators, checkpoint schedules, and update budgets are experiment-specific.

## Validation of this release

`python scripts/validate_source.py` checks Python syntax and benchmark entry-point presence without importing dependencies. The release also checks launcher command construction and local documentation paths. These checks do not constitute a dataset-dependent reproduction or a GPU benchmark. No new accuracy measurements are claimed by this release.

from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
import clip
from datasets import build_dataset
from datasets.utils import build_data_loader, DatasetWrapper
from p2.diagnostic_sampler import build_diagnostic_manifest, diagnostic_seed
from p2.logging_utils import write_json_with_hash
from p2.support_manifest import image_id, manifest
from run_utils import get_arguments, set_random_seed
from train import run_p2

def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

def restore_rng(state):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    if state[3] is not None: torch.cuda.set_rng_state_all(state[3])

class P2DatasetWrapper(DatasetWrapper):
    """Plain DatasetWrapper plus the image ID required for order auditing."""
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, image_id(self.data_source[index])

def main():
    args = get_arguments()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    set_random_seed(args.seed)
    model, preprocess = clip.load(args.backbone, args.device, download_root=args.clip_cache)
    # This call occurs at exactly the Plain point after seeding/model loading.
    dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
    saved = rng_state()
    full_dataset = build_dataset(args.dataset, args.root_path, -1, preprocess)
    restore_rng(saved)
    support = manifest(dataset.train_x, split="support", dataset=args.dataset, shots=args.shots, seed=args.seed)
    validation = manifest(dataset.val, split="validation", dataset=args.dataset, shots=args.shots, seed=args.seed)
    full = manifest(full_dataset.train_x, split="full_reference", dataset=args.dataset, shots=args.shots, seed=args.seed)
    if not {x["image_path"] for x in support} <= {x["image_path"] for x in full}:
        raise RuntimeError("support is not a subset of the full training reference")
    run_id = f"{args.protocol}__{args.optimizer_variant}__{args.dataset}__k{args.shots}__s{args.seed}__ViT-B-16"
    run_dir = Path(args.result_root) / args.protocol / run_id
    # An incomplete/FAILED directory is intentionally resumable. Valid DONE
    # directories are skipped by the worker before invoking this entry point.
    run_dir.mkdir(parents=True, exist_ok=True)
    args.run_id, args.run_dir = run_id, str(run_dir)
    args.support_hash = write_json_with_hash(run_dir/"support_manifest.json", support)
    args.validation_hash = write_json_with_hash(run_dir/"validation_manifest.json", validation)
    args.full_reference_hash = write_json_with_hash(run_dir/"full_reference_manifest.json", full)
    diag_sizes = [16]
    if (not args.smoke and args.protocol=="augmented"
            and args.dataset in {"eurosat","dtd","fgvc","stanford_cars"}
            and args.shots in {1,16} and args.optimizer_variant in {"sgd","adam_raw"}):
        diag_sizes += [8,32]
    args.diagnostic_manifests = {}
    for size in diag_sizes:
        diag = build_diagnostic_manifest(dataset.train_x, full_dataset.train_x, batch_size=size,
            seed=diagnostic_seed(args.dataset,args.shots,args.seed,size))
        name = f"diagnostic_manifest_b{size}.json"
        args.diagnostic_manifests[size] = (diag, write_json_with_hash(run_dir/name, diag))
    train_tfm = preprocess
    if args.protocol == "augmented":
        train_tfm = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(.08,1.), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(.5), transforms.ToTensor(),
            transforms.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = build_data_loader(dataset.train_x, batch_size=64, tfm=train_tfm, is_train=True,
        shuffle=True, num_workers=args.num_workers, dataset_wrapper=P2DatasetWrapper)
    run_p2(args, model, preprocess, dataset, full_dataset, train_loader, generator)

if __name__ == "__main__":
    main()

# Third-party notices

Original project contributions by Nairouz Mrabah are licensed under the [MIT License](LICENSE). The release preserves the supplied source headers and credits. Third-party code retains its original authorship and license terms. The project-author header identifies Nairouz Mrabah as the author of this project, not as the original author of bundled third-party code. The following license texts apply to the corresponding third-party components and their upstream material.

| Component | Location | Upstream reference | Included terms |
| --- | --- | --- | --- |
| OpenAI CLIP | Bundled `clip/` implementations and tokenizer vocabulary | [openai/CLIP](https://github.com/openai/CLIP) | [MIT](licenses/CLIP-MIT.txt) |
| Microsoft LoRA | `PEFT/LoRA/loralib/` and `PEFT/PiSSA/loralib/` identify loralib as a source | [microsoft/LoRA](https://github.com/microsoft/LoRA) | [MIT](licenses/LoRA-MIT.txt) |
| CoOp dataset code and split conventions | Bundled dataset loaders use the corresponding few-shot split conventions | [KaiyangZhou/CoOp](https://github.com/KaiyangZhou/CoOp) | [MIT](licenses/CoOp-MIT.txt) |
| GaLore | `PEFT/GaLoRE/optimizers/` | [jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore) | [Apache 2.0](licenses/GaLore-Apache-2.0.txt) |

The LoRA layer headers also credit Baijiong Lin and link to `Baijiong-Lin/MOML`. The attention adaptation source credits `KyanChen/MakeMultiHeadNaive`; those comments remain in place. `PEFT/SAFT/optimizers/SparseAdam.py` retains its Apache-2.0 SPDX notice. External dependencies, datasets, and pretrained model weights are subject to their own terms. License texts for upstream components do not relicense unrelated files.

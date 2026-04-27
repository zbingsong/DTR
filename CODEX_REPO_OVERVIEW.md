# CODEX_REPO_OVERVIEW

## 1. Summary

Observed: This repository is a compact research codebase for paired histopathology image translation across several modality pairs, including the HEMIT variant that translates H&E imagery to mIHC-like outputs. The operational code is concentrated in top-level scripts such as `train.py`, `predict.py`, `Attention_GAN.py`, `batch_utils.py`, and dataset adapters under `datasets/`.

Observed: The architecture is shallow rather than package-oriented. A single attention U-Net generator in `Attention_GAN.py` is reused across datasets, dataset-specific loaders normalize differently shaped paired image corpora into a shared `{'input', 'gt', 'name'}` format, and `reglib/reg.py` provides deformation-field models used only in training-time misalignment regularization.

Strongly inferred: The repository’s primary implemented workflow is paired patch translation rather than true slide-scale inference, because `train.py` and `predict.py` both consume cropped PNG pairs through `DataLoader`s and no OpenSlide, WSI tiling, region reading, or stitching code is present. For HEMIT specifically, `datasets/hemit.py`, `predict_hemit.sh`, and `play_with_the_pretrained_model.ipynb` show 1024×1024 paired images cropped to 512×512 patches and passed directly through `Attention_GAN.Generator`.

## 2. Architecture

Observed: The codebase follows a research-script layout centered on a few top-level runtime files.

- `train.py` is the training orchestrator. It parses CLI arguments, builds the generator and discriminator from `Attention_GAN.py`, instantiates two registration networks plus a spatial transformer from `reglib/reg.py`, creates dataset loaders via `batch_utils.py`, computes losses, evaluates PSNR, and saves checkpoints.
- `predict.py` is the inference/export orchestrator. It reuses the same dataset abstraction and generator architecture but omits the discriminator and registration models. It writes concatenated input / generated / target PNG outputs with OpenCV.
- `batch_utils.py` is the dataset dispatch layer. It maps string dataset names such as `hemit`, `af2he`, `aperio`, `cuhk`, and `he2pas` to concrete dataset classes in `datasets/`.
- `datasets/` contains dataset-specific filesystem and cropping assumptions. `datasets/hemit.py` is the HEMIT-specific adapter.
- `Attention_GAN.py` contains the actual translation network: an attention-gated U-Net-like generator and a CNN discriminator.
- `reglib/` contains deformation-field and warping code used to make training tolerant to misalignment. It is part of training, visualization, and deformation export, but not part of `predict.py`.
- `downstream_tasks/` is a secondary evaluation subsystem that treats saved prediction PNGs as horizontally concatenated HE|IHC image pairs, extracts frozen ResNet-50 features, and trains downstream classifiers or linear probes.
- `play_with_the_pretrained_model.ipynb` and `visualize.ipynb` are notebook-side demos. They help clarify intended usage but do not introduce a separate software architecture.

Observed: The repository is not organized as an installable Python package. Imports are direct file imports, and the shell scripts export `PYTHONPATH` pointing at the author’s external directories.

## 3. Key Components

- `Patch translation runtime` — coordinates dataset loading, model construction, training, evaluation, and checkpointing. Main files: `train.py`, `predict.py`, `batch_utils.py`. Observed.
- `Generator and discriminator` — the generator is an attention-gated U-Net-style network with skip attention blocks and Tanh output; the discriminator is a CNN with global average pooling and a small MLP head. Main file: `Attention_GAN.py`. Observed.
- `HEMIT data adapter` — loads paired `input` and `label` PNGs from `data_root/<phase>/...`, applies optional affine perturbation to the input side during training, crops 1024×1024 images to 512×512 patches for HEMIT, flips training samples, and returns normalized tensors in `[-1, 1]`. Main file: `datasets/hemit.py`. Observed.
- `Misalignment regularization subsystem` — predicts dense deformation fields between generated and target images (`RegGT`) and between input and generated or target images (`RegX`), then warps images with `Transformer_2D` to compute reconstruction and smoothness penalties. Main files: `reglib/reg.py`, `train.py`, `get_deformation_field.py`, `visualize.ipynb`. Observed.
- `Dataset-specific wrappers` — unify several paired translation datasets with differing folder layouts and crop sizes into the common runtime contract. Main files: `datasets/af2he.py`, `datasets/aperio.py`, `datasets/cuhk.py`, `datasets/he2pas.py`. Observed.
- `Downstream evaluation pipeline` — consumes saved concatenated prediction images, splits them back into HE and IHC halves, extracts frozen ImageNet ResNet-50 features, and trains a downstream classifier or linear probe. Main files: `downstream_tasks/dataset.py`, `downstream_tasks/generate_feature.py`, `downstream_tasks/model.py`, `downstream_tasks/train.py`, `downstream_tasks/linear_prob.py`. Observed.
- `Notebook demos` — show single-image checkpoint loading and visualization, not batch or WSI orchestration. Main files: `play_with_the_pretrained_model.ipynb`, `visualize.ipynb`. Observed.
- `Peripheral or presently unused code` — `swin_transformer.py` is present but no import path from the main training or inference scripts was observed. Strongly inferred based on repository-wide import search showing no callers outside the file itself.

## 4. Dependency Graph

- `train.py` -> `batch_utils.py`, `Attention_GAN.py`, `metrics.py`, `pytorch_ssim/__init__.py`, `reglib/reg.py`
- `predict.py` -> `batch_utils.py`, `Attention_GAN.py`, `cv2`, `torch.utils.data.DataLoader`
- `batch_utils.py` -> `datasets/af2he.py`, `datasets/hemit.py`, `datasets/cuhk.py`, `datasets/aperio.py`, `datasets/he2pas.py`
- `datasets/*.py` -> `datasets/random_affine.py`, `PIL`, `torchvision.transforms.functional`
- `get_deformation_field.py` -> `Attention_GAN.py`, `reglib/reg.py`
- `visualize.ipynb` -> `Attention_GAN.py`, `reglib/reg.py`
- `play_with_the_pretrained_model.ipynb` -> `Attention_GAN.py`, `weights/*.pth`, `assets/*`
- `downstream_tasks/generate_feature.py` -> `downstream_tasks/dataset.py`, `downstream_tasks/model.py`
- `downstream_tasks/train.py` -> `downstream_tasks/dataset.py`, `downstream_tasks/model.py`
- `downstream_tasks/linear_prob.py` -> `downstream_tasks/dataset.py`, `downstream_tasks/model.py`

## 5. Execution Flow

### Primary flow

1. `train.py` parses CLI flags such as `--dataset_name`, `--data_root`, `--crop_size`, `--noise`, and checkpoint settings. Observed.
2. `train.py` constructs dataset loaders by calling `UNI(args.dataset_name, ...)` from `batch_utils.py`, which delegates to the dataset-specific adapter. Observed.
3. For HEMIT, `datasets/hemit.py` enumerates paired files under `data_root/<phase>/input` and `data_root/<phase>/label`, applies optional `RandomAffine` perturbations only to the input branch during training, crops patches, applies random flips during training, and normalizes tensors to `[-1, 1]`. Observed.
4. `train.py` builds `Attention_GAN.Generator`, `Attention_GAN.Discriminator`, `reg.Reg` for `RegGT`, `reg.Reg` for `RegX`, and `reg.Transformer_2D`. Observed.
5. Each training batch feeds `batch['input']` through the generator to produce `rec`, predicts a deformation field from `rec` to `gt` with `RegGT`, warps `rec` with `Transformer_2D`, and computes reconstruction, SSIM, smoothness, identity-mesh, and GAN losses. Observed.
6. After an initial warmup (`decouple_iteration = 100`), `train.py` also updates `RegX` to model the deformation relationship between input/generated and input/target pairs. Observed.
7. On evaluation epochs, `train.py` runs the generator on the validation loader, converts outputs from `[-1, 1]` to `[0, 1]`, computes PSNR with `metrics.easy_psnr`, logs the result, and saves `netG`, `netD`, `RegGT`, and `RegX` checkpoints. Observed.

### Secondary flows

- `Patch inference / export` — `predict.py` parses similar CLI flags, instantiates only `Attention_GAN.Generator`, loads a checkpoint, runs over the dataset in test mode, converts tensors back to image space, concatenates input, generated, and target images horizontally, and writes PNGs to `results_dir`. Observed.
- `HEMIT-focused inference` — `predict_hemit.sh` calls `predict.py` with `--dataset_name hemit`, `--crop_size 512`, and `--noise 0`; `datasets/hemit.py` test mode uses a centered 512×512 crop from each 1024×1024 paired image. Observed. Strongly inferred: This is the repo’s closest implementation of “predicting mIHC from H&E WSIs,” but it operates on pre-extracted or pre-aligned image tiles rather than reading WSIs directly.
- `Deformation inspection` — `get_deformation_field.py` and `visualize.ipynb` load saved generator and registration checkpoints and export or visualize deformation fields and warped reconstructions. Observed.
- `Downstream pathology classification` — `downstream_tasks/generate_feature.py` extracts frozen ResNet-50 features from prediction PNGs, and `downstream_tasks/train.py` or `downstream_tasks/linear_prob.py` train classifiers over HE-only or HE+IHC features. Observed.

### Notebooks

- `play_with_the_pretrained_model.ipynb` is a simple single-image demo for `af2he`, `he2pas`, `aperio`, and `hemit`. It loads PNG samples from `assets/*`, loads generator checkpoints from `weights/*.pth`, performs one forward pass, and plots input/target/output. Observed.
- `visualize.ipynb` is a deformation-visualization notebook for the registration-enhanced training setup. Observed.
- Unknown: No notebook cell implementing WSI tiling, slide region reading, or reconstructed full-slide mosaics was found.

## 6. Data Flow

Observed: The main translation pipeline is paired-image and tensor-centric.

- Data ingress:
  `datasets/hemit.py` expects `data_root/train|val|test/input/*.png` and `.../label/*.png`.
  Other dataset adapters expect different directory conventions such as nested slide/patch folders or JSON pair lists.
- Preprocessing:
  PIL images are loaded as RGB.
  Training loaders optionally perturb only the source image with `datasets/random_affine.py`.
  Dataset loaders crop fixed-size patches and optionally apply random horizontal and vertical flips during training.
- Tensorization:
  All translation datasets convert images with `torchvision.transforms.functional.to_tensor` and scale from `[0, 1]` to `[-1, 1]`.
- Model input/output:
  `Attention_GAN.Generator` receives a 3-channel source patch and emits a 3-channel translated patch through a final `Tanh`.
- Training-only intermediate flow:
  `RegGT` predicts a dense 2-channel deformation field from generated and target patches.
  `Transformer_2D` warps the generated image and produces a validity mask.
  `RegX` predicts auxiliary deformation fields for the identity / decoupling losses.
- Persistence:
  `train.py` writes checkpoints plus validation montage images.
  `predict.py` writes horizontally concatenated PNG triptychs: input | generated | target.
  `get_deformation_field.py` writes `.npy` deformation fields.
  `downstream_tasks/generate_feature.py` writes extracted feature tensors to `extracted_features.pth`.

Strongly inferred: The downstream evaluation pipeline assumes the translation outputs remain concatenated montages rather than separate HE and IHC files, because `downstream_tasks/dataset.py` splits each PNG exactly in half along width.

## 7. Configuration

Observed: Runtime configuration is CLI-driven rather than config-file-driven.

- `train.py` and `predict.py` use `argparse` for all major settings.
- Dataset choice is a required string key interpreted by `batch_utils.py`.
- Data layout assumptions are hard-coded inside each dataset adapter rather than externalized.
- Shell scripts such as `train_hemit.sh` and `predict_hemit.sh` act as documented parameter presets for each dataset.

Observed: Noise / misalignment augmentation is controlled numerically (`--noise 0..5`) and maps to `torchvision.transforms.RandomAffine` settings in `datasets/random_affine.py`.

Observed: `tools.py` defines a `ConfigObj` class with hard-coded values, but no import path from the main scripts was observed.

Strongly inferred: Effective configuration precedence is “CLI flags first, then dataset-internal constants,” because the main scripts only pass CLI values into dataset classes and model constructors, while crop origins, folder names, center-crop rules, and test/train mode normalization are defined internally in `datasets/*.py`.

## 8. Environment Setup

### Documented

Observed: `README.md` documents the following setup:

- `conda create --name DTR python=3.9`
- `conda activate DTR`
- `pip install -r requirements.txt`

Observed: The current user prompt states that the conda environment already exists and can be activated with `conda activate DTR`.

### Inferred

Observed: `requirements.txt` pins a lightweight Python stack centered on `torch==2.1.2`, `torchvision==0.16.2`, `timm==0.9.12`, OpenCV, Pillow, `scikit-image`, and `tqdm`.

Strongly inferred: The repository expects direct script execution from the repo root rather than package installation, because imports are relative to top-level files and shell scripts export `PYTHONPATH` to external author directories instead of invoking an installed module.

Unknown: No repository-local `environment.yml`, `pyproject.toml`, or reproducible lockfile was found.

## 9. Optional Pipelines

- `Training pipeline` — top-level paired translation training with adversarial and registration-aware losses. Entry point: `train.py`; presets: `train_af2he.sh`, `train_aperio.sh`, `train_cuhk.sh`, `train_hemit.sh`. Observed.
- `Prediction pipeline` — checkpointed patch inference/export for each dataset. Entry point: `predict.py`; presets: `predict_af2he.sh`, `predict_aperio.sh`, `predict_cuhk.sh`, `predict_hemit.sh`. Observed.
- `Registration inspection pipeline` — deformation export and qualitative visualization. Entry points: `get_deformation_field.py`, `visualize.ipynb`. Observed.
- `Downstream pathology classification pipeline` — feature extraction followed by either end-to-end classification or linear probing on extracted features. Entry points: `downstream_tasks/generate_feature.py`, `downstream_tasks/train.py`, `downstream_tasks/linear_prob.py`, plus helper shell scripts in `downstream_tasks/`. Observed.
- `WSI inference pipeline` — Unknown. No slide reader, tiler, overlap handler, or stitcher was found in the repository.

## 10. Other Notes

- Observed: `README.md` refers to the project as “DGR” while the repository directory is `DTR`.
- Observed: The README’s inference notebook link label references `play_with_the_pretrained_model.ipynb`, but the linked GitHub path points to `visualize.ipynb`.
- Observed: Only `weights/hemit_weight.pth` is present in the checked-in `weights/` directory, even though the notebook expects additional weights for other datasets.
- Observed: The working tree already contains user changes outside this overview file (`play_with_the_pretrained_model.ipynb`, `requirements.txt`, and `.codex`).

## 11. Uncertainties

- Unknown: Whether the authors perform full WSI H&E→mIHC inference outside this repository. Evidence is insufficient because no in-repo code reads `.svs`, `.tif`, or other slide formats, and no tiling/stitching pipeline is present.
- Unknown: Whether the `PYTHONPATH` exports in the shell scripts are still required in the current repo state or are leftovers from the author’s original environment. The imports resolve locally for static analysis, but the scripts point to external absolute paths.
- Unknown: Whether `swin_transformer.py` is intended for future work, abandoned experimentation, or external downstream use. No caller was observed.
- Unknown: Whether HEMIT “input” images are already cropped from WSIs or represent some other preprocessing stage. The loader only sees paired PNGs and does not expose provenance metadata.

## 12. Coverage Report

- docs read: `README.md`, `/home/bingo/.codex/skills/repo-study-overview/SKILL.md`, workflow and template assets for the skill
- config files read: `requirements.txt`, dataset-specific shell presets `train_*.sh` and `predict_*.sh`
- core code inspected: `train.py`, `predict.py`, `Attention_GAN.py`, `batch_utils.py`, `metrics.py`, `datasets/hemit.py`, `datasets/af2he.py`, `datasets/aperio.py`, `datasets/cuhk.py`, `datasets/he2pas.py`, `datasets/random_affine.py`, `reglib/reg.py`, `get_deformation_field.py`
- notebooks inspected: `play_with_the_pretrained_model.ipynb`, `visualize.ipynb`
- secondary code inspected: `downstream_tasks/dataset.py`, `downstream_tasks/generate_feature.py`, `downstream_tasks/model.py`, `downstream_tasks/train.py`, `downstream_tasks/linear_prob.py`
- skipped or lightly sampled: `assets/` image contents, `reglib/layers.py`, `reglib/nn.py`, `reglib/trainer.py`, notebook output blobs, helper scripts under `downstream_tasks/scripts/`

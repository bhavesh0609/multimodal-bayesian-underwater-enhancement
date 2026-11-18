# MM-BEM: Multimodal Bayesian Enhancement for Underwater Images

**Team 21 | Deep Learning for Computer Vision Project**

Prajyot Pyati (266150005), Bhavesh Chamaria (230121012), Devarsh Kumar (230121017), Harivathsha R (230121027) and Don Devasiachan (230121019).

[Project repository](https://github.com/bhavesh0609/multimodal-bayesian-underwater-enhancement) · [Original paper](https://arxiv.org/abs/2501.14265) · [Original BEM implementation](https://github.com/BinCVER/BEM) · [Project checkpoint folder](https://drive.google.com/drive/folders/1GI-eOBvNljgM6dXzRBhlSswPuRyXZ1Gz?usp=sharing)

## Contents

1. [Project overview](#1-project-overview)
2. [Environment and dependencies](#2-environment-and-dependencies)
3. [Dataset preparation](#3-dataset-preparation)
4. [Training](#4-training)
5. [Inference and evaluation](#5-inference-and-evaluation)
6. [Checkpoints and results](#6-checkpoints-and-results)
7. [Code organization and reproducibility](#7-code-organization-and-reproducibility)
8. [References](#8-references)

## 1. Project overview

Underwater photographs often suffer from color casts, low contrast, and uneven illumination. Several enhanced images can be plausible for the same degraded input, making enhancement a one-to-many mapping problem.

This project builds on **Bayesian Neural Networks for One-to-Many Mapping in Image Enhancement**, by Guoxi Huang, Qirui Yang, Ruirui Lin, Zipeng Qi, David Bull, and Nantheera Anantrasirichai (AAAI 2026). Their Bayesian Enhancement Model (BEM) uses a Bayesian neural network to generate coarse enhancement conditions and a deterministic network to reconstruct image details.

### Proposed contribution

The proposed **Multimodal Bayesian Enhancement Model (MM-BEM)** replaces the single diagonal Gaussian weight posterior with a learnable mixture of three diagonal Gaussians:

$$
q(w)=\sum_{k=1}^{3}\pi_k,\mathcal{N}\!\left(w;\mu_k,\operatorname{diag}(\sigma_k^2)\right),
\qquad \pi_k\geq 0,\quad \sum_k\pi_k=1.
$$

Each Bayesian convolution stores component means, scale parameters, and mixture logits. Softmax converts the logits into mixing probabilities; softplus converts the scale parameters into positive standard deviations. Sampling a component and its weights allows a layer to explore different regions of parameter space.

The project report motivates this choice using DenseNet-201 features and t-SNE visualizations of UIEB reference images. The mixture posterior increases the expressiveness of the weight distribution; its effect on enhancement quality is assessed through PSNR and SSIM.

Here, *multimodal* refers to multiple modes in the weight posterior. The enhancement input is an RGB image.

### Two-stage pipeline

1. **Stage I: Bayesian condition prediction.** FFT-based low-pass filtering and resizing prepare the input. A Bayesian encoder-decoder predicts a three-channel latent condition, which is upsampled to the input dimensions. A coarse reconstruction is formed as `y_tilde = (x + ALPHA * z) * z`.
2. **Stage II: deterministic refinement.** A second encoder-decoder receives the six-channel concatenation `[x, z]` and predicts the enhanced RGB image. During training, its condition is the mean of five Stage I samples.

Both supplied networks use a convolutional encoder-decoder with base width 48, three downsampling levels, residual blocks, GELU activations, and additive skip connections. Stage I converts `nn.Conv2d` layers to mixture-based Bayesian layers; the transposed convolutions remain deterministic. The adaptive Gaussian prior follows moments of the mixture posterior.

## 2. Environment and dependencies

### Recommended environment

|Component|Selection|
|-|-|
|Operating system|Linux, Windows, or Ubuntu under WSL 2|
|Python|3.10|
|Deep learning framework|PyTorch 2.5.1|
|NVIDIA CUDA wheel|CUDA 12.1 (`cu121`)|
|NumPy|1.26.4|
|Pillow|10.4.0|
|Matplotlib|3.9.2|
|Compute|CUDA-capable NVIDIA GPU recommended; automatic CPU fallback|
|Memory planning|Adjust `BATCH_SIZE` to GPU capacity; 150-sample evaluation also stores multiple image-sized tensors|

The standalone MM-BEM scripts use PyTorch, NumPy, Pillow, and Matplotlib. CUDA execution requires a compatible NVIDIA driver. The standalone scripts use prebuilt PyTorch operations; the original BEM baseline has its own environment and selective-scan build described in Section 4.

Commands below use Bash, as available in Linux and WSL. The Conda, pip, and direct Python commands can also be run from an appropriate Windows terminal. Run project commands from the directory containing `train.py` and `test.py`.

### Create the MM-BEM environment

From the repository root, run:

```bash
conda create -n mm-bem python=3.10 -y
conda activate mm-bem
python -m pip install --upgrade pip
```

For an NVIDIA GPU:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

For CPU execution, use this installation instead:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The accompanying `requirements.txt` contains:

```text
torch==2.5.1
numpy==1.26.4
Pillow==10.4.0
matplotlib==3.9.2
```

Check the environment:

```bash
python -c "import sys, torch, numpy, PIL, matplotlib; print('Python:', sys.version); print('PyTorch:', torch.__version__); print('CUDA wheel:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 3. Dataset preparation

Use the **Underwater Image Enhancement Benchmark (UIEB)**. Download the raw images and corresponding reference images from the [official UIEB dataset page](https://li-chongyi.github.io/proj_benchmark). UIEB contains 890 paired images and 60 additional challenging images. Paired training and evaluation use the images with references. Follow the dataset page's usage terms.

### Directory structure

The dataset root must contain these directories:

|Directory relative to `DATA_ROOT`|Contents|
|-|-|
|`train/lq/`|Degraded training images|
|`train/gt/`|Corresponding training reference images|
|`val/lq/`|Degraded evaluation images|
|`val/gt/`|Corresponding evaluation reference images|

If you already have the team's prepared dataset, point `paths.data_root` in `configs/mm_bem.json` at its root. Export the image lists from that dataset and commit them so others can reconstruct the split:

```bash
python scripts/prepare_uieb.py export --root data/UIEB --output splits
```

To reconstruct the same split from the downloaded UIEB archives, extract each archive and supply the two folders that contain the image files:

```bash
python scripts/prepare_uieb.py prepare --raw downloads/raw --reference downloads/reference --splits splits --output data/UIEB
```

The destination must be new. Each list contains one input filename per line. The preparation script matches each reference by filename stem and checks for overlapping partitions. The team’s historical lists have not yet been supplied.

### Pairing and preprocessing

* Images are loaded as RGB, converted to `float32`, and scaled to `[0, 1]`.
* Input/reference pairing uses filename stems.
* Training resizes a reference to the input dimensions when necessary. Images smaller than the crop size are enlarged before a paired random crop.
* Training uses `256 × 256` crops with synchronized random horizontal and vertical flips.
* Evaluation uses a batch size of one and resizes dimensions down to a multiple of `DOWNSAMPLE_FACTOR`, using bilinear interpolation.
* FFT low-pass filtering and latent downsampling happen inside Stage I.

Use equal input/reference dimensions in the evaluation set. With evaluation factor 16, keep both image dimensions at least 128 pixels so the three-level Stage I encoder can downsample its input.

Check the prepared image pairs:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image
import train

root = Path(train.DATA_ROOT)
extensions = set(train.IMAGE_EXTENSIONS)
partitions = {}
for split in ("train", "val"):
    inputs = {p.stem: p for p in (root / split / "lq").iterdir()
              if p.is_file() and p.suffix.lower() in extensions}
    targets = {p.stem: p for p in (root / split / "gt").iterdir()
               if p.is_file() and p.suffix.lower() in extensions}
    assert inputs, f"Empty input partition: {split}"
    assert inputs.keys() == targets.keys(), f"Pair mismatch: {split}"
    for stem in inputs:
        with Image.open(inputs[stem]) as x, Image.open(targets[stem]) as y:
            assert x.size == y.size, f"Dimension mismatch: {split}/{stem}"
    partitions[split] = set(inputs)
    print(f"{split}: {len(inputs)} pairs")
assert partitions["train"].isdisjoint(partitions["val"]), "Overlapping partitions"
assert len(partitions["train"]) >= train.BATCH_SIZE, "Training needs a full batch"
print("Dataset checks passed")
PY
```

## 4. Training

### Proposed model: MM-BEM

Training reads the paths, model parameters and training settings from `configs/mm_bem.json`.

|Setting|Value|
|-|-|
|Random seed|1234|
|Training patch|`256 × 256`|
|Batch size|16|
|Stage I iterations|18,000|
|Stage II iterations|18,000|
|Optimizer|Adam; default betas `(0.9, 0.999)` and zero weight decay|
|Initial / minimum learning rate|`2e-4` / `1e-6`|
|Learning rate schedule|Cosine decay, independently for each stage|
|Gradient clipping norm|1.0|
|Training downsampling factor|1|
|Coarse reconstruction coefficient `ALPHA`|0.025|
|Gaussian mixture components|3|
|Weight KL Monte Carlo samples|5|
|Initial standard deviation|0.05|
|Mixture initialization jitter|0.01|
|Adaptive prior decay|0.998, with the step-dependent adjustment in `update_adaptive_prior`|
|Stage II condition|Mean of five sampled Stage I conditions|
|Checkpoint interval|10,000 iterations and the end of each stage|
|Console logging interval|100 iterations|
|DataLoader workers|4|
|Precision|CUDA automatic mixed precision with gradient scaling; standard CPU execution|

Stage I minimizes reconstruction MSE plus the scaled Monte Carlo KL term:

```text
loss_stage1 = MSE((x + ALPHA * z) * z, y)
              + (0.01 / BATCH_SIZE) * KL
```

The implementation averages weight KL estimates over five samples and normalizes each parameter-tensor contribution by its element count. Bias KL uses the cached bias sample. Stage II minimizes `L1(prediction, y)` while obtaining its mean condition under `torch.no_grad()`.

For a fresh run, choose an empty `paths.checkpoint_dir` in the JSON configuration. Both stages resume from their corresponding checkpoints if present.

Create output directories and train both stages in sequence:

```bash
python -c "from pathlib import Path; Path('logs').mkdir(exist_ok=True); Path('experiments/mm_bem').mkdir(parents=True, exist_ok=True)"
python -u train.py > logs/train_mm_bem.log 2>&1
```

To display progress directly in the terminal, use `python -u train.py`.

Training initializes Stage I from ordinary convolutional weights and constructs its three Gaussian components around those weights. Stage II loads the trained Stage I model from `STAGE1_CHECKPOINT`. Existing checkpoints supply model state, optimizer state, and the saved iteration for continuation.

### Original BEM baseline

Use the [BEM reproduction instructions](baseline/README.md) for installation, dataset configuration, training and Monte Carlo evaluation. The authors’ two-stage baseline uses different backbones and training settings from this implementation.

## 5. Inference and evaluation

Return to the MM-BEM source directory and activate `mm-bem` before running the proposed model.

### Configure checkpoints and evaluation

Download the model files from the [project checkpoint folder](https://drive.google.com/drive/folders/1GI-eOBvNljgM6dXzRBhlSswPuRyXZ1Gz?usp=sharing) and arrange the checkpoints using the paths below, or set `paths.checkpoint_dir` in the JSON configuration:

|Constant in `test.py`|Default path / value|
|-|-|
|`VAL_INPUT_DIR`|`<DATA_ROOT>/val/lq`|
|`VAL_TARGET_DIR`|`<DATA_ROOT>/val/gt`|
|`STAGE1_CHECKPOINT`|`./experiments/mm_bem/stage1_latest_modified_1.pth`|
|`STAGE2_CHECKPOINT`|`./experiments/mm_bem/stage2_latest_modified_1.pth`|
|`OUTPUT_DIR`|`./experiments/mm_bem/inference_results`|
|`NUM_MIXTURE_COMPONENTS`|3|
|`DOWNSAMPLE_FACTOR`|16|
|`MC_SAMPLES`|150|
|`SAVE_VISUALS`|`True`|
|DataLoader workers|2|

Use both stage checkpoints from the same experiment. For evaluation of a new training run, set `eval.downsample_factor` to the value used in that run; the supplied training and evaluation configurations specify 1 and 16, respectively.

Check the checkpoint paths before evaluation:

```bash
python -c "from pathlib import Path; import test; paths=[test.STAGE1_CHECKPOINT, test.STAGE2_CHECKPOINT]; [print(p, 'OK' if Path(p).is_file() else 'MISSING') for p in paths]; assert all(Path(p).is_file() for p in paths), 'Place both checkpoints at the configured paths'"
```

Run the paired inference and evaluation workflow:

```bash
python -c "from pathlib import Path; Path('logs').mkdir(exist_ok=True)"
python -u test.py > logs/eval_mm_bem.log 2>&1
```

The packaged evaluator sets the PyTorch seed from `eval.seed` (default 1234). The historical run seed is unconfirmed.

### Evaluation procedure

For each paired evaluation image, `test.py`:

1. Draws 150 latent conditions from Stage I.
2. Runs Stage II separately for each condition and clamps the predictions to `[0, 1]`.
3. Computes RGB PSNR and SSIM against the reference image.
4. Selects the prediction with the highest PSNR and records the SSIM of that same prediction.
5. Averages the selected per-image scores across the evaluation set.

This is **reference-assisted best-of-150 evaluation**. The checkpoint-based script uses the reference image for candidate selection. Its final summary contains the selected predictions' average PSNR and SSIM. Read the `Best PSNR` entries and `FINAL EVALUATION RESULTS` summary when collecting scores.

PSNR uses RGB MSE with data range 1. SSIM uses the script's channel-wise `5 × 5` binomial kernel, zero padding of two pixels, constants `0.01²` and `0.03²`, and an average over channels and spatial positions. For metric comparisons, use the same image partition, resizing, sample count, selection rule, and metric implementation.

### Saved outputs

For an input with filename stem `example`, the script writes:

|Output|Description|
|-|-|
|`example_pred.png`|Enhanced candidate selected by highest PSNR|
|`example_uncertainty.png`|Channel-averaged Stage I latent variance, normalized per image to grayscale|
|`example_comparison.png`|Input, reference, selected prediction, and four randomly chosen remaining predictions|

All images are saved under `OUTPUT_DIR`. Console output includes per-image scores, overall mean PSNR and SSIM, elapsed time, and the absolute output directory.

### Inference without reference images

For an input without a reference, average the Stage I latent samples and run Stage II once. This produces one enhanced image per input.

```bash
python infer.py --input inputs --output experiments/mm_bem/predictions_mc
```

### Original BEM evaluation

Use the [baseline instructions](baseline/README.md) to evaluate the authors’ released weights.

## 6. Checkpoints and results

### Model artifacts

**Project download location:** [MM-BEM shared folder](https://drive.google.com/drive/folders/1GI-eOBvNljgM6dXzRBhlSswPuRyXZ1Gz?usp=sharing).

Both scripts read the shared `configs/mm_bem.json` file. The default training/evaluation artifact paths are:

|Artifact|Path|
|-|-|
|Stage I model|`experiments/mm_bem/stage1_latest_modified_1.pth`|
|Stage II model|`experiments/mm_bem/stage2_latest_modified_1.pth`|
|Training log generated by the documented command|`logs/train_mm_bem.log`|
|Evaluation log generated by the documented command|`logs/eval_mm_bem.log`|
|Evaluation images|`experiments/mm_bem/inference_results/`|
|Reference-free inference images|`experiments/mm_bem/predictions_mc/`|

Training checkpoints contain `step`, `model`, `optimizer`, and `seed`. Both inference examples load the network parameters from the `model` entry. Stage I and Stage II use separate checkpoint files.

### Results in the project report

|Method|PSNR (dB) ↑|SSIM ↑|
|-|-:|-:|
|Original BEM-MC, published UIEB-R90 result (paper, Table 2)|**23.920**|**0.913**|
|Proposed MM-BEM, project report|19.453|0.868|
|Difference: proposed minus baseline|−4.467|−0.045|

The proposed model scored lower on both reported metrics. The experiment explores the Gaussian-mixture posterior as an extension of BEM; the project results do not show an improvement over the published baseline. The two rows retain their source protocols: the original BEM-MC paper result and the MM-BEM result reported by the team. The report does not identify the exact split or confirm the settings used for its scores. The [comparison table](results/comparison.csv) records each source.

## 7. Code organization and reproducibility

### Source map

|File / component|Responsibility|
|-|-|
|`configs/mm_bem.json`|Dataset paths, training schedule, model settings, and checkpoint destinations|
|`PairedEnhancementDataset`|Paired RGB loading, matching, crops, resizing, and augmentation|
|`fft_low_pass`, `downsample_low_pass`, `upsample`|Frequency filtering and spatial scaling|
|`EncoderDecoder`, `ConvBlock`, `LayerNorm2d`|Convolutional backbone|
|`MultimodalBayesianMixin`|Mixture posterior, moment calculation, sampling, log densities, and adaptive prior|
|`BayesianConv2d`, `convert_to_multimodal_bnn`|Proposed Bayesian convolution construction and integration|
|`Stage1BNN`, `construct_coarse_prediction`|Stochastic condition prediction and coarse image reconstruction|
|`Stage2DNN`|Six-channel conditioned deterministic refinement|
|`train_stage1`, `train_stage2`|Stage-specific optimization and checkpoint loading/saving|
|`test.py`: model definitions|Inference counterparts of the trained architecture|
|`InferenceDataset`, `run_inference`|Paired inference, candidate selection, metrics, and outputs|
|`calculate_psnr`, `calculate_ssim`|Full-reference metric implementations|
|`requirements.txt`|Pinned MM-BEM runtime dependencies|

The proposed posterior modifications are concentrated in `MultimodalBayesianMixin`, `BayesianConv2d`, `convert_to_multimodal_bnn`, and `Stage1BNN`. Their inference counterparts are defined in `test.py`.

### Repository contents

| Path | Purpose |
|---|---|
| `configs/mm_bem.json` | Paths, model settings, training schedule and evaluation settings |
| `scripts/prepare_uieb.py` | Export the existing split or reconstruct it from filename lists |
| `baseline/` | Pinned BEM setup and UIEB configuration |
| `infer.py` | Enhance images without reference targets |
| `results/comparison.csv` | Paper and team-reported metrics |
| `docs/project_report.pdf` | Team 21 project report |

Commit the actual split lists and training logs when available. The historical GPU model, memory, exact partitions and evaluation seed are unconfirmed. For each new run, save the JSON configuration alongside the log and weights and record the Python, PyTorch, CUDA and device versions using the environment command above.

The packaged training script initializes the adaptive prior mean from the starting convolution weights. The uploaded script allocated that buffer without initializing it. Both scripts use the same Stage I checkpoint name for saving and resuming. Evaluation requires a matching pair of complete checkpoints.

## 8. References

1. **Huang et al. (2026).** *Bayesian Neural Networks for One-to-Many Mapping in Image Enhancement.* Proceedings of the AAAI Conference on Artificial Intelligence. [Paper](https://arxiv.org/abs/2501.14265) · [Implementation](https://github.com/BinCVER/BEM).
2. **Li et al.** *An Underwater Image Enhancement Benchmark Dataset and Beyond.* [Official UIEB project and downloads](https://li-chongyi.github.io/proj_benchmark).
3. **Team 21 project report.** *Multimodal modelling of bayesian neural network for one to many mapping in image enhancement.* Source of the proposed contribution and project results.
4. **PyTorch installation reference.** [Official previous-version installation commands](https://pytorch.org/get-started/previous-versions/).

```bibtex
@inproceedings{huang2026bayesian,
  title={Bayesian Neural Networks for One-to-Many Mapping in Image Enhancement},
  author={Huang, Guoxi and Yang, Qirui and Lin, Ruirui and Qi, Zipeng and Bull, David and Anantrasirichai, Nantheera},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```


# BEM on UIEB

The baseline row in the project report comes from Table 2 of Huang et al. The following commands use the authors’ BEM implementation. Run them in Bash from the MM-BEM project root with a CUDA 11.8 build environment.

```bash
conda create -n bem-baseline python=3.10.12 -y
conda activate bem-baseline
git clone https://github.com/BinCVER/BEM.git external/BEM
git -C external/BEM checkout ffd10299a2b9677f30f26f8ad9af33b3b4e099bd
cd external/BEM
python -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install natsort PyYAML
python setup.py develop --no_cuda_ext
python -m pip install ./kernels/selective_scan
cd ../..
```

## Train

Prepare UIEB using the project README. The configuration script sets the paths in copies of the upstream YAML files. Stage II runs for 150,000 iterations to match its shipped learning-rate schedule and the paper. Other hyperparameters follow the pinned implementation, which includes changes since the paper experiments.

```bash
python baseline/configure.py external/BEM data/UIEB
mkdir -p logs
cd external/BEM
python -u basicsr/train.py --opt Options/CG_UNet_UIEB_local.yml > ../../logs/train_bem_stage1.log 2>&1
python -u basicsr/train.py --opt Options/IE_UNet_UIEB_local.yml > ../../logs/train_bem_stage2.log 2>&1
cd ../..
```

## Evaluate the released weights

Download the authors’ UIEB checkpoint/configuration pairs from their [checkpoint section](https://github.com/BinCVER/BEM#checkpoints). Place each pair under the matching directory in the command below. Use the published R90 images to compare with Table 2. Check that inputs and references have matching filenames in corresponding sort order.

```bash
mkdir -p logs
cd external/BEM
python -u Enhancement/eval.py \
  --opt experiments/CG_UNet_UIEB/CG_UNet_UIEB.yml \
  --weights experiments/CG_UNet_UIEB/ckpt.pth \
  --cond_opt experiments/IE_UNet_UIEB/IE_UNet_UIEB.yml \
  --cond_weights experiments/IE_UNet_UIEB/ckpt.pth \
  --input_dir ../../data/UIEB/val/lq --target_dir ../../data/UIEB/val/gt \
  --dataset UIEB --Monte_Carlo --num_samples 25 \
  --result_dir ../../results/bem > ../../logs/eval_bem.log 2>&1
cd ../..
```

Read the lines labelled `MC_PSNR` and `MC_SSIM` in the evaluation log. Use the paired local YAML files and their corresponding model paths to evaluate a new baseline training run.

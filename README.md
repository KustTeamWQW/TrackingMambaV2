# TrackingmambaV2

TrackingmambaV2 is a UAV visual object tracking project for mountain-jungle scenes.  The implementation follows the main ideas of the TrackingMambaV2 paper:

- a Vision Mamba backbone with four-direction spatial scanning;
- an EinFFT frequency-domain channel mixer;
- a fixed-size accumulated historical feature tensor for temporal enhancement;
- UAV mountain-jungle oriented data augmentation.

The codebase is organized around the `trackingmambav2` model name.

## Highlights

TrackingmambaV2 targets challenging UAV tracking cases such as low light, camouflage, heavy vegetation, camera shake, occlusion, disappearance, and large target scale changes.

The current implementation includes:

| Component | Implementation |
| --- | --- |
| Backbone | Vision Mamba small backbone, loaded through `timm.create_model` in `lib/models/trackingmambav2/trackingmambav2.py` |
| Four-direction scan | `bimamba_type="v3"` branch in `lib/models/trackingmambav2/mamba_simple.py` |
| EinFFT | `EinFFT` in `lib/models/trackingmambav2/models_mamba.py` |
| Temporal enhancement | fixed historical tensor decoder in `lib/models/layers/transformer_dec.py` |
| Mountain-jungle augmentation | `MountainJungleAugment` in `lib/train/data/transforms.py` |
| OTMJ epoch sweep | `tracking/eval_otmj_epochs.py` |

The default full training config is:

```text
experiments/trackingmambav2/TrackingmambaV2-ep150-full-256.yaml
```

This config trains on `LASOT`, `GOT10K_vottrain`, `COCO17`, and `TRACKINGNET`.  OTMJ is reserved for testing and is not included in the training set.

## Environment

The current local experiments were run with the `OSMTrack` conda environment, PyTorch 2.1, CUDA 11.8, and two NVIDIA GPUs.  A fresh environment can be prepared as follows:

```bash
conda create -n trackingmambav2 python=3.10 -y
conda activate trackingmambav2

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
  --index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc -y
conda install packaging -y
pip install timm==0.9.16 easydict pyyaml opencv-python pandas tqdm matplotlib \
  tensorboardX lmdb jpeg4py thop pycocotools visdom wandb
```

Install Mamba and causal convolution packages compatible with CUDA 11.8 and PyTorch 2.1:

```bash
pip install mamba-ssm causal-conv1d
```

If binary wheels are not available for your system, install the matching wheels/source packages from:

- https://github.com/state-spaces/mamba
- https://github.com/Dao-AILab/causal-conv1d

## Data Preparation

Set the training and testing dataset paths in:

```text
lib/train/admin/local.py
lib/test/evaluation/local.py
```

You can generate default local path files with:

```bash
python tracking/create_default_local_file.py \
  --workspace_dir . \
  --data_dir ./data \
  --save_dir ./output
```

Expected training datasets:

```text
${PROJECT_ROOT}/data
|-- lasot
|-- got10k
|-- coco
|-- trackingnet
```

Expected OTMJ test dataset:

```text
${PROJECT_ROOT}/data/OTMJ
|-- 01
|-- 02
|-- 03
...
```

For this repository's local setup, OTMJ is configured through:

```python
settings.otmj_path = '/home/zly/projects/datasets/OTMJ'
```

OTMJ should only be used for evaluation. 

## Pretrained Backbone

The model tries to load Vim pretrained weights from the following locations:

```text
/home/zly/projects/pythonprojects/pretrained_models/hustvlVim-small-midclstok+/vim_s_midclstok_ft_81p6acc.pth
/home/zly/projects/pythonprojects/pretrained_models/hustvlVim-small-midclstok+/vim_s_midclstok_80p5acc.pth
${PROJECT_ROOT}/pretrained_models/hustvlVim-small-midclstok+/vim_s_midclstok_ft_81p6acc.pth
${PROJECT_ROOT}/pretrained_models/hustvlVim-small-midclstok+/vim_s_midclstok_80p5acc.pth
```

You can override the path explicitly:

```bash
export TRACKINGMAMBAV2_BACKBONE_PRETRAIN=/path/to/vim_s_midclstok_ft_81p6acc.pth
```

The load log should contain a line similar to:

```text
Load pretrained model from: /path/to/vim_s_midclstok_ft_81p6acc.pth
Backbone pretrained load: missing=..., unexpected=...
```

## Training

Use the default two-GPU training script:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHON_BIN=/home/zly/miniconda3/envs/OSMTrack/bin/python \
CONFIG=TrackingmambaV2-ep150-full-256 \
SAVE_DIR=./output \
NPROC_PER_NODE=2 \
bash train.sh
```

Equivalent direct command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --nproc_per_node 2 \
  lib/train/run_training.py \
  --script trackingmambav2 \
  --config TrackingmambaV2-ep150-full-256 \
  --save_dir ./output \
  --use_lmdb 0
```

Checkpoints are saved to:

```text
output/checkpoints/train/trackingmambav2/TrackingmambaV2-ep150-full-256/
```

Checkpoint names follow:

```text
TrackingmambaV2_ep0262.pth.tar
```

## Testing

Evaluate a single checkpoint on OTMJ with one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python tracking/test_epoch.py \
  --tracker_name trackingmambav2 \
  --tracker_param TrackingmambaV2-ep150-full-256 \
  --dataset_name otmj \
  --save_name ep0262 \
  --threads 1 \
  --num_gpus 1 \
  --epoch 262
```

The `--epoch` argument is passed through `TRACKINGMAMBAV2_TEST_EPOCH` and selects:

```text
output/checkpoints/train/trackingmambav2/TrackingmambaV2-ep150-full-256/TrackingmambaV2_ep0262.pth.tar
```

Tracking results are saved under:

```text
output/test/tracking_results/trackingmambav2/TrackingmambaV2-ep150-full-256/otmj/ep0262/
```

## Project Structure

```text
TrackingmambaV2
|-- experiments/trackingmambav2/       # YAML configs
|-- lib/models/trackingmambav2/        # backbone, Mamba blocks, EinFFT, tracker model
|-- lib/models/layers/                 # head and temporal decoder layers
|-- lib/train/                         # training actors, loaders, transforms
|-- lib/test/                          # tracker, datasets, evaluation tools
|-- tracking/                          # train/test/sweep entry scripts
|-- output/                            # checkpoints, logs, test results
|-- tensorboard/                       # tensorboard event files
```

## Citation

If you use this project, cite the TrackingMamba work and the related OTMJ dataset as appropriate.

```bibtex
@ARTICLE{TrackingMambaV2,
  author={Wang, Qingwang and Zhou, Liyao and Fan, Huaiyu and Cheng, Bei and Zhang, Zhen and Gu, Yanfeng and Shen, Tao},
  journal={IEEE GEOSCIENCE AND REMOTE SENSING MAGAZINE},
  title={TrackingMambaV2: UAV Visual Object Tracking in  Mountain Jungle},
  year={2026},
  pages={ },
  keywords={UAV remote sensing; mountain jungle scenes; Mamba; object tracking},
  doi={}
}
```

## Acknowledgments

This repository follows the structure and evaluation style of modern single-stream trackers and refers to:

- TrackingMamba: https://github.com/KustTeamWQW/TrackingMamba
- OTMJ Dataset: https://github.com/KustTeamWQW/OTMJ_Dataset
- OSTrack: https://github.com/botaoye/OSTrack
- AQATrack: https://github.com/orgs/GXNU-ZhongLab
- Mamba: https://github.com/state-spaces/mamba
- Vim: https://github.com/hustvl/Vim

# SCDFNet

**SCDFNet: Saliency-Calibrated Discrepancy Fusion Network for Real-Time Multimodal Semantic Segmentation in Driving Scenes**

A lightweight DDRNet-based fusion network for real-time multimodal (RGB-Depth / RGB-Thermal /
RGB-Polarization) semantic segmentation.

---

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies: `torch`, `torchvision`, `opencv-python`, `timm`, `einops`, `scikit-learn`, `tqdm`.

Download the DDRNet ImageNet backbones from the official DDRNet release into `Backbone/`:
`ddrnet_23_slim.pth`, `ddrnet_23.pth`, `ddrnet_39.pth`.

---

## Datasets

Organise the four supported datasets as below; `${DATASET_ROOT}` is the directory holding them.

```
${DATASET_ROOT}/MFNet/ir_seg_dataset/    images/ (4-channel RGB+T PNG), labels/, train.txt, test.txt
${DATASET_ROOT}/FMB/{train,test}/        Visible/, Infrared/, Label/
${DATASET_ROOT}/ZJU/{train,val}/         0/, 45/, 90/, 135/, train_label/ or val_label/
${DATASET_ROOT}/cityscapes/{train,valid}/  rgb/, depth_raw/ (.npy), labels_19/
```

---

## Training

```bash
python train.py \
    --dataset MFNet \
    --network SCDFNet-2 \
    --num_classes 9 \
    --batch_size 8 \
    --batch_size_valid 4 \
    --epochs 400 \
    --lr 0.01 \
    --weight_decay 0.0005 \
    --crop_H 480 --crop_W 640 \
    --img_train_dir ${DATASET_ROOT}/MFNet/ir_seg_dataset/ \
    --img_test_dir  ${DATASET_ROOT}/MFNet/ir_seg_dataset/ \
    --gpu_ids 0 \
    --pretrained True \
    --backbone_path ./Backbone/ddrnet_39.pth
```

For the other datasets, keep the same optimizer settings and replace the dataset-specific arguments:

| `--dataset` | `--num_classes` | `--crop_H` / `--crop_W` | `--img_train_dir` | `--img_test_dir` |
| --- | --- | --- | --- | --- |
| `MFNet` | 9 | 480 / 640 | `${DATASET_ROOT}/MFNet/ir_seg_dataset/` | `${DATASET_ROOT}/MFNet/ir_seg_dataset/` |
| `FMB` | 14 | 600 / 800 | `${DATASET_ROOT}/FMB/train/` | `${DATASET_ROOT}/FMB/test/` |
| `ZJU` | 8 | 512 / 612 | `${DATASET_ROOT}/ZJU/train/` | `${DATASET_ROOT}/ZJU/val/` |
| `Cityscapes` | 19 | 512 / 1024 | `${DATASET_ROOT}/cityscapes/train/` | `${DATASET_ROOT}/cityscapes/valid/` |

Checkpoints go to `Checkpoints/{dataset}/` (override with `--checkpoint_dir`), logs to `Logs/{dataset}/`.

---

## Evaluation

```bash
python eval.py \
    --dataset MFNet \
    --network SCDFNet-2 \
    --num_classes 9 \
    --batch_size_valid 8 \
    --img_test_dir ${DATASET_ROOT}/MFNet/ir_seg_dataset/ \
    --gpu_ids 0 \
    --weight_path ./Checkpoints/MFNet/SCDFNet-2_MFNet.pth \
    --save_results --results_dir ./Logs/Results/MFNet
```

---

## Model Variants

`--network` selects the backbone scale; pass the matching `--backbone_path` when `--pretrained True`.

| `--network` | Backbone | `--backbone_path` |
| --- | --- | --- |
| `SCDFNet-1-slim` | DDRNet-23-slim | `./Backbone/ddrnet_23_slim.pth` |
| `SCDFNet-1` | DDRNet-23 | `./Backbone/ddrnet_23.pth` |
| `SCDFNet-2` (default) | DDRNet-39 | `./Backbone/ddrnet_39.pth` |

---

## License

For academic research only. The pretrained backbone weights originate from the official DDRNet
release; please follow the license terms of the relevant datasets and DDRNet.

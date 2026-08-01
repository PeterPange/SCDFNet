import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import sys
import logging
from datetime import datetime
from PIL import Image

from Dataloader.Data import Data
from scdfnet import build_scdfnet
from Metric.mIoU import StreamSegMetrics
from args import ArgumentParser


def get_class_names(dataset, num_classes):
    """Return the class names for a dataset."""
    if dataset == "Cityscapes":
        class_names = [
            'Road', 'Sidewalk', 'Building', 'Wall', 'Fence', 'Pole',
            'Traffic Light', 'Traffic Sign', 'Vegetation', 'Terrain', 'Sky',
            'Person', 'Rider', 'Car', 'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle'
        ]
    elif dataset == "MFNet":
        class_names = [
            'unlabeled', 'car', 'person', 'bike', 'curve',
            'car_stop', 'guardrail', 'color_cone', 'bump'
        ]
    elif dataset == "FMB":
        class_names = [
            'Road', 'Sidewalk', 'Building', 'Lamp', 'Sign', 'Vegetation', 'Sky',
            'Person', 'Car', 'Truck', 'Bus', 'Motorcycle', 'Bicycle', 'Pole'
        ]
    elif dataset == "ZJU":
        class_names = [
            'Building', 'Glass', 'Car', 'Road', 'Tree', 'Sky', 'Pedestrian', 'Bicycle'
        ]
    else:
        class_names = [f'Class_{i}' for i in range(num_classes)]

    while len(class_names) < num_classes:
        class_names.append(f'Class_{len(class_names)}')

    return class_names[:num_classes]


def get_color_palette(num_classes, dataset):
    """Return the color palette for a dataset as [(R, G, B), ...]."""
    if dataset == "Cityscapes":
        palette = [
            [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
            [0, 0, 230], [119, 11, 32],
        ]
    elif dataset == "MFNet":
        palette = [
            [0, 0, 0], [0, 0, 0], [170, 110, 40], [89, 51, 21], [240, 180, 80],
            [160, 90, 110], [120, 120, 120], [140, 140, 140], [200, 150, 140],
            [100, 120, 150],
        ]
    elif dataset == "FMB":
        palette = [
            [228, 228, 179], [133, 57, 181], [177, 162, 67], [50, 178, 200],
            [199, 45, 132], [84, 172, 66], [79, 73, 179], [166, 99, 76],
            [253, 121, 66], [91, 165, 137], [152, 97, 155], [140, 153, 105],
            [158, 215, 222], [90, 113, 135],
        ]
    elif dataset == "ZJU":
        palette = [
            (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
            (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 64, 64),
        ]
    else:
        np.random.seed(42)
        palette = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_classes + 1)]

    while len(palette) < num_classes + 1:
        palette.append((0, 0, 0))

    return palette


def mask_to_color_image(mask, palette):
    """Convert a label mask to a color image."""
    h, w = mask.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, color in enumerate(palette):
        if label_id >= len(palette):
            break
        color_image[mask == label_id] = color
    return color_image


def parse_args():
    parser = ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    parser.add_argument('--save_results', action='store_true', help='Save segmentation results')
    parser.add_argument('--results_dir', type=str, default='./Results', help='Directory to save segmentation results')
    args = parser.parse_args()
    return args


def eval_main():
    args = parse_args()

    log_dir = os.path.join("Logs", args.dataset)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"eval_{args.network}_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='w')
        ]
    )

    logging.info(f"Evaluation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Network: {args.network}")
    logging.info(f"Weight path: {args.weight_path}")
    logging.info(f"Log file: {log_file}\n")

    val_data = Data(args.dataset, "test", args.img_test_dir, None)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_data, batch_size=args.batch_size_valid, shuffle=False, num_workers=0)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    if args.gpu_ids is not None and len(args.gpu_ids) > 0:
        gpu_id = int(args.gpu_ids.split(',')[0])
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using GPU: cuda:{gpu_id}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using default device: {device}")

    # Build model (no backbone pretrained weights needed for evaluation)
    network = build_scdfnet(
        version=args.network, pretrain=False, backbone_path=None,
        dataset=args.dataset, num_classes=args.num_classes)
    logging.info(f"Using SCDFNet variant: {args.network}")

    network = network.to(device)
    logging.info(f"Model loaded on {device}")

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    import warnings
    warnings.filterwarnings("ignore")

    if args.weight_path is not None:
        # Load weights and strip the 'module.' prefix produced by DataParallel training
        state_dict = torch.load(args.weight_path)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value
        network.load_state_dict(new_state_dict)
        logging.info(f"Weights loaded from: {args.weight_path}")

    ############################################################################
    # Evaluation
    ############################################################################
    metrics = StreamSegMetrics(args.num_classes)
    epoch_losses_val = []

    network.eval()
    batch_losses = []

    if args.save_results:
        os.makedirs(args.results_dir, exist_ok=True)
        logging.info(f"Saving segmentation results to: {args.results_dir}")
        palette = get_color_palette(args.num_classes, args.dataset)
        logging.info(f"Using color palette for {args.dataset} with {args.num_classes} classes")

        if args.dataset == "MFNet":
            gt_base_dir = os.path.join(args.img_test_dir, "labels")
        elif args.dataset == "FMB":
            gt_base_dir = os.path.join(args.img_test_dir, "Label")
        elif args.dataset == "Cityscapes":
            gt_base_dir = os.path.join(args.img_test_dir, "labels_19")
        elif args.dataset == "ZJU":
            gt_base_dir = os.path.join(args.img_test_dir, "val_label")
        else:
            gt_base_dir = None

    for idx, (RGB, X, label) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            label = label - 1
            RGB = RGB.to(device)
            X = X.to(device)
            label = (label.type(torch.LongTensor)).to(device)

            outputs = network(RGB, X)

            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = label.cpu().numpy()
            metrics.update(targets, preds)

            if args.save_results:
                for i in range(preds.shape[0]):
                    global_idx = idx * args.batch_size_valid + i
                    pred_color = mask_to_color_image(preds[i], palette)

                    gt_color = None
                    if gt_base_dir and hasattr(val_data, 'examples'):
                        try:
                            gt_path = val_data.examples[global_idx]["label_path"]
                            if os.path.exists(gt_path):
                                gt_raw = np.array(Image.open(gt_path).convert('L'))
                                if args.dataset == "MFNet":
                                    gt_shifted = np.zeros_like(gt_raw)
                                    mask = gt_raw > 0
                                    gt_shifted[mask] = gt_raw[mask] + 1
                                    gt_color = mask_to_color_image(gt_shifted, palette)
                                elif args.dataset == "Cityscapes":
                                    gt_shifted = gt_raw.astype(np.int32) - 1
                                    gt_color = mask_to_color_image(gt_shifted, palette)
                                else:
                                    gt_color = mask_to_color_image(gt_raw, palette)
                        except Exception as e:
                            logging.warning(f"Failed to process GT for index {global_idx}: {e}")

                    if args.dataset == "FMB":
                        combined_img = pred_color
                    elif gt_color is not None and gt_color.shape == pred_color.shape:
                        combined_img = np.hstack([gt_color, pred_color])
                    else:
                        combined_img = pred_color

                    result_path = os.path.join(args.results_dir, f"pred_{global_idx:06d}.png")
                    img = Image.fromarray(combined_img)
                    img.save(result_path)

            loss = criterion(outputs, label)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

    z = metrics.get_results()
    metrics.reset()
    epoch_loss = np.mean(batch_losses)
    epoch_losses_val.append(epoch_loss)

    class_names = get_class_names(args.dataset, args.num_classes)

    logging.info("=" * 80)
    logging.info("Evaluation Results")
    logging.info("=" * 80)
    logging.info("test/val loss: %g" % epoch_loss)
    logging.info("Overall Accuracy: %.2f%%" % (100 * z["Overall Acc"]))
    logging.info("Mean Accuracy: %.2f%%" % (100 * z["Mean Acc"]))
    logging.info("FreqW Accuracy: %.2f%%" % (100 * z["FreqW Acc"]))
    logging.info("Mean IoU: %.2f%%" % (100 * z["Mean IoU"]))
    logging.info("=" * 80)

    logging.info("\nPer-Class IoU:")
    logging.info("-" * 80)
    logging.info(f"{'Class ID':<10} {'Class Name':<20} {'IoU':>10}")
    logging.info("-" * 80)

    class_iou_dict = z["Class IoU"]
    for class_id in range(args.num_classes):
        iou_value = class_iou_dict.get(class_id, 0.0)
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        logging.info(f"{class_id:<10} {class_name:<20} {iou_value * 100:>9.2f}%")

    logging.info("-" * 80)
    logging.info(f"{'Mean IoU':<30} {z['Mean IoU'] * 100:>9.2f}%")
    logging.info("=" * 80)

    logging.info(f"\nEvaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Best mIoU: {z['Mean IoU'] * 100:.2f}%")
    logging.info(f"Evaluation logs saved to: {log_file}")


if __name__ == '__main__':
    eval_main()

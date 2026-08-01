import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import sys
import logging
from datetime import datetime

from Dataloader.Data import Data
from scdfnet import build_scdfnet
from Metric.mIoU import StreamSegMetrics
from args import ArgumentParser


def parse_args():
    parser = ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    args = parser.parse_args()
    return args


def build_model(args):
    """Build the SCDFNet model."""
    network = build_scdfnet(
        version=args.network, pretrain=args.pretrained,
        backbone_path=args.backbone_path, dataset=args.dataset,
        num_classes=args.num_classes)
    logging.info(f"Using SCDFNet variant: {args.network}")
    if args.pretrained and args.backbone_path:
        logging.info(f"Loading pretrained weights from: {args.backbone_path}")
    return network


def train_main():
    args = parse_args()

    # Create log directory
    log_dir = os.path.join("Logs", args.dataset)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{args.network}_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='w')
        ]
    )

    logging.info(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Network: {args.network}")
    logging.info(f"Log file: {log_file}\n")

    train_data = Data(args.dataset, "train", args.img_train_dir, (args.crop_H, args.crop_W))
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_data = Data(args.dataset, "valid", args.img_test_dir, None)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_data, batch_size=args.batch_size_valid, shuffle=False, num_workers=args.num_workers)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Device selection
    if args.gpu_ids is not None and len(args.gpu_ids) > 0:
        gpu_list = [int(g) for g in args.gpu_ids.split(',')]
        device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using specified GPU(s): {args.gpu_ids}")
        logging.info(f"Primary device: {device}")
    else:
        gpu_list = None
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using default device: {device}")

    # Build model
    network = build_model(args)

    # Move model to device and wrap with DataParallel
    num_gpus = len(gpu_list) if args.gpu_ids else args.num_gpus

    if num_gpus > 1 and torch.cuda.device_count() >= num_gpus:
        device_ids = [int(g) for g in args.gpu_ids.split(',')] if args.gpu_ids else list(range(num_gpus))
        logging.info(f"Using {num_gpus} GPUs: {device_ids}")
        network = nn.DataParallel(network, device_ids=device_ids)
    elif torch.cuda.device_count() >= 1:
        if args.gpu_ids:
            gpu_id = int(args.gpu_ids.split(',')[0])
            logging.info(f"Using single GPU: {gpu_id}")
            network = nn.DataParallel(network, device_ids=[gpu_id])
        else:
            logging.info("Using single GPU: 0")
            network = nn.DataParallel(network, device_ids=[0])
    else:
        logging.warning("No GPU available, using CPU")

    network = network.to(device)

    optimizer = torch.optim.SGD(network.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay)

    # Polynomial decay learning rate schedule: (1 - epoch / epochs) ^ 0.9
    def lr_lambda(epoch):
        return (1 - epoch / args.epochs) ** 0.9

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

    # Loss configuration
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    logging.info("Using CrossEntropy Loss only")

    # Gradient clipping to prevent exploding gradients
    max_grad_norm = 5.0

    import warnings
    warnings.filterwarnings("ignore")

    # Resume training (optional)
    if args.weight_path is not None:
        network.load_state_dict(torch.load(args.weight_path))
        logging.info(f"Resumed weights from: {args.weight_path}")

    metrics = StreamSegMetrics(args.num_classes)
    epoch_losses_train = []
    epoch_losses_val = []
    num_epoch = args.epochs
    z = 0
    r = 0
    mean_iou = 0

    for epoch in range(1, num_epoch + 1):
        logging.info("epoch: %d/%d" % (epoch, num_epoch))
        ############################################################################
        # train
        ############################################################################
        network.train()
        batch_losses = []
        for RGB, X, label in tqdm(train_loader, colour="blue", leave=False, file=sys.stdout):
            label = label - 1
            RGB = RGB.to(device)
            X = X.to(device)
            label = (label.type(torch.LongTensor)).to(device)
            outputs = network(RGB, X)

            loss = criterion(outputs, label)

            loss_value = loss.data.detach().cpu().numpy()
            batch_losses.append(loss_value)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
            optimizer.step()

        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)

        current_lr = optimizer.param_groups[0]['lr']
        logging.info("train loss: %g, lr: %.6f" % (epoch_loss, current_lr))

        scheduler.step()

        ############################################################################
        # validation
        ############################################################################
        if epoch >= args.eval_epochs_start:
            network.eval()
            batch_losses = []
            for RGB, X, label in tqdm(val_loader, colour="red", file=sys.stdout, leave=False):
                with torch.no_grad():
                    label = label - 1
                    RGB = RGB.to(device)
                    X = X.to(device)
                    label = (label.type(torch.LongTensor)).to(device)

                    outputs = network(RGB, X)

                    preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                    targets = label.cpu().numpy()
                    metrics.update(targets, preds)

                    loss = criterion(outputs, label)

                    loss_value = loss.data.cpu().numpy()
                    batch_losses.append(loss_value)

            z = metrics.get_results()
            metrics.reset()
            epoch_loss = np.mean(batch_losses)
            epoch_losses_val.append(epoch_loss)
            logging.info("test/val loss: %g" % epoch_loss)

            mean_iou = z["Mean IoU"]
            logging.info("mIoU: %.2f%%" % (100 * mean_iou))

            checkpoint_dir = args.checkpoint_dir or f"Checkpoints/{args.dataset}"
            os.makedirs(checkpoint_dir, exist_ok=True)

            if mean_iou > r:
                checkpoint_path = f"{checkpoint_dir}/best_{args.network}_{args.dataset}.pth"
                torch.save(network.state_dict(), checkpoint_path)
                logging.info("########################################")
                logging.info("              BEST RESULT               ")
                logging.info(f"Model saved to: {checkpoint_path}")
                logging.info("########################################")
                r = mean_iou

            # Always keep the final-epoch weights as well
            torch.save(network.state_dict(),
                       f"{checkpoint_dir}/last_{args.network}_{args.dataset}.pth")

    logging.info(f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Best mIoU: {r * 100:.2f}%")
    logging.info(f"Training logs saved to: {log_file}")


if __name__ == '__main__':
    train_main()

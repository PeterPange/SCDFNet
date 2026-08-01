import argparse


def str2bool(value):
    """Parse a boolean command-line flag.

    ``type=bool`` cannot be used directly with argparse: any non-empty string
    (including "False") would evaluate to True.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    if value.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got: {value}')


# Supported model variants (all based on the SCDFNet architecture).
# The default and recommended variant is SCDFNet-2 (DDRNet-39 backbone).
SUPPORTED_NETWORKS = [
    'SCDFNet-1-slim',
    'SCDFNet-1',
    'SCDFNet-2',
]


class ArgumentParser(argparse.ArgumentParser):
    def set_common_args(self):

        self.add_argument('--batch_size', type=int, default=8,
                          help='batch size for training')

        self.add_argument('--batch_size_valid', type=int, default=8,
                          help='batch size for validation / evaluation')

        self.add_argument('--epochs', default=300, type=int, metavar='N',
                          help='number of total epochs to run')

        self.add_argument('--eval_epochs_start', default=1, type=int, metavar='N',
                          help='epochs that evaluation start')

        # training hyper parameters
        self.add_argument('--lr', '--learning-rate', default=0.01, type=float)

        self.add_argument('--weight_decay', type=float, default=5e-4,
                          help='weight decay for optimizer')

        self.add_argument('--num_gpus', type=int, default=1, choices=[1, 2],
                          help='number of gpus (used when --gpu_ids is not specified)')

        self.add_argument('--gpu_ids', type=str, default=None,
                          help='GPU IDs to use (comma-separated), e.g., "0" or "0,1". Overrides --num_gpus')

        # model
        self.add_argument('--network', type=str,
                          default='SCDFNet-2',
                          choices=SUPPORTED_NETWORKS,
                          help='select model version')

        self.add_argument('--pretrained', type=str2bool,
                          default=False,
                          help='load the ImageNet-pretrained DDRNet backbone')

        self.add_argument('--backbone_path', type=str,
                          help='Path to pretrained backbone if pretrained == True')

        self.add_argument('--weight_path', type=str,
                          help='Path to trained model weight (for resuming training or evaluation)')

        self.add_argument('--checkpoint_dir', type=str, default=None,
                          help='Directory to save checkpoints (default: Checkpoints/{dataset})')

        # dataset
        self.add_argument('--dataset', default='MFNet',
                          choices=['Cityscapes', 'MFNet', 'ZJU', 'FMB'])

        self.add_argument('--img_train_dir', default=None,
                          help='Path to dataset train image root.')

        self.add_argument('--img_test_dir', default=None,
                          help='Path to dataset test/val image root.')

        self.add_argument('--num_classes', type=int, default=9,
                          help='number of classes')

        self.add_argument('--crop_H', type=int, default=480,
                          help='height of the random crop size in training')
        self.add_argument('--crop_W', type=int, default=640,
                          help='width of the random crop size in training')

        # Data loader
        self.add_argument('--num_workers', type=int, default=4,
                          help='number of workers for data loading')

import argparse
import math
import os, sys
import re
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix


def positive_finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError('must be a positive finite number')
    return parsed


def sha256_hex(value):
    if re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise argparse.ArgumentTypeError('must be exactly 64 lowercase hexadecimal characters')
    return value


def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument(
        '--require-cuda-device-name',
        default=None,
        help='Fail closed unless the selected CUDA device has this exact name',
    )
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/Data/val/', help='Directory of validation data')
    
    parser.add_argument('--cascade', type=int, default=1, help='Number of cascades | Should be less than 12') ## important hyperparameter
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    parser.add_argument(
        '--score-aligned-loss',
        action='store_true',
        help='Use the evaluator-aligned SSIM objective for training only',
    )
    parser.add_argument(
        '--retain-val-epochs',
        action='store_true',
        help='Retain reconstruction-only validation H5 outputs for every epoch',
    )
    parser.add_argument(
        '--resume-checkpoint',
        type=Path,
        default=None,
        help='Safe training checkpoint whose complete state should be resumed',
    )
    parser.add_argument(
        '--resume-checkpoint-sha256',
        type=sha256_hex,
        default=None,
        help='Expected SHA-256 verified from the exact checkpoint descriptor before loading',
    )
    parser.add_argument(
        '--resume-lr',
        type=positive_finite_float,
        default=None,
        help='Optional learning-rate override applied after optimizer resume',
    )
    parser.add_argument(
        '--allow-inexact-resume',
        action='store_true',
        help='Allow a sanitized legacy checkpoint without saved RNG state',
    )

    args = parser.parse_args()
    if args.resume_checkpoint_sha256 is not None and args.resume_checkpoint is None:
        parser.error('--resume-checkpoint-sha256 requires --resume-checkpoint')
    if args.resume_lr is not None and args.resume_checkpoint is None:
        parser.error('--resume-lr requires --resume-checkpoint')
    if args.allow_inexact_resume and args.resume_checkpoint is None:
        parser.error('--allow-inexact-resume requires --resume-checkpoint')
    return args

if __name__ == '__main__':
    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    result_root = Path("../result")
    args.exp_dir = result_root / args.net_name / "checkpoints"
    args.val_dir = result_root / args.net_name / "reconstructions_val"
    args.val_epochs_dir = result_root / args.net_name / "reconstructions_val_epochs"
    args.main_dir = result_root / args.net_name / Path(__file__).name
    args.val_loss_dir = result_root / args.net_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    train(args)

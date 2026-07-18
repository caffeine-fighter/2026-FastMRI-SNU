import argparse
import math
import os
import re
import sys
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
    parser = argparse.ArgumentParser(description='Train a reconstruction model on FastMRI challenge images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model-family',
        choices=('varnet', 'promptmr-plus'),
        default='varnet',
        help='Model family routed through this single training entrypoint',
    )
    parser.add_argument('--one-step-smoke', action='store_true', help='Run exactly one bounded optimizer-step smoke')
    parser.add_argument(
        '--no-register-experiment',
        action='store_true',
        help='Assert that this invocation must not register an experiment',
    )
    parser.add_argument('--precision', choices=('fp32',), default='fp32', help='Training precision')
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument(
        '--require-cuda-device-name',
        default=None,
        help='Fail closed unless the selected CUDA device has this exact name',
    )
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=positive_finite_float, default=None, help='Learning rate')
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

    if args.model_family == 'varnet':
        args.lr = 1e-3 if args.lr is None else args.lr
        if args.one_step_smoke:
            parser.error('--one-step-smoke is reserved for promptmr-plus')
    else:
        if args.batch_size != 1:
            parser.error('promptmr-plus requires batch size 1')
        if args.lr is not None and args.lr != 1e-4:
            parser.error('promptmr-plus requires the pinned learning rate 0.0001')
        if args.seed != 430:
            parser.error('promptmr-plus requires the pinned seed 430')
        if (
            args.input_key != 'kspace'
            or args.target_key != 'image_label'
            or args.max_key != 'max'
        ):
            parser.error('promptmr-plus requires pinned kspace/image_label/max data keys')
        if args.score_aligned_loss:
            parser.error('promptmr-plus requires the pinned upstream SSIM loss')
        if args.one_step_smoke and not args.no_register_experiment:
            parser.error('--one-step-smoke requires --no-register-experiment')
        if args.one_step_smoke and args.resume_checkpoint is not None:
            parser.error('--one-step-smoke cannot resume a checkpoint')
        if (
            args.one_step_smoke
            and args.require_cuda_device_name != 'NVIDIA GeForce RTX 3090'
        ):
            parser.error(
                '--one-step-smoke requires --require-cuda-device-name '
                "'NVIDIA GeForce RTX 3090'"
            )
        if args.one_step_smoke and re.fullmatch(
            r'FEATURE_PROMPTMR_PLUS_[A-Z0-9_]{1,96}', os.fspath(args.net_name)
        ) is None:
            parser.error(
                '--one-step-smoke requires a safe FEATURE_PROMPTMR_PLUS_* net name'
            )
        args.lr = 1e-4 if args.lr is None else args.lr
        args.promptmr_weight_decay = 0.01
        args.promptmr_lr_step_size = 35
        args.promptmr_lr_gamma = 0.1
        args.promptmr_gradient_clip_norm = 0.01
        args.promptmr_uniform_resolution = (384, 384)
        args.promptmr_use_checkpoint = True
        args.promptmr_compute_sens_per_coil = True
    return args


def configure_result_paths(args, result_root=Path("../result")):
    """Bind output paths while leaving guarded PromptMR+ creation to its runner."""
    args.run_dir = Path(result_root) / args.net_name
    args.exp_dir = args.run_dir / "checkpoints"
    args.val_dir = args.run_dir / "reconstructions_val"
    args.val_epochs_dir = args.run_dir / "reconstructions_val_epochs"
    args.main_dir = args.run_dir / Path(__file__).name
    args.val_loss_dir = args.run_dir
    return args


def prepare_runtime(args):
    """Verify PromptMR+ bytes before seed/backend/CUDA mutation."""
    if args.model_family == "promptmr-plus":
        from utils.learning.promptmr_plus_training import load_promptmr_training_recipe

        load_promptmr_training_recipe()
    if args.seed is not None:
        seed_fix(args.seed)


if __name__ == '__main__':
    args = parse()
    prepare_runtime(args)

    configure_result_paths(args)

    if args.model_family == "varnet":
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        args.val_dir.mkdir(parents=True, exist_ok=True)

    train(args)

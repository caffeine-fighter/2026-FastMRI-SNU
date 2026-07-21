import argparse
import math
import os, sys
import re
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train
from utils.learning.fi_acc8_training import (
    FI_ACC8_RECIPE,
    run_fi_acc8_training_fit_smoke,
)
from utils.learning.fi_acc8_full_training import (
    FI_ACC8_FULL_NAMESPACE,
    FI_ACC8_FULL_RECIPE,
)

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
        '--model-family',
        choices=('varnet', 'fi-varnet-acc8'),
        default='varnet',
        help='Model family selected through the shared training integration',
    )
    parser.add_argument(
        '--fi-acc8-one-step-smoke',
        action='store_true',
        help='Run only the review-gated exactly-one-step FI-VarNet training-fit smoke',
    )
    parser.add_argument(
        '--fi-acc8-full-training',
        action='store_true',
        help='Run only the separately reviewed checkpointed FI acc8 epochs-1-30 lane',
    )
    parser.add_argument(
        '--expected-gpu-uuid',
        default=None,
        help='Exact UUID of the otherwise idle 8192 MiB GTX 1080 selected for smoke',
    )
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
    parser.add_argument('--pools', type=int, default=4)
    parser.add_argument('--sens-pools', type=int, default=4)
    parser.add_argument('--acceleration', type=int, default=8)
    parser.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'), default='fp32')
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--ramp-steps', type=int, default=3704)
    parser.add_argument('--cosine-decay-start', type=int, default=46300)
    parser.add_argument('--max-steps', type=int, default=92600)
    parser.add_argument('--external-learned-state', type=Path, default=None)
    parser.add_argument('--no-scratch', action='store_true')
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
    if args.model_family == FI_ACC8_RECIPE.model_family:
        selected_modes = int(args.fi_acc8_one_step_smoke) + int(
            args.fi_acc8_full_training
        )
        if selected_modes != 1:
            parser.error(
                'FI-VarNet requires exactly one explicit lane: '
                '--fi-acc8-one-step-smoke or --fi-acc8-full-training'
            )
        if not args.expected_gpu_uuid:
            parser.error('--expected-gpu-uuid is required for FI-VarNet acc8')
        if args.data_path_train != Path('/root/Data/train'):
            parser.error(
                'FI-VarNet organizer root is frozen to /root/Data/train'
            )

        provided = {
            token.split('=', 1)[0]
            for token in sys.argv[1:]
            if token.startswith('-')
        }
        lane_epochs = (
            FI_ACC8_FULL_RECIPE.base_epochs
            if args.fi_acc8_full_training
            else FI_ACC8_RECIPE.epochs
        )
        lane_namespace = (
            Path(FI_ACC8_FULL_NAMESPACE)
            if args.fi_acc8_full_training
            else Path('LOCAL_FI_ACC8_CKPT_SMOKE_R1')
        )
        frozen = (
            (('-b', '--batch-size'), 'batch_size', FI_ACC8_RECIPE.batch_size),
            (('-e', '--num-epochs'), 'num_epochs', lane_epochs),
            (('-l', '--lr'), 'lr', FI_ACC8_RECIPE.lr),
            (('--seed',), 'seed', FI_ACC8_RECIPE.seed),
            (('--cascade',), 'cascade', FI_ACC8_RECIPE.num_cascades),
            (('--chans',), 'chans', FI_ACC8_RECIPE.chans),
            (('--sens_chans',), 'sens_chans', FI_ACC8_RECIPE.sens_chans),
            (('--pools',), 'pools', FI_ACC8_RECIPE.pools),
            (('--sens-pools',), 'sens_pools', FI_ACC8_RECIPE.sens_pools),
            (('--acceleration',), 'acceleration', FI_ACC8_RECIPE.acceleration),
            (('--precision',), 'precision', FI_ACC8_RECIPE.precision),
            (('--weight-decay',), 'weight_decay', FI_ACC8_RECIPE.weight_decay),
            (('--ramp-steps',), 'ramp_steps', FI_ACC8_RECIPE.ramp_steps),
            (('--cosine-decay-start',), 'cosine_decay_start', FI_ACC8_RECIPE.cosine_decay_start),
            (('--max-steps',), 'max_steps', FI_ACC8_RECIPE.max_steps),
            (('--input-key',), 'input_key', 'kspace'),
            (('--target-key',), 'target_key', 'image_label'),
            (('--max-key',), 'max_key', 'max'),
            (('-n', '--net-name'), 'net_name', lane_namespace),
        )
        for options, attribute, expected in frozen:
            if provided.intersection(options) and getattr(args, attribute) != expected:
                parser.error(
                    f'{options[-1]} is frozen to {expected!r} for FI-VarNet acc8'
                )
            setattr(args, attribute, expected)
        if args.fi_acc8_one_step_smoke and (
            args.resume_checkpoint is not None or args.allow_inexact_resume
        ):
            parser.error('FI-VarNet acc8 smoke is scratch-only; resume is forbidden')
        if args.fi_acc8_full_training:
            if (
                args.resume_checkpoint is not None
                and args.resume_checkpoint_sha256 is None
            ):
                parser.error(
                    'FI-VarNet full-training resume requires '
                    '--resume-checkpoint-sha256'
                )
            if args.allow_inexact_resume or args.resume_lr is not None:
                parser.error('FI-VarNet full training permits exact resume only')
        if args.external_learned_state is not None or args.no_scratch:
            parser.error('FI-VarNet acc8 forbids all external learned state')
        if args.score_aligned_loss:
            parser.error('FI-VarNet acc8 loss is frozen to upstream SSIMLoss')
    return args

def main(args=None, result_root=Path('../result')):
    if args is None:
        args = parse()

    if getattr(args, 'model_family', 'varnet') == FI_ACC8_RECIPE.model_family:
        if getattr(args, 'fi_acc8_full_training', False):
            from utils.learning.fi_acc8_full_training import run_fi_acc8_full_training

            net_name = Path(args.net_name)
            if net_name != Path(FI_ACC8_FULL_NAMESPACE):
                raise ValueError(
                    f'FI-VarNet full-training net name must be {FI_ACC8_FULL_NAMESPACE}'
                )
            if net_name.is_absolute() or len(net_name.parts) != 1:
                raise ValueError('FI-VarNet full-training net name must be one basename')
            output_dir = Path(result_root) / net_name.name / 'fi-acc8-full-training'
            return run_fi_acc8_full_training(args, output_dir)
        if not getattr(args, 'fi_acc8_one_step_smoke', False):
            raise RuntimeError('FI-VarNet requires an explicit execution lane')
        net_name = Path(args.net_name)
        expected_net_name = Path('LOCAL_FI_ACC8_CKPT_SMOKE_R1')
        if net_name != expected_net_name:
            raise ValueError(
                'FI-VarNet checkpoint smoke net name must be '
                'LOCAL_FI_ACC8_CKPT_SMOKE_R1'
            )
        if (
            net_name.is_absolute()
            or len(net_name.parts) != 1
            or net_name.name in {'', '.', '..'}
        ):
            raise ValueError('FI-VarNet smoke net name must be one relative basename')
        output_dir = (
            Path(result_root) / net_name.name / 'fi-acc8-training-fit-smoke'
        )
        return run_fi_acc8_training_fit_smoke(args, output_dir)

    # Preserve the legacy VarNet entrypoint behavior unchanged.
    if args.seed is not None:
        seed_fix(args.seed)

    result_root = Path(result_root)
    args.exp_dir = result_root / args.net_name / "checkpoints"
    args.val_dir = result_root / args.net_name / "reconstructions_val"
    args.val_epochs_dir = result_root / args.net_name / "reconstructions_val_epochs"
    args.main_dir = result_root / args.net_name / Path(__file__).name
    args.val_loss_dir = result_root / args.net_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)
    return train(args)


if __name__ == '__main__':
    main()

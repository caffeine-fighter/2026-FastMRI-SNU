"""Fail-closed FI-VarNet acc8 training-fit integration.

The full trainer is intentionally unavailable.  This module contains the frozen
recipe and the review-gated exactly-one-step smoke path.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import torch

from utils.learning.resume import (
    _open_anonymous_file,
    _publish_fd_without_overwrite,
)
from utils.learning.train_part import (
    _cleanup_staged_directory,
    _create_staged_directory,
    _publish_staged_directory_no_replace,
    _seal_staged_directory,
    _staged_directory_descriptor_path,
    build_model,
)
from utils.model.fi_varnet_adapter import (
    build_pinned_ssim_loss,
    enable_fi_activation_checkpointing,
    verify_pinned_upstream_sources,
)


@dataclass(frozen=True)
class FIAcc8Recipe:
    schema: str = "fi-varnet-acc8-training-recipe-v1"
    model_family: str = "fi-varnet-acc8"
    scratch: bool = True
    external_learned_state: bool = False
    seed: int = 431
    batch_size: int = 1
    precision: str = "fp32"
    autocast: bool = False
    optimizer: str = "AdamW"
    lr: float = 3e-4
    weight_decay: float = 0.0
    loss: str = "upstream-fastmri-SSIMLoss"
    gradient_clipping: bool = False
    num_cascades: int = 12
    chans: int = 18
    pools: int = 4
    sens_chans: int = 8
    sens_pools: int = 4
    acceleration: int = 8
    train_files: int = 85
    slices_per_epoch: int = 2315
    epochs: int = 40
    max_steps: int = 92600
    ramp_steps: int = 3704
    cosine_decay_start: int = 46300
    checkpoint_every_epoch: bool = True
    reconstruction_every_epoch: bool = True

    def as_dict(self):
        return asdict(self)


FI_ACC8_RECIPE = FIAcc8Recipe()
FI_ACC8_PRODUCTION_ROOT = Path('/root/Data/train')
FI_ACC8_ORGANIZER_TOTAL_FILES = 170
FI_ACC8_CORESIDENT_ACC4_FILES = 85
FI_ACTIVATION_CHECKPOINT_CONTRACT = {
    'enabled': True,
    'implementation': 'torch.utils.checkpoint.checkpoint',
    'use_reentrant': False,
    'preserve_rng_state': True,
    'feature_cascades': 12,
    'image_cascades': 12,
    'state_dict_unchanged': True,
}


def _validate_activation_checkpoint_contract(contract):
    if (
        type(contract) is not dict
        or set(contract) != set(FI_ACTIVATION_CHECKPOINT_CONTRACT)
        or any(
            type(contract[key]) is not type(expected) or contract[key] != expected
            for key, expected in FI_ACTIVATION_CHECKPOINT_CONTRACT.items()
        )
    ):
        raise ValueError('FI activation checkpoint evidence does not match exact contract')
    return dict(contract)


def fi_lr_multiplier(step):
    """Upstream FI-VarNet LambdaLR form with the frozen 40-epoch scalars."""
    if step < FI_ACC8_RECIPE.cosine_decay_start:
        return min(step / FI_ACC8_RECIPE.ramp_steps, 1.0)
    cosine_steps = FI_ACC8_RECIPE.max_steps - FI_ACC8_RECIPE.cosine_decay_start
    angle = (
        (step - FI_ACC8_RECIPE.cosine_decay_start)
        / cosine_steps
        * math.pi
        / 2
    )
    return max(math.cos(angle), 1e-8)


def build_fi_scheduler(optimizer):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fi_lr_multiplier)


def preflight_smoke_gpu(gpu_index, expected_uuid, runner=subprocess.run):
    """Verify physical GPU identity and idleness without touching CUDA."""
    identity_command = [
        'nvidia-smi',
        '--query-gpu=index,uuid,name,memory.total',
        '--format=csv,noheader,nounits',
    ]
    try:
        identity_result = runner(
            identity_command, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError('Could not verify smoke GPU with nvidia-smi') from exc
    rows = []
    for line in identity_result.stdout.splitlines():
        columns = [column.strip() for column in line.split(',')]
        if len(columns) != 4:
            raise RuntimeError('Malformed nvidia-smi GPU identity output')
        rows.append(columns)
    matches = [row for row in rows if row[0] == str(gpu_index)]
    if len(matches) != 1:
        raise RuntimeError(f'GPU index {gpu_index} is missing or ambiguous')
    index, uuid, name, memory = matches[0]
    try:
        memory_mib = int(memory)
    except ValueError as exc:
        raise RuntimeError('Malformed nvidia-smi GPU memory output') from exc
    accepted_names = {'GeForce GTX 1080', 'NVIDIA GeForce GTX 1080'}
    if uuid != expected_uuid or name not in accepted_names or memory_mib != 8192:
        raise RuntimeError(
            'Selected GPU does not match exact UUID/name/8192MiB smoke contract'
        )

    owners_command = [
        'nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits',
    ]
    try:
        owners_result = runner(
            owners_command, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError('Could not verify zero GPU compute owners') from exc
    owner_lines = [
        line.strip()
        for line in owners_result.stdout.splitlines()
        if line.strip() and 'No running processes found' not in line
    ]
    for line in owner_lines:
        columns = [column.strip() for column in line.split(',')]
        if len(columns) != 2:
            raise RuntimeError('Malformed nvidia-smi compute-owner output')
        if columns[0] == expected_uuid:
            raise RuntimeError(f'Selected GPU is occupied by compute PID {columns[1]}')
    return {
        'index': int(index),
        'uuid': uuid,
        'name': name,
        'memory_mib': memory_mib,
        'compute_owners': [],
    }


def prepare_smoke_output(output_dir):
    """Create a new smoke generation root; never reuse or overwrite one."""
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(f'Smoke output already exists: {output_dir}') from exc
    return output_dir.resolve()


def _center_crop_to_smallest(target, output):
    height = min(target.shape[-2], output.shape[-2])
    width = min(target.shape[-1], output.shape[-1])

    def crop(tensor):
        top = (tensor.shape[-2] - height) // 2
        left = (tensor.shape[-1] - width) // 2
        return tensor[..., top:top + height, left:left + width]

    return crop(target), crop(output)


def _assert_finite_parameters(model, description):
    for name, parameter in model.named_parameters():
        if parameter.is_floating_point() and parameter.dtype != torch.float32:
            raise ValueError(f'FI-VarNet parameter {name} is not FP32')
        if not torch.isfinite(parameter.detach()).all():
            raise FloatingPointError(f'nonfinite {description} parameter: {name}')


def _trainable_parameter_snapshot(model):
    snapshot = {
        name: parameter.detach().to(device='cpu', copy=True).contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not snapshot:
        raise FloatingPointError('nonfinite training fit: no trainable model parameters')
    return snapshot


def _parameter_snapshot_sha256(snapshot):
    digest = hashlib.sha256()
    for name in sorted(snapshot):
        tensor = snapshot[name]
        metadata = json.dumps(
            {
                'dtype': str(tensor.dtype),
                'name': name,
                'shape': list(tensor.shape),
            },
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
        payload = tensor.numpy().tobytes(order='C')
        digest.update(len(metadata).to_bytes(8, byteorder='big'))
        digest.update(metadata)
        digest.update(len(payload).to_bytes(8, byteorder='big'))
        digest.update(payload)
    return digest.hexdigest()


def run_one_finite_optimizer_step(
    model, sample, loss_fn, optimizer, scheduler, device
):
    """Execute exactly one FP32, no-autocast optimizer step with finite guards."""
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError(f'Unsupported smoke device: {device}')
    model.train()
    _assert_finite_parameters(model, 'pre-step')
    if len(optimizer.param_groups) != 1:
        raise ValueError('FI-VarNet smoke requires exactly one optimizer parameter group')
    nominal_step0_multiplier = float(fi_lr_multiplier(0))
    nominal_step0_lr = float(FI_ACC8_RECIPE.lr * nominal_step0_multiplier)
    observed_nominal_step0_lr = float(optimizer.param_groups[0]['lr'])
    if (
        not math.isfinite(observed_nominal_step0_lr)
        or observed_nominal_step0_lr != nominal_step0_lr
    ):
        raise ValueError('Optimizer must reflect nominal upstream scheduler step-0 LR')
    parameters_before = _trainable_parameter_snapshot(model)
    pre_step_parameter_sha256 = _parameter_snapshot_sha256(parameters_before)
    kspace = sample['kspace'].to(device=device, dtype=torch.float32)
    stored_mask = sample['mask']
    target = sample['target'].to(device=device, dtype=torch.float32)
    maximum = sample['maximum'].to(device=device, dtype=torch.float32)
    if any(tensor.dtype != torch.float32 for tensor in (kspace, target, maximum)):
        raise ValueError('FI-VarNet smoke requires FP32 kspace, target, and maximum')
    expected_mask_shape = (kspace.shape[0], 1, 1, kspace.shape[-2], 1)
    if stored_mask.dtype != torch.float32:
        raise ValueError('Stored FI-VarNet mask must be float32')
    if tuple(stored_mask.shape) != expected_mask_shape:
        raise ValueError(
            f'Stored FI-VarNet mask must have shape {expected_mask_shape}'
        )
    if not torch.isfinite(stored_mask).all():
        raise ValueError('Stored FI-VarNet mask must be finite')
    if not torch.all((stored_mask == 0.0) | (stored_mask == 1.0)):
        raise ValueError('Stored FI-VarNet mask must be binary')
    mask = stored_mask.to(device=device, dtype=torch.bool)
    if torch.count_nonzero(kspace.masked_select(~mask)).item() != 0:
        raise ValueError('Stored FI-VarNet mask was not applied to kspace')
    applied_mask_bytes = (
        stored_mask.detach().to(device='cpu', copy=True).reshape(-1).contiguous()
        .numpy().tobytes(order='C')
    )
    mask_contract = {
        'stored_dtype': 'float32',
        'stored_shape': [stored_mask.shape[-2]],
        'model_dtype': 'bool',
        'model_shape': list(mask.shape),
        'binary': True,
        'applied_to_kspace': True,
        'masked_out_kspace_zero': True,
        'applied_mask_sha256': hashlib.sha256(applied_mask_bytes).hexdigest(),
    }

    output = model(kspace, mask, crop_size=tuple(target.shape[-2:]))
    target, output = _center_crop_to_smallest(target, output)
    if not torch.isfinite(output).all():
        raise FloatingPointError('nonfinite model output')
    loss = loss_fn(output.unsqueeze(1), target.unsqueeze(1), maximum)
    if loss.numel() != 1 or not torch.isfinite(loss.detach()).all():
        raise FloatingPointError('nonfinite loss')

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_parameter_count = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient_parameter_count += 1
        if not torch.isfinite(parameter.grad.detach()).all():
            raise FloatingPointError(f'nonfinite parameter gradient: {name}')
    if gradient_parameter_count == 0:
        raise FloatingPointError('nonfinite training fit: no parameter gradients')

    # LambdaLR intentionally leaves the optimizer at the nominal upstream
    # step-0 LR (zero).  This non-resumable, exactly-one-step smoke primes only
    # its sole feasibility-probe update to the frozen base LR so the probe can
    # establish that the real model is trainable.  scheduler.step() below then
    # returns the optimizer to the untouched nominal upstream schedule.
    smoke_applied_lr = float(FI_ACC8_RECIPE.lr)
    if not math.isfinite(smoke_applied_lr) or smoke_applied_lr <= 0.0:
        raise ValueError('Smoke-only applied LR must be finite and positive')
    optimizer.param_groups[0]['lr'] = smoke_applied_lr
    if float(optimizer.param_groups[0]['lr']) != smoke_applied_lr:
        raise ValueError('Smoke-only applied LR priming did not take effect')
    optimizer.step()
    _assert_finite_parameters(model, 'post-step')
    parameters_after = _trainable_parameter_snapshot(model)
    if set(parameters_after) != set(parameters_before):
        raise RuntimeError('Trainable parameter set changed during smoke update')
    changed_parameter_count = sum(
        not torch.equal(parameters_before[name], parameters_after[name])
        for name in parameters_before
    )
    post_step_parameter_sha256 = _parameter_snapshot_sha256(parameters_after)
    if changed_parameter_count == 0:
        raise FloatingPointError(
            'Smoke optimizer step changed no trainable model parameters'
        )
    if post_step_parameter_sha256 == pre_step_parameter_sha256:
        raise RuntimeError('Smoke parameter digest did not change after optimizer step')

    scheduler.step()
    post_step_nominal_multiplier = float(fi_lr_multiplier(1))
    post_step_nominal_lr = float(optimizer.param_groups[0]['lr'])
    expected_post_step_nominal_lr = float(
        FI_ACC8_RECIPE.lr * post_step_nominal_multiplier
    )
    if (
        not math.isfinite(post_step_nominal_lr)
        or post_step_nominal_lr != expected_post_step_nominal_lr
    ):
        raise ValueError('Scheduler did not restore the nominal upstream post-step LR')
    return {
        'loss': float(loss.detach().cpu()),
        'gradient_parameter_count': gradient_parameter_count,
        'trainable_parameter_count': len(parameters_before),
        'changed_parameter_count': changed_parameter_count,
        'pre_step_parameter_sha256': pre_step_parameter_sha256,
        'post_step_parameter_sha256': post_step_parameter_sha256,
        'global_step': 1,
        'nominal_step0_multiplier': nominal_step0_multiplier,
        'nominal_step0_lr': nominal_step0_lr,
        'smoke_applied_lr': smoke_applied_lr,
        'lr_area': smoke_applied_lr,
        'post_step_nominal_multiplier': post_step_nominal_multiplier,
        'post_step_nominal_lr': post_step_nominal_lr,
        'mask_contract': mask_contract,
        'reconstruction': output.detach().to(device='cpu', copy=True),
    }


def _cpu_snapshot(value):
    if torch.is_tensor(value):
        return value.detach().to(device='cpu', copy=True)
    # Normalize primitive subclasses such as torch.torch_version.TorchVersion so
    # the checkpoint remains loadable by PyTorch's restricted weights-only loader.
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return value


def _rng_snapshot():
    numpy_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': {
            'name': numpy_state[0],
            # PyTorch 1.13 cannot serialize torch.uint32 storage.  Preserve every
            # MT19937 word losslessly in its supported signed 64-bit dtype.
            'keys': torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            'position': int(numpy_state[2]),
            'has_gauss': int(numpy_state[3]),
            'cached_gaussian': float(numpy_state[4]),
        },
        'torch_cpu': torch.get_rng_state().clone(),
    }


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_snapshot_equal(expected, actual, description='checkpoint'):
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual) or not torch.equal(expected, actual):
            raise ValueError(f'{description} tensor validation mismatch')
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f'{description} mapping validation mismatch')
        for key in expected:
            _assert_snapshot_equal(expected[key], actual[key], f'{description}.{key}')
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise ValueError(f'{description} sequence validation mismatch')
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _assert_snapshot_equal(expected_item, actual_item, f'{description}[{index}]')
        return
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f'{description} scalar validation mismatch')


def _validate_smoke_step_result(step_result):
    if not isinstance(step_result, dict):
        raise ValueError('Smoke step result must be a mapping')

    smoke_applied_lr = step_result.get('smoke_applied_lr')
    if (
        type(smoke_applied_lr) is not float
        or not math.isfinite(smoke_applied_lr)
        or smoke_applied_lr <= 0.0
    ):
        raise ValueError('Smoke-only applied LR must be finite and positive')
    expected_post_multiplier = float(fi_lr_multiplier(1))
    expected_post_lr = float(FI_ACC8_RECIPE.lr * expected_post_multiplier)
    exact_values = {
        'global_step': 1,
        'nominal_step0_multiplier': 0.0,
        'nominal_step0_lr': 0.0,
        'smoke_applied_lr': float(FI_ACC8_RECIPE.lr),
        'lr_area': float(FI_ACC8_RECIPE.lr),
        'post_step_nominal_multiplier': expected_post_multiplier,
        'post_step_nominal_lr': expected_post_lr,
    }
    for key, expected in exact_values.items():
        if type(step_result.get(key)) is not type(expected) or step_result.get(key) != expected:
            raise ValueError(f'Smoke step result has invalid {key}')

    positive_counts = (
        'gradient_parameter_count',
        'trainable_parameter_count',
        'changed_parameter_count',
    )
    for key in positive_counts:
        if type(step_result.get(key)) is not int or step_result[key] <= 0:
            raise ValueError(f'Smoke step result requires positive integer {key}')
    if step_result['changed_parameter_count'] > step_result['trainable_parameter_count']:
        raise ValueError('Changed parameter count exceeds trainable parameter count')
    if type(step_result.get('loss')) is not float or not math.isfinite(step_result['loss']):
        raise ValueError('Smoke step result requires finite float loss')

    pre_digest = step_result.get('pre_step_parameter_sha256')
    post_digest = step_result.get('post_step_parameter_sha256')
    if (
        not isinstance(pre_digest, str)
        or re.fullmatch(r'[0-9a-f]{64}', pre_digest) is None
        or not isinstance(post_digest, str)
        or re.fullmatch(r'[0-9a-f]{64}', post_digest) is None
        or pre_digest == post_digest
    ):
        raise ValueError('Smoke step result requires distinct parameter snapshot digests')

    mask_contract = step_result.get('mask_contract')
    mask_contract_keys = {
        'stored_dtype',
        'stored_shape',
        'model_dtype',
        'model_shape',
        'binary',
        'applied_to_kspace',
        'masked_out_kspace_zero',
        'applied_mask_sha256',
    }
    if not isinstance(mask_contract, dict) or set(mask_contract) != mask_contract_keys:
        raise ValueError('Smoke step result requires an exact mask contract')
    stored_shape = mask_contract['stored_shape']
    if (
        mask_contract['stored_dtype'] != 'float32'
        or not isinstance(stored_shape, list)
        or len(stored_shape) != 1
        or type(stored_shape[0]) is not int
        or stored_shape[0] <= 0
        or mask_contract['model_dtype'] != 'bool'
        or mask_contract['model_shape'] != [1, 1, 1, stored_shape[0], 1]
        or mask_contract['binary'] is not True
        or mask_contract['applied_to_kspace'] is not True
        or mask_contract['masked_out_kspace_zero'] is not True
        or not isinstance(mask_contract['applied_mask_sha256'], str)
        or re.fullmatch(r'[0-9a-f]{64}', mask_contract['applied_mask_sha256']) is None
    ):
        raise ValueError('Smoke step result has invalid mask contract')
    return step_result


def _validate_activation_checkpoint_evidence(container):
    activation_checkpointing = _validate_activation_checkpoint_contract(
        container.get('activation_checkpointing')
    )
    provenance = container.get('provenance')
    if not isinstance(provenance, dict):
        raise ValueError('FI activation checkpoint provenance is absent')
    provenance_contract = _validate_activation_checkpoint_contract(
        provenance.get('activation_checkpointing')
    )
    if provenance_contract != activation_checkpointing:
        raise ValueError('FI activation checkpoint evidence disagrees with provenance')
    return activation_checkpointing


def _validate_smoke_checkpoint(checkpoint):
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get('format_version') != 1
        or checkpoint.get('kind') != 'fi-varnet-acc8-training-fit-smoke'
        or checkpoint.get('scope') != 'SMOKE_ONLY'
        or checkpoint.get('resumable') is not False
        or checkpoint.get('nominal_resumable_step') is not False
        or checkpoint.get('optimizer_update_semantics')
        != 'SMOKE_ONLY_LR_PRIMED_FINITE_UPDATE_PROBE'
        or checkpoint.get('full_training_authorized') is not False
    ):
        raise ValueError('FI-VarNet smoke checkpoint must remain explicitly non-resumable')
    _validate_activation_checkpoint_evidence(checkpoint)
    if checkpoint.get('global_step') != 1:
        raise ValueError('FI-VarNet smoke checkpoint must contain exactly one step')
    _validate_smoke_step_result(checkpoint.get('smoke_result'))
    return checkpoint


def save_smoke_checkpoint(
    output_dir,
    *,
    model,
    optimizer,
    scheduler,
    step_result,
    sampler_state,
    provenance,
):
    """Publish one immutable, explicitly non-resumable CPU smoke checkpoint."""
    output_dir = Path(output_dir)
    step_result = _validate_smoke_step_result(step_result)
    provenance = dict(provenance)
    if 'activation_checkpointing' not in provenance:
        provenance['activation_checkpointing'] = dict(
            FI_ACTIVATION_CHECKPOINT_CONTRACT
        )
    activation_checkpointing = _validate_activation_checkpoint_contract(
        provenance['activation_checkpointing']
    )
    checkpoint = {
        'format_version': 1,
        'kind': 'fi-varnet-acc8-training-fit-smoke',
        'scope': 'SMOKE_ONLY',
        'resumable': False,
        'nominal_resumable_step': False,
        'optimizer_update_semantics': 'SMOKE_ONLY_LR_PRIMED_FINITE_UPDATE_PROBE',
        'full_training_authorized': False,
        'activation_checkpointing': activation_checkpointing,
        'recipe': FI_ACC8_RECIPE.as_dict(),
        'model': _cpu_snapshot(model.state_dict()),
        'optimizer': _cpu_snapshot(optimizer.state_dict()),
        'scheduler': _cpu_snapshot(scheduler.state_dict()),
        'rng': _rng_snapshot(),
        'sampler': _cpu_snapshot(sampler_state),
        'global_step': 1,
        'lr_area': float(step_result['lr_area']),
        'smoke_result': _cpu_snapshot(step_result),
        'provenance': _cpu_snapshot(provenance),
    }
    final_name = 'checkpoint-step-000001.pt'
    final_path = output_dir / final_name
    directory_fd = os.open(
        output_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    )
    named_stage = None
    publication_attempted = False
    publication_succeeded = False
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        try:
            handle = _open_anonymous_file(directory_fd)
        except OSError as exc:
            if exc.errno not in (errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL):
                raise
            # Trusted/cooperating-directory fallback: retain this random file's
            # descriptor through validation and publish that descriptor's inode.
            handle = tempfile.NamedTemporaryFile(
                mode='w+b',
                dir=f'/proc/self/fd/{directory_fd}',
                prefix='.checkpoint-unpublished-',
                delete=False,
            )
            named_stage = Path(handle.name).name
        with handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
            handle.seek(0)
            try:
                reloaded = torch.load(handle, map_location='cpu', weights_only=True)
                _validate_smoke_checkpoint(reloaded)
                _assert_snapshot_equal(checkpoint, reloaded)
            except BaseException as exc:
                raise ValueError(
                    'Smoke checkpoint validation failed before publication'
                ) from exc
            os.fchmod(handle.fileno(), 0o444)
            try:
                publication_attempted = True
                _publish_fd_without_overwrite(handle, directory_fd, final_name)
                publication_succeeded = True
            except OSError as exc:
                if exc.errno == errno.EEXIST:
                    raise FileExistsError(
                        f'Smoke checkpoint already exists: {final_path}'
                    ) from exc
                raise
    finally:
        sync_directory = publication_succeeded
        try:
            if named_stage is not None and (
                not publication_attempted or publication_succeeded
            ):
                try:
                    os.unlink(named_stage, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
                sync_directory = True
            if sync_directory:
                os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return {'path': final_path, 'sha256': _sha256_file(final_path)}


def load_smoke_checkpoint_cpu(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    return _validate_smoke_checkpoint(checkpoint)


@dataclass(frozen=True)
class Acc8DirectoryRecord:
    path: str
    st_dev: int
    st_ino: int


@dataclass(frozen=True)
class Acc8FileRecord:
    name: str
    kspace_path: Path
    image_path: Path
    slices: int
    kspace_shape: tuple
    target_shape: tuple
    mask_sha256: str
    kspace_sha256: str
    image_sha256: str
    kspace_size: int
    kspace_st_dev: int
    kspace_st_ino: int
    image_size: int
    image_st_dev: int
    image_st_ino: int


@dataclass(frozen=True)
class IgnoredAcc4FileRecord:
    name: str
    kspace_size: int
    kspace_st_dev: int
    kspace_st_ino: int
    kspace_st_mtime_ns: int
    kspace_st_ctime_ns: int
    image_size: int
    image_st_dev: int
    image_st_ino: int
    image_st_mtime_ns: int
    image_st_ctime_ns: int


@dataclass(frozen=True)
class Acc8DataManifest:
    root: Path
    directory_identities: tuple
    records: tuple
    ignored_acc4_records: tuple
    total_entries: int
    selected_acc8_count: int
    ignored_acc4_count: int
    ignored_acc4_identity_sha256: str
    slice_count: int
    selected_file: str
    selected_slice: int
    input_key: str
    target_key: str
    max_key: str
    maximum_input_shape: tuple
    manifest_sha256: str


def _directory_record(path, metadata):
    return Acc8DirectoryRecord(
        path=str(path), st_dev=int(metadata.st_dev), st_ino=int(metadata.st_ino)
    )


def _same_directory_identity(actual, expected):
    return (
        int(actual.st_dev) == expected.st_dev
        and int(actual.st_ino) == expected.st_ino
    )


def _open_directory_chain_nofollow(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('FI-VarNet data root must be an absolute path')
    opened = []
    identities = []
    try:
        current_fd = os.open(
            '/', os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        opened.append(current_fd)
        current_path = Path('/')
        identities.append(_directory_record(current_path, os.fstat(current_fd)))
        for component in path.parts[1:]:
            child_fd, identity = _open_child_directory_nofollow(
                current_fd, component, current_path / component
            )
            opened.append(child_fd)
            identities.append(identity)
            current_fd = child_fd
            current_path /= component
        return tuple(opened), tuple(identities)
    except (OSError, ValueError) as exc:
        for fd in reversed(opened):
            os.close(fd)
        if isinstance(exc, ValueError):
            raise
        raise ValueError('FI-VarNet data path must be nofollow directories') from exc


def _open_child_directory_nofollow(parent_fd, name, display_path):
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError('FI-VarNet data path must contain no symlink directories')
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError('FI-VarNet data path components must be directories')
        fd = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise ValueError('FI-VarNet data path must be nofollow directories') from exc
    opened = os.fstat(fd)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
    ):
        os.close(fd)
        raise ValueError('FI-VarNet data directory changed while opening')
    return fd, _directory_record(display_path, opened)


def _open_regular_file_nofollow(directory_fd, name):
    if not name or name in {'.', '..'} or Path(name).name != name:
        raise ValueError('FI-VarNet H5 name must be one basename')
    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f'FI-VarNet H5 leaf must not be a symlink: {name}')
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f'FI-VarNet H5 leaf must be a regular file: {name}')
        fd = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise ValueError(f'FI-VarNet H5 leaf must be nofollow regular file: {name}') from exc
    opened = os.fstat(fd)
    if (
        not stat.S_ISREG(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
    ):
        os.close(fd)
        raise ValueError(f'FI-VarNet H5 identity changed while opening: {name}')
    return fd, opened


def _stat_ignored_acc4_nofollow(directory_fd, name, description):
    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f'{description} must be a nofollow entry: {name}') from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f'{description} must not be a symlink: {name}')
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f'{description} must be a regular file: {name}')
    return metadata


def _stat_signature(metadata):
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _sha256_fd(fd, description):
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f'{description} is not a regular file')
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(fd, min(1024 * 1024, before.st_size - offset), offset)
        if not chunk:
            raise ValueError(f'{description} changed while hashing')
        digest.update(chunk)
        offset += len(chunk)
    after = os.fstat(fd)
    if offset != before.st_size or _stat_signature(before) != _stat_signature(after):
        raise ValueError(f'{description} changed while hashing')
    return digest.hexdigest(), after


@contextmanager
def _h5_from_fd(fd):
    stream = os.fdopen(fd, 'rb', buffering=0, closefd=False)
    try:
        with h5py.File(stream, 'r') as h5_file:
            yield h5_file
    finally:
        stream.close()


def _classify_acceleration_name(name):
    if (
        not isinstance(name, str)
        or not name
        or name in {'.', '..'}
        or Path(name).name != name
        or Path(name).suffix != '.h5'
    ):
        raise ValueError(f'FI-VarNet organizer filename contract rejected {name!r}')
    tokens = Path(name).stem.split('_')
    acceleration_tokens = [token for token in tokens if token in {'acc4', 'acc8'}]
    if len(acceleration_tokens) > 1:
        raise ValueError(
            f'FI-VarNet organizer filename has ambiguous acceleration tokens: {name!r}'
        )
    if any(not token for token in tokens) or len(acceleration_tokens) != 1:
        raise ValueError(
            'FI-VarNet organizer filename contract requires exactly one lowercase '
            f'acc4 or acc8 token: {name!r}'
        )
    return acceleration_tokens[0]


def inspect_acc8_training_data(
    root,
    *,
    input_key,
    target_key,
    max_key,
    expected_files=FI_ACC8_RECIPE.train_files,
    expected_slices=FI_ACC8_RECIPE.slices_per_epoch,
    maximum_input_shape=(15, 640, 386),
    expected_root=None,
    expected_total_files=None,
    expected_ignored_acc4_files=None,
):
    """Validate paired acc8 H5 metadata through descriptor-bound H5 reads."""
    root = Path(root)
    if expected_root is not None and root != Path(expected_root):
        raise ValueError(f'FI-VarNet production data root must be {Path(expected_root)}')
    if expected_root is not None and Path(expected_root) == FI_ACC8_PRODUCTION_ROOT:
        if expected_total_files not in (None, FI_ACC8_ORGANIZER_TOTAL_FILES):
            raise ValueError('FI-VarNet production organizer total expectation is frozen')
        if expected_ignored_acc4_files not in (None, FI_ACC8_CORESIDENT_ACC4_FILES):
            raise ValueError('FI-VarNet production organizer acc4 expectation is frozen')
        expected_total_files = FI_ACC8_ORGANIZER_TOTAL_FILES
        expected_ignored_acc4_files = FI_ACC8_CORESIDENT_ACC4_FILES

    directory_fds, root_identities = _open_directory_chain_nofollow(root)
    kspace_directory_fd = None
    image_directory_fd = None
    try:
        kspace_directory_fd, kspace_identity = _open_child_directory_nofollow(
            directory_fds[-1], 'kspace', root / 'kspace'
        )
        image_directory_fd, image_identity = _open_child_directory_nofollow(
            directory_fds[-1], 'image', root / 'image'
        )
        directory_identities = root_identities + (kspace_identity, image_identity)
        kspace_names = set(os.listdir(kspace_directory_fd))
        image_names = set(os.listdir(image_directory_fd))
        if kspace_names != image_names:
            raise ValueError('FI-VarNet image and kspace filename sets must match exactly')
        classified_names = {
            name: _classify_acceleration_name(name) for name in kspace_names
        }
        acc8_names = {
            name for name, acceleration in classified_names.items() if acceleration == 'acc8'
        }
        acc4_names = {
            name for name, acceleration in classified_names.items() if acceleration == 'acc4'
        }
        if expected_total_files is not None and len(kspace_names) != expected_total_files:
            raise ValueError(
                f'FI-VarNet organizer total entry count must be {expected_total_files}, '
                f'found {len(kspace_names)}'
            )
        if (
            expected_ignored_acc4_files is not None
            and len(acc4_names) != expected_ignored_acc4_files
        ):
            raise ValueError(
                f'FI-VarNet ignored acc4 count must be {expected_ignored_acc4_files}, '
                f'found {len(acc4_names)}'
            )
        if len(acc8_names) != expected_files:
            raise ValueError(
                f'FI-VarNet selected acc8 file count must be {expected_files}, '
                f'found {len(acc8_names)}'
            )

        ignored_acc4_stats = {}
        ignored_acc4_records = []
        for name in sorted(acc4_names):
            kspace_stat = _stat_ignored_acc4_nofollow(
                kspace_directory_fd, name, 'FI-VarNet ignored acc4 kspace entry'
            )
            image_stat = _stat_ignored_acc4_nofollow(
                image_directory_fd, name, 'FI-VarNet ignored acc4 image entry'
            )
            ignored_acc4_stats[name] = (kspace_stat, image_stat)
            ignored_acc4_records.append(
                IgnoredAcc4FileRecord(
                    name=name,
                    kspace_size=int(kspace_stat.st_size),
                    kspace_st_dev=int(kspace_stat.st_dev),
                    kspace_st_ino=int(kspace_stat.st_ino),
                    kspace_st_mtime_ns=int(kspace_stat.st_mtime_ns),
                    kspace_st_ctime_ns=int(kspace_stat.st_ctime_ns),
                    image_size=int(image_stat.st_size),
                    image_st_dev=int(image_stat.st_dev),
                    image_st_ino=int(image_stat.st_ino),
                    image_st_mtime_ns=int(image_stat.st_mtime_ns),
                    image_st_ctime_ns=int(image_stat.st_ctime_ns),
                )
            )

        records = []
        total_slices = 0
        max_candidates = []
        for name in sorted(acc8_names):
            kspace_fd = None
            image_fd = None
            try:
                kspace_fd, kspace_open_stat = _open_regular_file_nofollow(
                    kspace_directory_fd, name
                )
                image_fd, image_open_stat = _open_regular_file_nofollow(
                    image_directory_fd, name
                )
                kspace_sha256, kspace_before = _sha256_fd(
                    kspace_fd, f'kspace H5 {name}'
                )
                image_sha256, image_before = _sha256_fd(
                    image_fd, f'image H5 {name}'
                )
                if (
                    _stat_signature(kspace_open_stat) != _stat_signature(kspace_before)
                    or _stat_signature(image_open_stat) != _stat_signature(image_before)
                ):
                    raise ValueError(f'H5 identity changed before preflight: {name}')

                with _h5_from_fd(kspace_fd) as khf, _h5_from_fd(image_fd) as ihf:
                    if input_key not in khf or target_key not in ihf or 'mask' not in khf:
                        raise ValueError(f'Missing required dataset in {name}')
                    kspace_dataset = khf[input_key]
                    target_dataset = ihf[target_key]
                    mask_dataset = khf['mask']
                    if len(kspace_dataset.shape) != 4:
                        raise ValueError(f'Malformed kspace shape in {name}')
                    slices = int(kspace_dataset.shape[0])
                    if not target_dataset.shape or int(target_dataset.shape[0]) != slices:
                        raise ValueError(f'Image/kspace slice count mismatch in {name}')
                    kspace_shape = tuple(int(value) for value in kspace_dataset.shape)
                    target_shape = tuple(int(value) for value in target_dataset.shape)
                    spatial_shape = kspace_shape[1:]
                    if len(spatial_shape) != len(maximum_input_shape) or any(
                        value <= 0 for value in spatial_shape
                    ) or any(
                        actual > maximum
                        for actual, maximum in zip(spatial_shape, maximum_input_shape)
                    ):
                        raise ValueError(
                            f'Kspace exceeds frozen maximum input shape in {name}'
                        )
                    if mask_dataset.dtype != np.dtype(np.float32):
                        raise ValueError(f'Stored mask must be float32 in {name}')
                    if mask_dataset.shape != (spatial_shape[-1],):
                        raise ValueError(f'Stored mask must be a [width] vector in {name}')
                    mask = np.asarray(mask_dataset[...])
                    if not np.all((mask == 0.0) | (mask == 1.0)):
                        raise ValueError(f'Stored mask must be binary in {name}')
                    if max_key not in ihf.attrs or not math.isfinite(
                        float(ihf.attrs[max_key])
                    ):
                        raise ValueError(
                            f'Missing or nonfinite {max_key!r} attribute in {name}'
                        )
                    if spatial_shape == tuple(maximum_input_shape):
                        max_candidates.extend((name, index) for index in range(slices))

                kspace_after_sha256, kspace_after = _sha256_fd(
                    kspace_fd, f'kspace H5 {name}'
                )
                image_after_sha256, image_after = _sha256_fd(
                    image_fd, f'image H5 {name}'
                )
                if (
                    kspace_after_sha256 != kspace_sha256
                    or image_after_sha256 != image_sha256
                    or _stat_signature(kspace_after) != _stat_signature(kspace_before)
                    or _stat_signature(image_after) != _stat_signature(image_before)
                ):
                    raise ValueError(f'H5 bytes changed during preflight: {name}')
                records.append(
                    Acc8FileRecord(
                        name=name,
                        kspace_path=root / 'kspace' / name,
                        image_path=root / 'image' / name,
                        slices=slices,
                        kspace_shape=kspace_shape,
                        target_shape=target_shape,
                        mask_sha256=hashlib.sha256(mask.tobytes(order='C')).hexdigest(),
                        kspace_sha256=kspace_sha256,
                        image_sha256=image_sha256,
                        kspace_size=int(kspace_before.st_size),
                        kspace_st_dev=int(kspace_before.st_dev),
                        kspace_st_ino=int(kspace_before.st_ino),
                        image_size=int(image_before.st_size),
                        image_st_dev=int(image_before.st_dev),
                        image_st_ino=int(image_before.st_ino),
                    )
                )
                total_slices += slices
            finally:
                if image_fd is not None:
                    os.close(image_fd)
                if kspace_fd is not None:
                    os.close(kspace_fd)

        for name, (kspace_before, image_before) in ignored_acc4_stats.items():
            kspace_after = _stat_ignored_acc4_nofollow(
                kspace_directory_fd, name, 'FI-VarNet ignored acc4 kspace entry'
            )
            image_after = _stat_ignored_acc4_nofollow(
                image_directory_fd, name, 'FI-VarNet ignored acc4 image entry'
            )
            if (
                _stat_signature(kspace_after) != _stat_signature(kspace_before)
                or _stat_signature(image_after) != _stat_signature(image_before)
            ):
                raise ValueError(
                    f'FI-VarNet ignored acc4 identity changed during preflight: {name}'
                )
        if total_slices != expected_slices:
            raise ValueError(
                f'FI-VarNet acc8 slice count must be {expected_slices}, found {total_slices}'
            )
        if not max_candidates:
            raise ValueError(
                f'No maximum input slice with shape {tuple(maximum_input_shape)} was found'
            )
        for fd, identity in zip(directory_fds, root_identities):
            if not _same_directory_identity(os.fstat(fd), identity):
                raise ValueError('FI-VarNet data root identity changed during preflight')
        if not _same_directory_identity(
            os.fstat(kspace_directory_fd), kspace_identity
        ) or not _same_directory_identity(os.fstat(image_directory_fd), image_identity):
            raise ValueError('FI-VarNet data subdirectory identity changed during preflight')
    finally:
        if image_directory_fd is not None:
            os.close(image_directory_fd)
        if kspace_directory_fd is not None:
            os.close(kspace_directory_fd)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)

    selected_file, selected_slice = max_candidates[0]
    ignored_acc4_identity_bytes = json.dumps(
        [asdict(record) for record in ignored_acc4_records],
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    ignored_acc4_identity_sha256 = hashlib.sha256(
        ignored_acc4_identity_bytes
    ).hexdigest()
    canonical = {
        'directories': [asdict(identity) for identity in directory_identities],
        'inventory_counts': {
            'total_entries': len(kspace_names),
            'selected_acc8_count': len(acc8_names),
            'ignored_acc4_count': len(acc4_names),
        },
        'ignored_acc4_files': [
            asdict(record) for record in ignored_acc4_records
        ],
        'ignored_acc4_identity_sha256': ignored_acc4_identity_sha256,
        'selected_acc8_files': [
            {
                'name': record.name,
                'slices': record.slices,
                'kspace_shape': record.kspace_shape,
                'target_shape': record.target_shape,
                'mask_sha256': record.mask_sha256,
                'kspace_sha256': record.kspace_sha256,
                'image_sha256': record.image_sha256,
                'kspace_size': record.kspace_size,
                'kspace_st_dev': record.kspace_st_dev,
                'kspace_st_ino': record.kspace_st_ino,
                'image_size': record.image_size,
                'image_st_dev': record.image_st_dev,
                'image_st_ino': record.image_st_ino,
            }
            for record in records
        ],
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    return Acc8DataManifest(
        root=root,
        directory_identities=directory_identities,
        records=tuple(records),
        ignored_acc4_records=tuple(ignored_acc4_records),
        total_entries=len(kspace_names),
        selected_acc8_count=len(acc8_names),
        ignored_acc4_count=len(acc4_names),
        ignored_acc4_identity_sha256=ignored_acc4_identity_sha256,
        slice_count=total_slices,
        selected_file=selected_file,
        selected_slice=selected_slice,
        input_key=input_key,
        target_key=target_key,
        max_key=max_key,
        maximum_input_shape=tuple(maximum_input_shape),
        manifest_sha256=manifest_sha256,
    )


def _require_file_identity(metadata, *, size, st_dev, st_ino, description):
    if (
        int(metadata.st_size) != size
        or int(metadata.st_dev) != st_dev
        or int(metadata.st_ino) != st_ino
    ):
        raise ValueError(f'{description} identity changed after preflight')


def load_fi_acc8_sample(manifest):
    """Load the selected maximum input from revalidated descriptor-bound H5 files."""
    matching_records = [
        record for record in manifest.records if record.name == manifest.selected_file
    ]
    if len(matching_records) != 1:
        raise ValueError('Selected H5 inventory identity is missing or ambiguous')
    record = matching_records[0]

    directory_fds, root_identities = _open_directory_chain_nofollow(manifest.root)
    kspace_directory_fd = None
    image_directory_fd = None
    kspace_fd = None
    image_fd = None
    try:
        kspace_directory_fd, kspace_identity = _open_child_directory_nofollow(
            directory_fds[-1], 'kspace', manifest.root / 'kspace'
        )
        image_directory_fd, image_identity = _open_child_directory_nofollow(
            directory_fds[-1], 'image', manifest.root / 'image'
        )
        actual_directories = root_identities + (kspace_identity, image_identity)
        if actual_directories != manifest.directory_identities:
            raise ValueError('FI-VarNet data directory identity changed after preflight')

        kspace_fd, kspace_open_stat = _open_regular_file_nofollow(
            kspace_directory_fd, record.name
        )
        image_fd, image_open_stat = _open_regular_file_nofollow(
            image_directory_fd, record.name
        )
        _require_file_identity(
            kspace_open_stat,
            size=record.kspace_size,
            st_dev=record.kspace_st_dev,
            st_ino=record.kspace_st_ino,
            description='Selected kspace H5',
        )
        _require_file_identity(
            image_open_stat,
            size=record.image_size,
            st_dev=record.image_st_dev,
            st_ino=record.image_st_ino,
            description='Selected image H5',
        )
        kspace_before_sha256, kspace_before = _sha256_fd(
            kspace_fd, 'selected kspace H5'
        )
        image_before_sha256, image_before = _sha256_fd(
            image_fd, 'selected image H5'
        )
        _require_file_identity(
            kspace_before,
            size=record.kspace_size,
            st_dev=record.kspace_st_dev,
            st_ino=record.kspace_st_ino,
            description='Selected kspace H5',
        )
        _require_file_identity(
            image_before,
            size=record.image_size,
            st_dev=record.image_st_dev,
            st_ino=record.image_st_ino,
            description='Selected image H5',
        )
        if (
            kspace_before_sha256 != record.kspace_sha256
            or image_before_sha256 != record.image_sha256
        ):
            raise ValueError('Selected H5 bytes changed after preflight')

        with _h5_from_fd(kspace_fd) as khf, _h5_from_fd(image_fd) as ihf:
            kspace = np.asarray(
                khf[manifest.input_key][manifest.selected_slice], dtype=np.complex64
            )
            mask = np.asarray(khf['mask'][...], dtype=np.float32)
            target = np.asarray(
                ihf[manifest.target_key][manifest.selected_slice], dtype=np.float32
            )
            maximum = float(ihf.attrs[manifest.max_key])

        kspace_after_sha256, kspace_after = _sha256_fd(
            kspace_fd, 'selected kspace H5'
        )
        image_after_sha256, image_after = _sha256_fd(
            image_fd, 'selected image H5'
        )
        if (
            kspace_after_sha256 != record.kspace_sha256
            or image_after_sha256 != record.image_sha256
            or _stat_signature(kspace_after) != _stat_signature(kspace_before)
            or _stat_signature(image_after) != _stat_signature(image_before)
        ):
            raise ValueError('Selected H5 bytes changed while loading sample')
        for fd, identity in zip(directory_fds, root_identities):
            if not _same_directory_identity(os.fstat(fd), identity):
                raise ValueError('FI-VarNet data root identity changed while loading sample')
        if not _same_directory_identity(
            os.fstat(kspace_directory_fd), kspace_identity
        ) or not _same_directory_identity(os.fstat(image_directory_fd), image_identity):
            raise ValueError('FI-VarNet data directory identity changed while loading sample')
    finally:
        if image_fd is not None:
            os.close(image_fd)
        if kspace_fd is not None:
            os.close(kspace_fd)
        if image_directory_fd is not None:
            os.close(image_directory_fd)
        if kspace_directory_fd is not None:
            os.close(kspace_directory_fd)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)

    if tuple(kspace.shape) != manifest.maximum_input_shape:
        raise ValueError('Selected maximum input changed after preflight')
    if mask.shape != (kspace.shape[-1],) or mask.dtype != np.float32:
        raise ValueError('Stored mask changed after preflight')
    masked = np.ascontiguousarray(kspace * mask)
    return {
        'mask': torch.from_numpy(mask.copy()).reshape(1, 1, 1, kspace.shape[-1], 1),
        'kspace': torch.view_as_real(torch.from_numpy(masked)).unsqueeze(0),
        'target': torch.from_numpy(np.ascontiguousarray(target)).unsqueeze(0),
        'maximum': torch.tensor([maximum], dtype=torch.float32),
        'fname': record.name,
        'slice': manifest.selected_slice,
    }


def _select_smoke_device(gpu_index):
    """Select CUDA only after source/GPU/output gates have completed."""
    if not torch.cuda.is_available():
        raise RuntimeError('FI-VarNet training-fit smoke requires the reviewed CUDA GPU')
    device = torch.device(f'cuda:{gpu_index}')
    torch.cuda.set_device(device)
    name = torch.cuda.get_device_name(device)
    if name not in {'GeForce GTX 1080', 'NVIDIA GeForce GTX 1080'}:
        raise RuntimeError(f'CUDA device name does not match GTX 1080 contract: {name!r}')
    return device


def _write_fsynced(path, payload):
    with Path(path).open('wb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def run_fi_acc8_training_fit_smoke(args, output_dir):
    """Run and atomically publish the sole authorized one-step training-fit smoke."""
    production_root = Path('/root/Data/train')
    if Path(args.data_path_train) != production_root:
        raise ValueError('FI-VarNet production data root must be /root/Data/train')
    source_provenance = verify_pinned_upstream_sources()
    gpu_provenance = preflight_smoke_gpu(args.GPU_NUM, args.expected_gpu_uuid)

    final_dir = Path(os.path.abspath(os.fspath(output_dir)))
    staged = _create_staged_directory(final_dir, 'FI-VarNet smoke')
    stage_path = _staged_directory_descriptor_path(staged)
    _write_fsynced(stage_path / 'INCOMPLETE', b'FI-VARNET-ACC8-SMOKE-INCOMPLETE\n')
    try:
        device = _select_smoke_device(args.GPU_NUM)
        random.seed(FI_ACC8_RECIPE.seed)
        np.random.seed(FI_ACC8_RECIPE.seed)
        torch.manual_seed(FI_ACC8_RECIPE.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(FI_ACC8_RECIPE.seed)

        model = build_model(args).to(device=device, dtype=torch.float32)
        activation_checkpointing = _validate_activation_checkpoint_contract(
            enable_fi_activation_checkpointing(model)
        )
        loss_fn = build_pinned_ssim_loss()
        if hasattr(loss_fn, 'to'):
            loss_fn = loss_fn.to(device=device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=FI_ACC8_RECIPE.lr,
            weight_decay=FI_ACC8_RECIPE.weight_decay,
        )
        scheduler = build_fi_scheduler(optimizer)

        manifest = inspect_acc8_training_data(
            args.data_path_train,
            input_key=args.input_key,
            target_key=args.target_key,
            max_key=args.max_key,
            expected_root=production_root,
        )
        sample = load_fi_acc8_sample(manifest)
        step_result = run_one_finite_optimizer_step(
            model, sample, loss_fn, optimizer, scheduler, device
        )
        reconstruction = step_result.pop('reconstruction')
        selected_records = [
            record for record in manifest.records if record.name == manifest.selected_file
        ]
        if len(selected_records) != 1:
            raise ValueError('Selected FI-VarNet manifest record is missing or ambiguous')
        mask_contract = step_result['mask_contract']
        if mask_contract['applied_mask_sha256'] != selected_records[0].mask_sha256:
            raise ValueError('Applied FI-VarNet mask digest does not match inventory')

        canonical_recipe = json.dumps(
            FI_ACC8_RECIPE.as_dict(), sort_keys=True, separators=(',', ':')
        ).encode('utf-8')
        provenance = {
            'source': source_provenance,
            'activation_checkpointing': activation_checkpointing,
            'data': {
                'manifest_sha256': manifest.manifest_sha256,
                'organizer_total_entries': manifest.total_entries,
                'selected_acc8_count': manifest.selected_acc8_count,
                'ignored_acc4_count': manifest.ignored_acc4_count,
                'ignored_acc4_identity_sha256': (
                    manifest.ignored_acc4_identity_sha256
                ),
                'ignored_acc4_access': {
                    'method': 'nofollow-stat-only',
                    'payload_opened': False,
                    'payload_hashed': False,
                    'h5_read': False,
                },
                'mask_contract': mask_contract,
                'slice_count': manifest.slice_count,
                'selected_file': manifest.selected_file,
                'selected_slice': manifest.selected_slice,
                'root': str(manifest.root),
                'directories': [
                    asdict(identity) for identity in manifest.directory_identities
                ],
                'ignored_acc4_files': [
                    asdict(record) for record in manifest.ignored_acc4_records
                ],
                'selected_acc8_files': [
                    {
                        'name': record.name,
                        'slices': record.slices,
                        'kspace_shape': record.kspace_shape,
                        'target_shape': record.target_shape,
                        'mask_sha256': record.mask_sha256,
                        'kspace_sha256': record.kspace_sha256,
                        'image_sha256': record.image_sha256,
                        'kspace_size': record.kspace_size,
                        'kspace_st_dev': record.kspace_st_dev,
                        'kspace_st_ino': record.kspace_st_ino,
                        'image_size': record.image_size,
                        'image_st_dev': record.image_st_dev,
                        'image_st_ino': record.image_st_ino,
                    }
                    for record in manifest.records
                ],
            },
            'recipe': {
                'sha256': hashlib.sha256(canonical_recipe).hexdigest(),
                'values': FI_ACC8_RECIPE.as_dict(),
            },
            'optimizer': {
                'class': 'torch.optim.AdamW',
                'lr': FI_ACC8_RECIPE.lr,
                'weight_decay': FI_ACC8_RECIPE.weight_decay,
                'betas': tuple(optimizer.defaults['betas']),
                'eps': float(optimizer.defaults['eps']),
            },
            'scheduler': {
                'class': 'torch.optim.lr_scheduler.LambdaLR',
                'ramp_steps': FI_ACC8_RECIPE.ramp_steps,
                'cosine_decay_start': FI_ACC8_RECIPE.cosine_decay_start,
                'max_steps': FI_ACC8_RECIPE.max_steps,
                'nominal_step0': {
                    'multiplier': step_result['nominal_step0_multiplier'],
                    'lr': step_result['nominal_step0_lr'],
                },
                'smoke_only_lr_priming': {
                    'applied_lr': step_result['smoke_applied_lr'],
                    'nominal_schedule_definition_modified': False,
                    'optimizer_steps': 1,
                    'purpose': 'finite-update-feasibility-probe',
                    'resumable': False,
                },
                'post_step_nominal': {
                    'multiplier': step_result['post_step_nominal_multiplier'],
                    'lr': step_result['post_step_nominal_lr'],
                },
            },
            'runtime': {
                'python_version': sys.version,
                'torch_version': str(torch.__version__),
                'numpy_version': str(np.__version__),
                'device_type': device.type,
                'gpu_preflight': gpu_provenance,
            },
        }
        sampler_state = {
            'algorithm': 'maximum-input-candidate-first',
            'seed': FI_ACC8_RECIPE.seed,
            'selected_file': manifest.selected_file,
            'selected_slice': manifest.selected_slice,
            'acceleration': FI_ACC8_RECIPE.acceleration,
        }
        checkpoint_publication = save_smoke_checkpoint(
            stage_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step_result=step_result,
            sampler_state=sampler_state,
            provenance=provenance,
        )

        reconstruction_path = stage_path / 'reconstruction-step-000001.pt'
        torch.save(reconstruction, reconstruction_path)
        reloaded_reconstruction = torch.load(
            reconstruction_path, map_location='cpu', weights_only=True
        )
        _assert_snapshot_equal(reconstruction, reloaded_reconstruction, 'reconstruction')
        reconstruction_fd = os.open(reconstruction_path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(reconstruction_fd)
            os.fchmod(reconstruction_fd, 0o444)
        finally:
            os.close(reconstruction_fd)

        report = {
            'schema': 'fi-varnet-acc8-training-fit-smoke-report-v1',
            'scope': 'SMOKE_ONLY',
            'resumable': False,
            'nominal_resumable_step': False,
            'optimizer_update_semantics': 'SMOKE_ONLY_LR_PRIMED_FINITE_UPDATE_PROBE',
            'full_training_authorized': False,
            'evaluation_authorized': False,
            'submission_authorized': False,
            'activation_checkpointing': activation_checkpointing,
            'global_step': step_result['global_step'],
            'nominal_step0_multiplier': step_result['nominal_step0_multiplier'],
            'nominal_step0_lr': step_result['nominal_step0_lr'],
            'smoke_applied_lr': step_result['smoke_applied_lr'],
            'lr_area': step_result['lr_area'],
            'post_step_nominal_multiplier': step_result[
                'post_step_nominal_multiplier'
            ],
            'post_step_nominal_lr': step_result['post_step_nominal_lr'],
            'loss': step_result['loss'],
            'gradient_parameter_count': step_result['gradient_parameter_count'],
            'trainable_parameter_count': step_result['trainable_parameter_count'],
            'changed_parameter_count': step_result['changed_parameter_count'],
            'pre_step_parameter_sha256': step_result['pre_step_parameter_sha256'],
            'post_step_parameter_sha256': step_result['post_step_parameter_sha256'],
            'mask_contract': mask_contract,
            'checkpoint_sha256': checkpoint_publication['sha256'],
            'reconstruction_sha256': _sha256_file(reconstruction_path),
            'sampler': sampler_state,
            'provenance': provenance,
        }
        _validate_activation_checkpoint_evidence(report)
        _write_fsynced(
            stage_path / 'report.json',
            (json.dumps(report, sort_keys=True, indent=2) + '\n').encode('utf-8'),
        )
        os.replace(stage_path / 'INCOMPLETE', stage_path / 'COMPLETE')
        _seal_staged_directory(staged)
        _publish_staged_directory_no_replace(staged, final_dir, 'FI-VarNet smoke')
        staged = None
    finally:
        if staged is not None:
            _cleanup_staged_directory(staged)

    checkpoint_path = final_dir / 'checkpoint-step-000001.pt'
    return {
        'output_dir': final_dir,
        'checkpoint_path': checkpoint_path,
        'checkpoint_sha256': _sha256_file(checkpoint_path),
        'checkpoint': load_smoke_checkpoint_cpu(checkpoint_path),
        'report': report,
    }

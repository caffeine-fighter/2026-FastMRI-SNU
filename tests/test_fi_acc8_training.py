import copy
from contextlib import contextmanager
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import NamedTuple
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import torch

import train
import utils.learning.fi_acc8_training as fi_acc8_training_module
import utils.model.fi_varnet_adapter as fi_varnet_adapter_module
from utils.learning.fi_acc8_training import (
    FI_ACC8_RECIPE,
    build_fi_scheduler,
    fi_lr_multiplier,
    inspect_acc8_training_data,
    load_fi_acc8_sample,
    preflight_smoke_gpu,
    prepare_smoke_output,
    run_one_finite_optimizer_step,
    save_smoke_checkpoint,
    load_smoke_checkpoint_cpu,
)
from utils.learning.train_part import build_model
from utils.model.fi_varnet_adapter import (
    PINNED_FEATURE_VARNET_SHA256,
    PINNED_LICENSE_SHA256,
    PINNED_LOSSES_SHA256,
    PINNED_UPSTREAM_COMMIT,
    load_pinned_fi_varnet_class,
    load_pinned_ssim_loss_class,
)


class FrozenRecipeTests(unittest.TestCase):
    def test_recipe_is_the_exact_frozen_40_epoch_contract(self):
        self.assertEqual(
            FI_ACC8_RECIPE.as_dict(),
            {
                "schema": "fi-varnet-acc8-training-recipe-v1",
                "model_family": "fi-varnet-acc8",
                "scratch": True,
                "external_learned_state": False,
                "seed": 431,
                "batch_size": 1,
                "precision": "fp32",
                "autocast": False,
                "optimizer": "AdamW",
                "lr": 3e-4,
                "weight_decay": 0.0,
                "loss": "upstream-fastmri-SSIMLoss",
                "gradient_clipping": False,
                "num_cascades": 12,
                "chans": 18,
                "pools": 4,
                "sens_chans": 8,
                "sens_pools": 4,
                "acceleration": 8,
                "train_files": 85,
                "slices_per_epoch": 2315,
                "epochs": 40,
                "max_steps": 92600,
                "ramp_steps": 3704,
                "cosine_decay_start": 46300,
                "checkpoint_every_epoch": True,
                "reconstruction_every_epoch": True,
            },
        )

    def test_scheduler_matches_upstream_functional_form_at_boundaries(self):
        self.assertEqual(fi_lr_multiplier(0), 0.0)
        self.assertAlmostEqual(fi_lr_multiplier(1852), 0.5)
        self.assertEqual(fi_lr_multiplier(3704), 1.0)
        self.assertEqual(fi_lr_multiplier(46299), 1.0)
        self.assertEqual(fi_lr_multiplier(46300), 1.0)
        expected = math.cos(((69450 - 46300) / (92600 - 46300)) * math.pi / 2)
        self.assertAlmostEqual(fi_lr_multiplier(69450), expected)
        self.assertEqual(fi_lr_multiplier(92600), 1e-8)
        self.assertEqual(fi_lr_multiplier(100000), 1e-8)

    def test_scheduler_is_step_lambda_lr_with_frozen_scalars(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=FI_ACC8_RECIPE.lr)
        scheduler = build_fi_scheduler(optimizer)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(
            optimizer.param_groups[0]["lr"],
            FI_ACC8_RECIPE.lr * fi_lr_multiplier(1),
        )


class FrozenCliTests(unittest.TestCase):
    def parse(self, *extra):
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--model-family",
                "fi-varnet-acc8",
                "--data-path-train",
                "/root/Data/train",
                *extra,
            ],
        ):
            return train.parse()

    def test_fi_mode_applies_frozen_values_and_requires_smoke_opt_in(self):
        args = self.parse("--fi-acc8-one-step-smoke", "--expected-gpu-uuid", "GPU-abc")
        self.assertEqual(args.num_epochs, 40)
        self.assertEqual(args.lr, 3e-4)
        self.assertEqual(args.seed, 431)
        self.assertEqual(args.cascade, 12)
        self.assertEqual(args.chans, 18)
        self.assertEqual(args.sens_chans, 8)
        self.assertEqual(args.pools, 4)
        self.assertEqual(args.sens_pools, 4)
        self.assertEqual(args.acceleration, 8)
        self.assertEqual(args.precision, "fp32")
        self.assertEqual(args.net_name, Path('LOCAL_FI_ACC8_CKPT_SMOKE_R1'))

    def test_fi_smoke_requires_exact_organizer_train_root(self):
        with self.assertRaises(SystemExit):
            self.parse(
                '--fi-acc8-one-step-smoke',
                '--expected-gpu-uuid',
                'GPU-abc',
                '--data-path-train',
                '/tmp/train',
            )

    def test_full_fi_training_is_explicitly_blocked(self):
        with self.assertRaises(SystemExit):
            self.parse("--expected-gpu-uuid", "GPU-abc")

    def test_recipe_mutations_are_rejected(self):
        mutations = (
            ("--batch-size", "2"),
            ("--num-epochs", "39"),
            ("--lr", "0.001"),
            ("--seed", "430"),
            ("--cascade", "11"),
            ("--chans", "17"),
            ("--sens_chans", "4"),
            ("--pools", "3"),
            ("--sens-pools", "3"),
            ("--acceleration", "4"),
            ("--precision", "bf16"),
            ("--weight-decay", "0.01"),
            ("--ramp-steps", "3703"),
            ("--cosine-decay-start", "46299"),
            ("--max-steps", "92599"),
            ("--input-key", "alternate_kspace"),
            ("--target-key", "alternate_target"),
            ("--max-key", "alternate_max"),
            ("--net-name", "LOCAL_FI_ACC8_SMOKE_R1"),
        )
        for option, value in mutations:
            with self.subTest(option=option):
                with self.assertRaises(SystemExit):
                    self.parse(
                        "--fi-acc8-one-step-smoke",
                        "--expected-gpu-uuid",
                        "GPU-abc",
                        option,
                        value,
                    )

    def test_resume_and_learned_state_are_rejected(self):
        forbidden = (
            ("--resume-checkpoint", "/tmp/model.pt"),
            ("--allow-inexact-resume",),
            ("--external-learned-state", "/tmp/model.pt"),
            ("--no-scratch",),
            ("--score-aligned-loss",),
        )
        for options in forbidden:
            with self.subTest(options=options):
                with self.assertRaises(SystemExit):
                    self.parse(
                        "--fi-acc8-one-step-smoke",
                        "--expected-gpu-uuid",
                        "GPU-abc",
                        *options,
                    )


class SmokeGuardAndStepTests(unittest.TestCase):
    def gpu_runner(self, identity, owners=''):
        def run(command, **kwargs):
            del kwargs
            if any('query-gpu=' in token for token in command):
                output = identity
            else:
                output = owners
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr='')
        return run

    def test_gpu_preflight_accepts_both_names_with_exact_uuid_memory_and_no_owners(self):
        for name in ('GeForce GTX 1080', 'NVIDIA GeForce GTX 1080'):
            with self.subTest(name=name):
                result = preflight_smoke_gpu(
                    0,
                    'GPU-exact',
                    runner=self.gpu_runner(f'0, GPU-exact, {name}, 8192\n'),
                )
                self.assertEqual(result['uuid'], 'GPU-exact')
                self.assertEqual(result['memory_mib'], 8192)

    def test_wrong_or_occupied_gpu_fails_before_any_cuda_call(self):
        bad_cases = (
            ('1, GPU-exact, NVIDIA GeForce GTX 1080, 8192\n', ''),
            ('0, GPU-wrong, NVIDIA GeForce GTX 1080, 8192\n', ''),
            ('0, GPU-exact, Tesla T4, 8192\n', ''),
            ('0, GPU-exact, NVIDIA GeForce GTX 1080, 8119\n', ''),
            ('0, GPU-exact, NVIDIA GeForce GTX 1080, 8192\n', 'GPU-exact, 1234\n'),
        )
        for identity, owners in bad_cases:
            with self.subTest(identity=identity, owners=owners), patch.object(
                torch.cuda, 'is_available', side_effect=AssertionError('CUDA touched')
            ) as cuda_call:
                with self.assertRaises(RuntimeError):
                    preflight_smoke_gpu(
                        0,
                        'GPU-exact',
                        runner=self.gpu_runner(identity, owners),
                    )
                cuda_call.assert_not_called()

    def test_output_collision_is_rejected_without_mutating_existing_output(self):
        with tempfile.TemporaryDirectory(prefix='fi-output-') as tmp:
            output = Path(tmp) / 'smoke'
            output.mkdir()
            marker = output / 'keep.txt'
            marker.write_bytes(b'keep')
            with self.assertRaisesRegex(FileExistsError, 'already exists'):
                prepare_smoke_output(output)
            self.assertEqual(marker.read_bytes(), b'keep')

    def assert_mask_rejected_at_boundary(self, mask, message):
        class ModelMustNotRun(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, kspace, model_mask, crop_size=None):
                del kspace, model_mask, crop_size
                raise AssertionError('model ran before mask rejection')

        model = ModelMustNotRun()
        sample = {
            'kspace': torch.ones(1, 1, 8, 8, 2, dtype=torch.float32),
            'mask': mask,
            'target': torch.ones(1, 8, 8, dtype=torch.float32),
            'maximum': torch.tensor([1.0], dtype=torch.float32),
        }
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=FI_ACC8_RECIPE.lr, weight_decay=0.0
        )
        scheduler = build_fi_scheduler(optimizer)
        with self.assertRaisesRegex(ValueError, message):
            run_one_finite_optimizer_step(
                model,
                sample,
                lambda output, target, maximum: ((output - target) ** 2).mean(),
                optimizer,
                scheduler,
                torch.device('cpu'),
            )

    def test_wrong_shape_stored_mask_is_rejected_at_model_boundary(self):
        self.assert_mask_rejected_at_boundary(
            torch.ones(1, 1, 8, 1, dtype=torch.float32),
            'shape',
        )

    def test_nonfinite_stored_mask_is_rejected_at_model_boundary(self):
        mask = torch.ones(1, 1, 1, 8, 1, dtype=torch.float32)
        mask[..., 3, :] = float('nan')
        self.assert_mask_rejected_at_boundary(mask, 'finite')

    def test_nonbinary_stored_mask_is_rejected_at_model_boundary(self):
        mask = torch.ones(1, 1, 1, 8, 1, dtype=torch.float32)
        mask[..., 3, :] = 0.5
        self.assert_mask_rejected_at_boundary(mask, 'binary')

    def test_stored_float32_binary_mask_becomes_bool_at_model_boundary(self):
        class MaskContractFI(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, kspace, mask, crop_size=None):
                del crop_size
                assert mask.dtype is torch.bool
                assert mask.shape == (1, 1, 1, 8, 1)
                return kspace[:, 0, :, :, 0] * self.scale

        model = MaskContractFI()
        stored_mask = torch.tensor(
            [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            dtype=torch.float32,
        ).reshape(1, 1, 1, 8, 1)
        kspace = torch.ones(1, 1, 8, 8, 2, dtype=torch.float32)
        kspace = kspace * stored_mask
        sample = {
            'kspace': kspace,
            'mask': stored_mask,
            'target': torch.ones(1, 8, 8, dtype=torch.float32),
            'maximum': torch.tensor([1.0], dtype=torch.float32),
        }
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=FI_ACC8_RECIPE.lr, weight_decay=0.0
        )
        scheduler = build_fi_scheduler(optimizer)

        result = run_one_finite_optimizer_step(
            model,
            sample,
            lambda output, target, maximum: ((output - target) ** 2).mean(),
            optimizer,
            scheduler,
            torch.device('cpu'),
        )

        self.assertEqual(result['global_step'], 1)
        applied_mask_sha256 = hashlib.sha256(
            stored_mask.numpy().reshape(8).tobytes(order='C')
        ).hexdigest()
        self.assertEqual(
            result['mask_contract'],
            {
                'stored_dtype': 'float32',
                'stored_shape': [8],
                'model_dtype': 'bool',
                'model_shape': [1, 1, 1, 8, 1],
                'binary': True,
                'applied_to_kspace': True,
                'masked_out_kspace_zero': True,
                'applied_mask_sha256': applied_mask_sha256,
            },
        )

    def test_exactly_one_cpu_step_has_finite_loss_and_gradients(self):
        class TinyFI(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, kspace, mask, crop_size=None):
                del mask, crop_size
                return kspace[..., 0, 0] * self.scale

        model = TinyFI()
        sample = {
            'kspace': torch.ones(1, 8, 8, 1, 2),
            'mask': torch.ones(1, 1, 1, 1, 1),
            'target': torch.ones(1, 8, 8),
            'maximum': torch.tensor([1.0]),
        }
        loss_fn = __import__('fastmri.losses', fromlist=['SSIMLoss']).SSIMLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
        scheduler = build_fi_scheduler(optimizer)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.0)
        optimizer_step = MagicMock(wraps=optimizer.step)
        scheduler_step = MagicMock(wraps=scheduler.step)
        optimizer.step = optimizer_step
        scheduler.step = scheduler_step
        parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        result = run_one_finite_optimizer_step(
            model, sample, loss_fn, optimizer, scheduler, torch.device('cpu')
        )

        changed_parameters = [
            name
            for name, parameter in model.named_parameters()
            if name in parameters_before
            and not torch.equal(parameters_before[name], parameter.detach())
        ]
        self.assertGreater(len(changed_parameters), 0)
        self.assertEqual(result['trainable_parameter_count'], 1)
        self.assertEqual(result['changed_parameter_count'], len(changed_parameters))
        self.assertEqual(result['nominal_step0_multiplier'], 0.0)
        self.assertEqual(result['nominal_step0_lr'], 0.0)
        self.assertEqual(result['smoke_applied_lr'], FI_ACC8_RECIPE.lr)
        self.assertEqual(result['lr_area'], FI_ACC8_RECIPE.lr)
        self.assertAlmostEqual(
            result['post_step_nominal_multiplier'], fi_lr_multiplier(1)
        )
        self.assertAlmostEqual(
            result['post_step_nominal_lr'],
            FI_ACC8_RECIPE.lr * fi_lr_multiplier(1),
        )
        self.assertRegex(result['pre_step_parameter_sha256'], r'^[0-9a-f]{64}$')
        self.assertRegex(result['post_step_parameter_sha256'], r'^[0-9a-f]{64}$')
        self.assertNotEqual(
            result['pre_step_parameter_sha256'],
            result['post_step_parameter_sha256'],
        )
        self.assertTrue(math.isfinite(result['loss']))
        self.assertGreater(result['gradient_parameter_count'], 0)
        self.assertEqual(result['global_step'], 1)
        optimizer_step.assert_called_once_with()
        scheduler_step.assert_called_once_with()
        self.assertIsNotNone(model.scale.grad)
        self.assertTrue(torch.isfinite(model.scale.grad).all())

    def test_unchanged_trainable_parameters_fail_closed_before_scheduler_step(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, kspace, mask, crop_size=None):
                del mask, crop_size
                return kspace[:, 0, :, :, 0] * self.weight

        model = Tiny()
        sample = {
            'kspace': torch.ones(1, 1, 8, 8, 2),
            'mask': torch.ones(1, 1, 1, 8, 1),
            'target': torch.ones(1, 8, 8),
            'maximum': torch.tensor([1.0]),
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=FI_ACC8_RECIPE.lr)
        scheduler = build_fi_scheduler(optimizer)
        with patch.object(optimizer, 'step') as optimizer_step, patch.object(
            scheduler, 'step'
        ) as scheduler_step, self.assertRaisesRegex(
            FloatingPointError, 'changed no trainable model parameters'
        ):
            run_one_finite_optimizer_step(
                model,
                sample,
                lambda output, target, maximum: ((output - target) ** 2).mean(),
                optimizer,
                scheduler,
                torch.device('cpu'),
            )
        optimizer_step.assert_called_once_with()
        scheduler_step.assert_not_called()

    def test_nonfinite_loss_gradient_or_parameter_is_rejected_before_step(self):
        class Tiny(torch.nn.Module):
            def __init__(self, value=1.0):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(value))

            def forward(self, kspace, mask, crop_size=None):
                del mask, crop_size
                return kspace[:, 0, :, :, 0] * self.weight

        sample = {
            'kspace': torch.ones(1, 1, 8, 8, 2),
            'mask': torch.ones(1, 1, 1, 8, 1),
            'target': torch.ones(1, 8, 8),
            'maximum': torch.tensor([1.0]),
        }
        cases = (
            (Tiny(float('inf')), lambda output, target, maximum: output.mean()),
            (Tiny(), lambda output, target, maximum: output.mean() * float('nan')),
            (Tiny(), lambda output, target, maximum: output.mean() * float('inf')),
        )
        for model, loss_fn in cases:
            with self.subTest(loss_fn=loss_fn):
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
                scheduler = build_fi_scheduler(optimizer)
                with patch.object(optimizer, 'step') as step:
                    with self.assertRaisesRegex(FloatingPointError, 'nonfinite'):
                        run_one_finite_optimizer_step(
                            model, sample, loss_fn, optimizer, scheduler,
                            torch.device('cpu'),
                        )
                    step.assert_not_called()


class SmokeCheckpointTests(unittest.TestCase):
    def smoke_step_result(self, **overrides):
        values = {
            'global_step': 1,
            'lr_area': FI_ACC8_RECIPE.lr,
            'loss': 0.5,
            'gradient_parameter_count': 1,
            'trainable_parameter_count': 1,
            'changed_parameter_count': 1,
            'nominal_step0_multiplier': 0.0,
            'nominal_step0_lr': 0.0,
            'smoke_applied_lr': FI_ACC8_RECIPE.lr,
            'post_step_nominal_multiplier': fi_lr_multiplier(1),
            'post_step_nominal_lr': FI_ACC8_RECIPE.lr * fi_lr_multiplier(1),
            'pre_step_parameter_sha256': '1' * 64,
            'post_step_parameter_sha256': '2' * 64,
            'mask_contract': {
                'stored_dtype': 'float32',
                'stored_shape': [1],
                'model_dtype': 'bool',
                'model_shape': [1, 1, 1, 1, 1],
                'binary': True,
                'applied_to_kspace': True,
                'masked_out_kspace_zero': True,
                'applied_mask_sha256': '3' * 64,
            },
        }
        values.update(overrides)
        return values

    def test_checkpoint_rejects_zero_smoke_applied_lr_before_publication(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=FI_ACC8_RECIPE.lr)
        scheduler = build_fi_scheduler(optimizer)
        with tempfile.TemporaryDirectory(prefix='fi-checkpoint-') as tmp:
            output = prepare_smoke_output(Path(tmp) / 'smoke')
            with self.assertRaisesRegex(ValueError, 'finite and positive'):
                save_smoke_checkpoint(
                    output,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step_result=self.smoke_step_result(
                        smoke_applied_lr=0.0,
                        lr_area=0.0,
                    ),
                    sampler_state={'seed': 431},
                    provenance={'source': {}, 'data': {}, 'runtime': {}},
                )
            self.assertFalse((output / 'checkpoint-step-000001.pt').exists())

    def test_checkpoint_rejects_invalid_mask_contract_before_publication(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=FI_ACC8_RECIPE.lr)
        scheduler = build_fi_scheduler(optimizer)
        with tempfile.TemporaryDirectory(prefix='fi-checkpoint-') as tmp:
            output = prepare_smoke_output(Path(tmp) / 'smoke')
            with self.assertRaisesRegex(ValueError, 'mask contract'):
                save_smoke_checkpoint(
                    output,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step_result=self.smoke_step_result(mask_contract=None),
                    sampler_state={'seed': 431},
                    provenance={'source': {}, 'data': {}, 'runtime': {}},
                )
            self.assertFalse((output / 'checkpoint-step-000001.pt').exists())

    def test_checkpoint_round_trips_on_cpu_without_moving_or_mutating_live_objects(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler = build_fi_scheduler(optimizer)
        before_model = copy.deepcopy(model.state_dict())
        before_optimizer = copy.deepcopy(optimizer.state_dict())
        before_devices = [parameter.device for parameter in model.parameters()]
        sampler_state = {
            'algorithm': 'maximum-input-candidate-first',
            'seed': 431,
            'selected_file': 'brain_acc8_sample.h5',
            'selected_slice': 0,
        }
        provenance = {
            'source': {'commit': PINNED_UPSTREAM_COMMIT},
            'data': {'manifest_sha256': 'a' * 64},
            'runtime': {'torch_version': torch.__version__},
        }
        caller_provenance = copy.deepcopy(provenance)
        serialized_provenance = {
            **caller_provenance,
            'activation_checkpointing': dict(
                fi_acc8_training_module.FI_ACTIVATION_CHECKPOINT_CONTRACT
            ),
        }

        with tempfile.TemporaryDirectory(prefix='fi-checkpoint-') as tmp:
            output = prepare_smoke_output(Path(tmp) / 'smoke')
            published = save_smoke_checkpoint(
                output,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step_result=self.smoke_step_result(),
                sampler_state=sampler_state,
                provenance=provenance,
            )
            checkpoint = load_smoke_checkpoint_cpu(published['path'])
            self.assertEqual(oct(published['path'].stat().st_mode & 0o777), '0o444')
            self.assertEqual(len(published['sha256']), 64)

        self.assertEqual(checkpoint['kind'], 'fi-varnet-acc8-training-fit-smoke')
        self.assertFalse(checkpoint['resumable'])
        self.assertFalse(checkpoint['full_training_authorized'])
        self.assertEqual(checkpoint['global_step'], 1)
        self.assertEqual(checkpoint['recipe'], FI_ACC8_RECIPE.as_dict())
        self.assertEqual(checkpoint['sampler'], sampler_state)
        self.assertEqual(checkpoint['provenance'], serialized_provenance)
        self.assertEqual(
            checkpoint['activation_checkpointing'],
            fi_acc8_training_module.FI_ACTIVATION_CHECKPOINT_CONTRACT,
        )
        self.assertEqual(provenance, caller_provenance)
        self.assertNotIn('activation_checkpointing', provenance)
        self.assertIn('optimizer', checkpoint)
        self.assertIn('scheduler', checkpoint)
        self.assertIn('rng', checkpoint)
        self.assertTrue(all(t.device.type == 'cpu' for t in checkpoint['model'].values()))
        self.assertEqual([parameter.device for parameter in model.parameters()], before_devices)
        live_model_state = model.state_dict()
        for key, value in before_model.items():
            self.assertTrue(torch.equal(live_model_state[key], value))
            self.assertNotEqual(checkpoint['model'][key].data_ptr(), live_model_state[key].data_ptr())
        live_optimizer_state = optimizer.state_dict()
        self.assertEqual(live_optimizer_state['param_groups'], before_optimizer['param_groups'])
        for parameter_id, expected_state in before_optimizer['state'].items():
            for key, expected in expected_state.items():
                actual = checkpoint['optimizer']['state'][parameter_id][key]
                if torch.is_tensor(expected):
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(actual.device.type, 'cpu')
                    self.assertNotEqual(
                        actual.data_ptr(),
                        live_optimizer_state['state'][parameter_id][key].data_ptr(),
                    )
                else:
                    self.assertEqual(actual, expected)

    def test_checkpoint_publication_refuses_overwrite_and_loader_rejects_resumption_shape(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = build_fi_scheduler(optimizer)
        kwargs = dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step_result=self.smoke_step_result(),
            sampler_state={'seed': 431},
            provenance={'source': {}, 'data': {}, 'runtime': {}},
        )
        with tempfile.TemporaryDirectory(prefix='fi-checkpoint-') as tmp:
            output = prepare_smoke_output(Path(tmp) / 'smoke')
            published = save_smoke_checkpoint(output, **kwargs)
            original = published['path'].read_bytes()
            with self.assertRaisesRegex(FileExistsError, 'checkpoint'):
                save_smoke_checkpoint(output, **kwargs)
            self.assertEqual(published['path'].read_bytes(), original)
            state = torch.load(published['path'], map_location='cpu', weights_only=True)
            state['resumable'] = True
            forged = output / 'forged.pt'
            torch.save(state, forged)
            with self.assertRaisesRegex(ValueError, 'non-resumable'):
                load_smoke_checkpoint_cpu(forged)

            state['resumable'] = False
            state.pop('activation_checkpointing')
            missing = output / 'missing-checkpoint-evidence.pt'
            torch.save(state, missing)
            with self.assertRaisesRegex(ValueError, 'checkpoint evidence'):
                load_smoke_checkpoint_cpu(missing)

            state['activation_checkpointing'] = dict(
                fi_acc8_training_module.FI_ACTIVATION_CHECKPOINT_CONTRACT
            )
            state['provenance']['activation_checkpointing'] = dict(
                state['activation_checkpointing'], feature_cascades=11
            )
            mismatched = output / 'mismatched-checkpoint-evidence.pt'
            torch.save(state, mismatched)
            with self.assertRaisesRegex(ValueError, 'checkpoint evidence'):
                load_smoke_checkpoint_cpu(mismatched)

    def test_checkpoint_serialization_is_validated_before_atomic_publication(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = build_fi_scheduler(optimizer)
        with tempfile.TemporaryDirectory(prefix='fi-checkpoint-') as tmp:
            output = prepare_smoke_output(Path(tmp) / 'smoke')

            def corrupt_save(value, handle):
                del value
                handle.write(b'not-a-checkpoint')

            with patch(
                'utils.learning.fi_acc8_training.torch.save',
                side_effect=corrupt_save,
            ), self.assertRaisesRegex(ValueError, 'validation'):
                save_smoke_checkpoint(
                    output,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step_result=self.smoke_step_result(),
                    sampler_state={'seed': 431},
                    provenance={'source': {}, 'data': {}, 'runtime': {}},
                )
            self.assertFalse((output / 'checkpoint-step-000001.pt').exists())

    def test_named_fallback_success_is_no_overwrite_and_fsyncs_directory(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = build_fi_scheduler(optimizer)
        kwargs = dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step_result=self.smoke_step_result(),
            sampler_state={'seed': 431},
            provenance={'source': {}, 'data': {}, 'runtime': {}},
        )
        with tempfile.TemporaryDirectory(prefix='fi-checkpoint-') as tmp:
            output = prepare_smoke_output(Path(tmp) / 'smoke')
            real_fsync = os.fsync
            real_unlink = os.unlink
            events = []

            def record_fsync(fd):
                if stat.S_ISDIR(os.fstat(fd).st_mode):
                    events.append('directory-fsync')
                return real_fsync(fd)

            def record_unlink(path, *args, **kwargs):
                events.append('unlink-stage')
                return real_unlink(path, *args, **kwargs)

            with patch(
                'utils.learning.fi_acc8_training._open_anonymous_file',
                side_effect=OSError(errno.ENOTSUP, 'forced named fallback'),
            ), patch(
                'utils.learning.fi_acc8_training.os.fsync',
                side_effect=record_fsync,
            ), patch(
                'utils.learning.fi_acc8_training.os.unlink',
                side_effect=record_unlink,
            ):
                published = save_smoke_checkpoint(output, **kwargs)
            original = published['path'].read_bytes()

            self.assertIn('directory-fsync', events)
            self.assertIn('unlink-stage', events)
            self.assertGreater(
                max(index for index, event in enumerate(events) if event == 'directory-fsync'),
                events.index('unlink-stage'),
            )
            self.assertEqual(list(output.glob('.checkpoint-unpublished-*')), [])
            with patch(
                'utils.learning.fi_acc8_training._open_anonymous_file',
                side_effect=OSError(errno.ENOTSUP, 'forced named fallback'),
            ), self.assertRaisesRegex(FileExistsError, 'checkpoint'):
                save_smoke_checkpoint(output, **kwargs)
            self.assertEqual(published['path'].read_bytes(), original)
            self.assertEqual(len(list(output.glob('.checkpoint-unpublished-*'))), 1)

    def test_named_fallback_validation_failure_cleans_owned_stage(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = build_fi_scheduler(optimizer)
        with tempfile.TemporaryDirectory(prefix='fi-checkpoint-') as tmp:
            output = prepare_smoke_output(Path(tmp) / 'smoke')

            def corrupt_save(value, handle):
                del value
                handle.write(b'not-a-checkpoint')

            with patch(
                'utils.learning.fi_acc8_training._open_anonymous_file',
                side_effect=OSError(errno.ENOTSUP, 'forced named fallback'),
            ), patch(
                'utils.learning.fi_acc8_training.torch.save',
                side_effect=corrupt_save,
            ), self.assertRaisesRegex(ValueError, 'validation'):
                save_smoke_checkpoint(
                    output,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step_result=self.smoke_step_result(),
                    sampler_state={'seed': 431},
                    provenance={'source': {}, 'data': {}, 'runtime': {}},
                )
            self.assertEqual(list(output.glob('.checkpoint-unpublished-*')), [])

    def test_named_fallback_failed_publication_preserves_replaced_stage_path(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = build_fi_scheduler(optimizer)
        with tempfile.TemporaryDirectory(prefix='fi-checkpoint-') as tmp:
            output = prepare_smoke_output(Path(tmp) / 'smoke')

            def replace_stage_then_fail(handle, directory_fd, final_name):
                del handle, directory_fd, final_name
                stage = next(output.glob('.checkpoint-unpublished-*'))
                stage.rename(output / '.original-stage-inode')
                stage.write_bytes(b'cooperating-writer-replacement')
                raise OSError(errno.EIO, 'injected publication failure')

            with patch(
                'utils.learning.fi_acc8_training._open_anonymous_file',
                side_effect=OSError(errno.ENOTSUP, 'forced named fallback'),
            ), patch(
                'utils.learning.fi_acc8_training._publish_fd_without_overwrite',
                side_effect=replace_stage_then_fail,
            ), self.assertRaisesRegex(OSError, 'injected publication failure'):
                save_smoke_checkpoint(
                    output,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step_result=self.smoke_step_result(),
                    sampler_state={'seed': 431},
                    provenance={'source': {}, 'data': {}, 'runtime': {}},
                )

            replacement = next(output.glob('.checkpoint-unpublished-*'))
            self.assertEqual(replacement.read_bytes(), b'cooperating-writer-replacement')
            self.assertFalse((output / 'checkpoint-step-000001.pt').exists())


class Acc8DataTests(unittest.TestCase):
    def make_pair(
        self,
        root,
        name='brain_acc8_sample.h5',
        slices=2,
        shape=(2, 8, 6),
        mask_dtype=np.float32,
        mask_shape=None,
    ):
        (root / 'kspace').mkdir(parents=True, exist_ok=True)
        (root / 'image').mkdir(parents=True, exist_ok=True)
        width = shape[-1]
        mask = np.array([0, 1] * ((width + 1) // 2), dtype=mask_dtype)[:width]
        if mask_shape is not None:
            mask = mask.reshape(mask_shape)
        kspace = (
            np.arange(slices * np.prod(shape), dtype=np.float32).reshape((slices, *shape))
            + 1j
        ).astype(np.complex64)
        with h5py.File(root / 'kspace' / name, 'w') as hf:
            hf.create_dataset('kspace', data=kspace)
            hf.create_dataset('mask', data=mask)
        with h5py.File(root / 'image' / name, 'w') as hf:
            hf.create_dataset('image_label', data=np.ones((slices, 8, 6), np.float32))
            hf.attrs['max'] = np.float32(2.0)
        return kspace, mask

    def inspect(self, root, **overrides):
        return inspect_acc8_training_data(
            root,
            input_key='kspace',
            target_key='image_label',
            max_key='max',
            expected_files=overrides.pop('expected_files', 1),
            expected_slices=overrides.pop('expected_slices', 2),
            maximum_input_shape=overrides.pop('maximum_input_shape', (2, 8, 6)),
            **overrides,
        )

    def test_root_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='fi-data-') as tmp:
            base = Path(tmp)
            outside = base / 'outside'
            self.make_pair(outside)
            linked_root = base / 'linked-root'
            linked_root.symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, 'symlink'):
                self.inspect(linked_root)

    def test_subdirectory_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='fi-data-') as tmp:
            base = Path(tmp)
            root = base / 'declared-root'
            outside = base / 'outside'
            self.make_pair(outside)
            root.mkdir()
            (root / 'kspace').symlink_to(outside / 'kspace', target_is_directory=True)
            (root / 'image').symlink_to(outside / 'image', target_is_directory=True)

            with self.assertRaisesRegex(ValueError, 'symlink'):
                self.inspect(root)

    def test_h5_leaf_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='fi-data-') as tmp:
            root = Path(tmp) / 'declared-root'
            self.make_pair(root)
            outside = Path(tmp) / 'outside'
            outside.mkdir()
            for child in ('kspace', 'image'):
                leaf = root / child / 'brain_acc8_sample.h5'
                target = outside / f'{child}.h5'
                leaf.rename(target)
                leaf.symlink_to(target)

            with self.assertRaisesRegex(ValueError, 'symlink'):
                self.inspect(root)

    def test_synthetic_h5_preflight_and_sample_apply_float32_width_mask(self):
        with tempfile.TemporaryDirectory(prefix='fi-data-') as tmp:
            root = Path(tmp)
            kspace, mask = self.make_pair(root)
            manifest = self.inspect(root)
            sample = load_fi_acc8_sample(manifest)

        self.assertEqual(manifest.selected_acc8_count, 1)
        self.assertEqual(manifest.total_entries, 1)
        self.assertEqual(manifest.ignored_acc4_count, 0)
        self.assertEqual(manifest.slice_count, 2)
        self.assertEqual(
            [identity.path for identity in manifest.directory_identities[-2:]],
            [str(root / 'kspace'), str(root / 'image')],
        )
        record = manifest.records[0]
        self.assertGreater(record.kspace_size, 0)
        self.assertGreater(record.image_size, 0)
        self.assertGreater(record.kspace_st_dev, 0)
        self.assertGreater(record.kspace_st_ino, 0)
        self.assertGreater(record.image_st_dev, 0)
        self.assertGreater(record.image_st_ino, 0)
        self.assertEqual(sample['kspace'].shape, (1, 2, 8, 6, 2))
        self.assertEqual(sample['mask'].shape, (1, 1, 1, 6, 1))
        self.assertEqual(sample['mask'].dtype, torch.float32)
        np.testing.assert_array_equal(
            sample['mask'].numpy().reshape(6), mask,
        )
        expected = kspace[manifest.selected_slice] * mask
        np.testing.assert_allclose(
            torch.view_as_complex(sample['kspace']).numpy()[0], expected,
        )
        self.assertEqual(sample['target'].dtype, torch.float32)
        self.assertEqual(sample['maximum'].tolist(), [2.0])

    def test_same_byte_same_size_leaf_replacement_is_rejected_by_inode(self):
        with tempfile.TemporaryDirectory(prefix='fi-data-') as tmp:
            root = Path(tmp)
            self.make_pair(root)
            manifest = self.inspect(root)
            leaf = root / 'kspace' / manifest.selected_file
            original = leaf.read_bytes()
            inventory_inode = leaf.stat().st_ino
            leaf.rename(root / 'kspace' / '.inventory-inode.h5')
            leaf.write_bytes(original)

            self.assertEqual(leaf.stat().st_size, len(original))
            self.assertNotEqual(leaf.stat().st_ino, inventory_inode)
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                load_fi_acc8_sample(manifest)

    def test_mutation_of_held_h5_inode_during_inventory_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='fi-data-') as tmp:
            root = Path(tmp)
            self.make_pair(root)
            leaf = root / 'kspace' / 'brain_acc8_sample.h5'
            pristine = leaf.read_bytes()
            detached = root / 'kspace' / '.held-inventory-inode.h5'
            real_h5_from_fd = fi_acc8_training_module._h5_from_fd
            mutated = False

            @contextmanager
            def mutate_after_h5_inspection(fd):
                nonlocal mutated
                with real_h5_from_fd(fd) as h5_file:
                    is_kspace = 'kspace' in h5_file
                    yield h5_file
                if is_kspace and not mutated:
                    leaf.rename(detached)
                    leaf.write_bytes(pristine)
                    with detached.open('r+b') as handle:
                        handle.seek(-1, os.SEEK_END)
                        byte = handle.read(1)
                        handle.seek(-1, os.SEEK_END)
                        handle.write(bytes([byte[0] ^ 0x01]))
                    mutated = True

            with patch.object(
                fi_acc8_training_module,
                '_h5_from_fd',
                side_effect=mutate_after_h5_inspection,
            ), self.assertRaisesRegex(ValueError, 'bytes changed during preflight'):
                self.inspect(root)
            self.assertEqual(leaf.read_bytes(), pristine)
            self.assertTrue(mutated)

    def test_mutation_of_held_h5_inode_is_rejected_after_path_is_replaced(self):
        with tempfile.TemporaryDirectory(prefix='fi-data-') as tmp:
            root = Path(tmp)
            self.make_pair(root)
            manifest = self.inspect(root)
            leaf = root / 'kspace' / manifest.selected_file
            pristine = leaf.read_bytes()
            detached = root / 'kspace' / '.held-inode.h5'
            real_h5_from_fd = fi_acc8_training_module._h5_from_fd
            mutated = False

            @contextmanager
            def mutate_after_h5_read(fd):
                nonlocal mutated
                with real_h5_from_fd(fd) as h5_file:
                    is_kspace = 'kspace' in h5_file
                    yield h5_file
                if is_kspace and not mutated:
                    leaf.rename(detached)
                    leaf.write_bytes(pristine)
                    with detached.open('r+b') as handle:
                        handle.seek(-1, os.SEEK_END)
                        byte = handle.read(1)
                        handle.seek(-1, os.SEEK_END)
                        handle.write(bytes([byte[0] ^ 0x01]))
                    mutated = True

            with patch.object(
                fi_acc8_training_module,
                '_h5_from_fd',
                side_effect=mutate_after_h5_read,
            ), self.assertRaisesRegex(ValueError, 'bytes changed'):
                load_fi_acc8_sample(manifest)
            self.assertEqual(leaf.read_bytes(), pristine)
            self.assertTrue(mutated)

    def test_selected_h5_bytes_must_not_change_between_preflight_and_load(self):
        with tempfile.TemporaryDirectory(prefix='fi-data-') as tmp:
            root = Path(tmp)
            self.make_pair(root)
            manifest = self.inspect(root)
            with h5py.File(root / 'kspace' / manifest.selected_file, 'r+') as hf:
                hf['kspace'][manifest.selected_slice, 0, 0, 0] += np.complex64(1.0)
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                load_fi_acc8_sample(manifest)

    def test_mixed_acceleration_inventory_selects_only_the_acc8_pair(self):
        with tempfile.TemporaryDirectory(prefix='fi-data-') as tmp:
            root = Path(tmp)
            self.make_pair(root, name='brain_acc8_sample.h5')
            self.make_pair(root, name='brain_acc4_sample.h5')
            with patch.object(
                fi_acc8_training_module,
                '_open_regular_file_nofollow',
                wraps=fi_acc8_training_module._open_regular_file_nofollow,
            ) as regular_opener, patch.object(
                fi_acc8_training_module,
                '_sha256_fd',
                wraps=fi_acc8_training_module._sha256_fd,
            ) as hasher:
                manifest = self.inspect(root)

        self.assertEqual(
            [record.name for record in manifest.records],
            ['brain_acc8_sample.h5'],
        )
        self.assertEqual(manifest.total_entries, 2)
        self.assertEqual(manifest.selected_acc8_count, 1)
        self.assertEqual(manifest.ignored_acc4_count, 1)
        self.assertEqual(
            [record.name for record in manifest.ignored_acc4_records],
            ['brain_acc4_sample.h5'],
        )
        ignored_acc4_identity_bytes = json.dumps(
            [vars(record) for record in manifest.ignored_acc4_records],
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
        self.assertEqual(
            manifest.ignored_acc4_identity_sha256,
            hashlib.sha256(ignored_acc4_identity_bytes).hexdigest(),
        )
        self.assertEqual(
            [call.args[1] for call in regular_opener.call_args_list],
            ['brain_acc8_sample.h5', 'brain_acc8_sample.h5'],
        )
        self.assertTrue(hasher.call_args_list)
        self.assertTrue(
            all('acc4' not in call.args[1] for call in hasher.call_args_list)
        )

    def test_expected_organizer_total_and_acc4_counts_are_enforced(self):
        with tempfile.TemporaryDirectory(prefix='fi-data-') as tmp:
            root = Path(tmp)
            self.make_pair(root, name='brain_acc8_sample.h5')
            self.make_pair(root, name='brain_acc4_sample.h5')

            with self.assertRaisesRegex(ValueError, 'total entry count'):
                self.inspect(
                    root,
                    expected_total_files=170,
                    expected_ignored_acc4_files=1,
                )
            with self.assertRaisesRegex(ValueError, 'ignored acc4 count'):
                self.inspect(
                    root,
                    expected_total_files=2,
                    expected_ignored_acc4_files=85,
                )

    def test_ignored_acc4_symlink_is_rejected_without_opening_payload(self):
        with tempfile.TemporaryDirectory(prefix='fi-data-') as tmp:
            root = Path(tmp)
            self.make_pair(root, name='brain_acc8_sample.h5')
            self.make_pair(root, name='brain_acc4_sample.h5')
            ignored = root / 'kspace' / 'brain_acc4_sample.h5'
            outside = root / 'ignored-acc4-target.h5'
            ignored.rename(outside)
            ignored.symlink_to(outside)

            with self.assertRaisesRegex(ValueError, 'ignored acc4.*symlink'):
                self.inspect(root)

    def test_nonregular_ignored_acc4_entry_is_rejected_without_opening_payload(self):
        with tempfile.TemporaryDirectory(prefix='fi-data-') as tmp:
            root = Path(tmp)
            self.make_pair(root, name='brain_acc8_sample.h5')
            self.make_pair(root, name='brain_acc4_sample.h5')
            for child in ('kspace', 'image'):
                ignored = root / child / 'brain_acc4_sample.h5'
                ignored.unlink()
                ignored.mkdir()

            with self.assertRaisesRegex(ValueError, 'ignored acc4.*regular file'):
                self.inspect(root)

    def test_unknown_or_malformed_organizer_filenames_are_rejected(self):
        names = (
            'brain_sample.h5',
            'brain_acc8_sample.txt',
            'brain__acc8_sample.h5',
            'brain_ACC8_sample.h5',
        )
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix='fi-data-') as tmp:
                root = Path(tmp)
                self.make_pair(root, name=name)
                with self.assertRaisesRegex(ValueError, 'filename contract'):
                    self.inspect(root)

    def test_ambiguous_acceleration_tokens_are_rejected(self):
        names = (
            'brain_acc4_acc8_sample.h5',
            'brain_acc8_acc8_sample.h5',
            'brain_acc4_acc4_sample.h5',
        )
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix='fi-data-') as tmp:
                root = Path(tmp)
                self.make_pair(root, name=name)
                with self.assertRaisesRegex(ValueError, 'ambiguous acceleration'):
                    self.inspect(root)

    def test_image_and_kspace_filename_or_slice_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='fi-data-') as tmp:
            root = Path(tmp)
            self.make_pair(root)
            (root / 'image' / 'brain_acc8_sample.h5').rename(
                root / 'image' / 'other_acc8_sample.h5'
            )
            with self.assertRaisesRegex(ValueError, 'filename sets'):
                self.inspect(root)
        with tempfile.TemporaryDirectory(prefix='fi-data-') as tmp:
            root = Path(tmp)
            self.make_pair(root)
            with h5py.File(root / 'image' / 'brain_acc8_sample.h5', 'r+') as hf:
                del hf['image_label']
                hf.create_dataset('image_label', data=np.ones((1, 8, 6), np.float32))
            with self.assertRaisesRegex(ValueError, 'slice count'):
                self.inspect(root)

    def test_mask_must_be_stored_binary_float32_width_vector(self):
        cases = (
            {'mask_dtype': np.uint8},
            {'mask_shape': (1, 6)},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory(prefix='fi-data-') as tmp:
                root = Path(tmp)
                self.make_pair(root, **kwargs)
                with self.assertRaisesRegex(ValueError, 'mask'):
                    self.inspect(root)
        with tempfile.TemporaryDirectory(prefix='fi-data-') as tmp:
            root = Path(tmp)
            self.make_pair(root)
            with h5py.File(root / 'kspace' / 'brain_acc8_sample.h5', 'r+') as hf:
                hf['mask'][0] = 0.5
            with self.assertRaisesRegex(ValueError, 'binary'):
                self.inspect(root)

    def test_requires_exact_organizer_counts_and_a_maximum_input_slice(self):
        with tempfile.TemporaryDirectory(prefix='fi-data-') as tmp:
            root = Path(tmp)
            self.make_pair(root)
            with self.assertRaisesRegex(ValueError, 'file count'):
                self.inspect(root, expected_files=85)
            with self.assertRaisesRegex(ValueError, 'slice count'):
                self.inspect(root, expected_slices=2315)
            with self.assertRaisesRegex(ValueError, 'maximum input'):
                self.inspect(root, maximum_input_shape=(3, 8, 6))


class ActivationCheckpointAdapterTests(unittest.TestCase):
    def test_enabling_all_fi_cascades_preserves_exact_state_dict(self):
        class FeatureCascade(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(2, 2))

            def forward(self, value):
                return value @ self.weight

        class ImageCascade(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.randn(2))

            def forward(self, value, reference, mask, sensitivity):
                return value + self.bias + reference + mask + sensitivity

        model = torch.nn.Module()
        model.cascades = torch.nn.Sequential(FeatureCascade(), FeatureCascade())
        model.image_cascades = torch.nn.ModuleList(
            [ImageCascade(), ImageCascade(), ImageCascade()]
        )
        before = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

        contract = fi_varnet_adapter_module.enable_fi_activation_checkpointing(model)
        second_contract = fi_varnet_adapter_module.enable_fi_activation_checkpointing(model)
        after = model.state_dict()

        expected_contract = {
            'enabled': True,
            'implementation': 'torch.utils.checkpoint.checkpoint',
            'use_reentrant': False,
            'preserve_rng_state': True,
            'feature_cascades': 2,
            'image_cascades': 3,
            'state_dict_unchanged': True,
        }
        self.assertEqual(contract, expected_contract)
        self.assertEqual(second_contract, expected_contract)
        self.assertEqual(list(after), list(before))
        self.assertEqual(len(after), len(before))
        for key in before:
            self.assertTrue(torch.equal(after[key], before[key]), key)

    def test_nested_namedtuple_cascades_recompute_and_make_exactly_one_update(self):
        class TinyFeatureImage(NamedTuple):
            features: torch.Tensor
            nested: dict

        class FeatureCascade(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.eye(3))
                self.forward_calls = 0

            def forward(self, feature_image):
                self.forward_calls += 1
                features = torch.sin(feature_image.features @ self.weight)
                return feature_image._replace(
                    features=features,
                    nested={'reference': feature_image.nested['reference']},
                )

        class ImageCascade(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.75))
                self.forward_calls = 0

            def forward(self, value, reference, mask, sensitivity):
                self.forward_calls += 1
                return torch.tanh(
                    value * self.scale + reference + mask + sensitivity
                )

        class TinyFI(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cascades = torch.nn.Sequential(
                    FeatureCascade(), FeatureCascade()
                )
                self.image_cascades = torch.nn.ModuleList(
                    [ImageCascade(), ImageCascade()]
                )

            def forward(self, value):
                reference = value.square() * 0.01
                feature_image = TinyFeatureImage(
                    features=value,
                    nested={'reference': reference},
                )
                feature_image = self.cascades(feature_image)
                output = feature_image.features
                mask = torch.full_like(output, 0.02)
                sensitivity = torch.full_like(output, 0.03)
                for cascade in self.image_cascades:
                    output = cascade(
                        output,
                        feature_image.nested['reference'],
                        mask,
                        sensitivity,
                    )
                return output

        model = TinyFI()
        fi_varnet_adapter_module.enable_fi_activation_checkpointing(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
        before = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

        output = model(torch.randn(2, 3, requires_grad=True))
        output.square().mean().backward()
        optimizer.step()

        self.assertTrue(all(module.forward_calls == 2 for module in model.cascades))
        self.assertTrue(
            all(module.forward_calls == 2 for module in model.image_cascades)
        )
        optimizer_steps = [
            float(state['step']) for state in optimizer.state.values()
            if 'step' in state
        ]
        self.assertEqual(len(optimizer_steps), len(list(model.parameters())))
        self.assertEqual(set(optimizer_steps), {1.0})
        self.assertTrue(
            any(
                not torch.equal(before[key], value)
                for key, value in model.state_dict().items()
            )
        )

    def test_eval_mode_with_grad_does_not_checkpoint_or_recompute(self):
        class Cascade(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(2.0))
                self.forward_calls = 0

            def forward(self, value, *extra):
                self.forward_calls += 1
                return value * self.weight + sum(extra)

        class TinyFI(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cascades = torch.nn.Sequential(Cascade())
                self.image_cascades = torch.nn.ModuleList([Cascade()])

            def forward(self, value):
                value = self.cascades(value)
                return self.image_cascades[0](value, value * 0.0)

        model = TinyFI().eval()
        fi_varnet_adapter_module.enable_fi_activation_checkpointing(model)

        model(torch.tensor(1.0, requires_grad=True)).backward()

        self.assertEqual(model.cascades[0].forward_calls, 1)
        self.assertEqual(model.image_cascades[0].forward_calls, 1)

    def test_pinned_fi_varnet_tiny_cpu_forward_backward_supports_both_cascade_abis(self):
        model_class = load_pinned_fi_varnet_class(
            Path('/root/upstream-fastMRI-91f2df47')
        )
        model = model_class(
            num_cascades=1,
            chans=2,
            pools=1,
            sens_chans=2,
            sens_pools=1,
            acceleration=2,
        ).train()
        contract = fi_varnet_adapter_module.enable_fi_activation_checkpointing(model)
        mask = torch.zeros(1, 1, 1, 16, 1, dtype=torch.bool)
        mask[..., 6:10, :] = True
        kspace = torch.randn(1, 2, 16, 16, 2, dtype=torch.float32)
        kspace = kspace * mask

        output = model(kspace, mask, num_low_frequencies=4, crop_size=(16, 16))
        output.square().mean().backward()

        self.assertEqual(tuple(output.shape), (1, 16, 16))
        self.assertTrue(torch.isfinite(output).all())
        gradients = [
            parameter.grad for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(any(torch.count_nonzero(gradient) for gradient in gradients))
        self.assertEqual(contract['feature_cascades'], 1)
        self.assertEqual(contract['image_cascades'], 1)


class ModelFactoryTests(unittest.TestCase):
    def test_source_attestation_pins_official_mask_transform(self):
        provenance = fi_varnet_adapter_module.verify_pinned_upstream_sources()
        self.assertEqual(
            provenance['transforms_sha256'],
            '0eedd9b6762ea720bd8014a8cd0365a022e1e16de96293b609e45fad96fb65c2',
        )

    def test_adapter_loads_only_the_pinned_upstream_fi_varnet(self):
        upstream = Path('/root/upstream-fastMRI-91f2df47')
        cls = load_pinned_fi_varnet_class(upstream)
        self.assertEqual(PINNED_UPSTREAM_COMMIT, '91f2df4711adbb6d643df1810f234e4abcf5881b')
        self.assertEqual(
            PINNED_FEATURE_VARNET_SHA256,
            '810bf9c18b6e81b38bfc7b3732a26e2b87dc146c907a9b8bbc2d63428ea45d99',
        )
        self.assertEqual(cls.__name__, 'FIVarNet')
        self.assertEqual(Path(sys.modules[cls.__module__].__file__).resolve(), (
            upstream / 'fastmri_examples/feature_varnet/feature_varnet.py'
        ).resolve())

    def test_loss_loader_uses_exact_pinned_upstream_source_and_attests_license(self):
        upstream = Path('/root/upstream-fastMRI-91f2df47')
        cls = load_pinned_ssim_loss_class(upstream)
        self.assertEqual(
            PINNED_LOSSES_SHA256,
            '73ebfe3bc2d9c72b04250cc5a8dc35f31b283496a9411ab92fb422eca59f57ad',
        )
        self.assertEqual(
            PINNED_LICENSE_SHA256,
            '52412d7bc7ce4157ea628bbaacb8829e0a9cb3c58f57f99176126bc8cf2bfc85',
        )
        self.assertEqual(cls.__name__, 'SSIMLoss')
        self.assertEqual(
            Path(sys.modules[cls.__module__].__file__).resolve(),
            (upstream / 'fastmri/losses.py').resolve(),
        )

    def test_pinned_upstream_ssim_loss_bytes_execute_on_cpu(self):
        loss_class = load_pinned_ssim_loss_class(Path('/root/upstream-fastMRI-91f2df47'))
        loss_fn = loss_class()
        prediction = torch.zeros(1, 1, 8, 8, dtype=torch.float32, requires_grad=True)
        target = torch.ones(1, 1, 8, 8, dtype=torch.float32)
        loss = loss_fn(prediction, target, torch.tensor([1.0], dtype=torch.float32))
        loss.backward()

        self.assertEqual(loss.numel(), 1)
        self.assertTrue(torch.isfinite(loss))
        gradient = prediction.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertTrue(torch.isfinite(gradient).all())

    def test_adapter_rejects_unpinned_source_before_import(self):
        with tempfile.TemporaryDirectory(prefix='fi-unpinned-') as tmp:
            with self.assertRaisesRegex(RuntimeError, 'pinned upstream commit'):
                load_pinned_fi_varnet_class(Path(tmp))

    def test_factory_passes_exact_architecture_to_thin_adapter(self):
        sentinel = torch.nn.Linear(1, 1)
        args = SimpleNamespace(
            model_family='fi-varnet-acc8', cascade=12, chans=18, pools=4,
            sens_chans=8, sens_pools=4, acceleration=8,
        )
        with patch(
            'utils.learning.train_part.build_pinned_fi_varnet',
            return_value=sentinel,
        ) as adapter:
            actual = build_model(args)
        self.assertIs(actual, sentinel)
        adapter.assert_called_once_with(
            num_cascades=12,
            chans=18,
            pools=4,
            sens_chans=8,
            sens_pools=4,
            acceleration=8,
        )

    def test_existing_varnet_factory_path_is_unchanged(self):
        sentinel = torch.nn.Linear(1, 1)
        args = SimpleNamespace(
            model_family='varnet', cascade=3, chans=9, sens_chans=4,
        )
        with patch('utils.learning.train_part.VarNet', return_value=sentinel) as varnet:
            actual = build_model(args)
        self.assertIs(actual, sentinel)
        varnet.assert_called_once_with(num_cascades=3, chans=9, sens_chans=4)


class TrainDispatchIntegrationTests(unittest.TestCase):
    def test_mocked_train_main_canonicalizes_relative_output_before_publication(self):
        class TinyFI(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, kspace, mask, crop_size=None):
                del mask, crop_size
                return kspace[:, 0, :, :, 0] * self.scale

        sample = {
            'kspace': torch.ones(1, 1, 8, 8, 2),
            'mask': torch.ones(1, 1, 1, 8, 1),
            'target': torch.ones(1, 8, 8),
            'maximum': torch.tensor([1.0]),
            'fname': 'brain_acc8_sample.h5',
            'slice': 0,
        }
        sample_mask_sha256 = hashlib.sha256(
            sample['mask'].numpy().reshape(8).tobytes(order='C')
        ).hexdigest()
        record = SimpleNamespace(
            name='brain_acc8_sample.h5',
            kspace_sha256='1' * 64,
            image_sha256='2' * 64,
            mask_sha256=sample_mask_sha256,
            kspace_shape=(1, 1, 8, 8),
            target_shape=(1, 8, 8),
            slices=1,
            kspace_size=1,
            kspace_st_dev=2,
            kspace_st_ino=3,
            image_size=4,
            image_st_dev=5,
            image_st_ino=6,
        )
        ignored_record = fi_acc8_training_module.IgnoredAcc4FileRecord(
            name='brain_acc4_sample.h5',
            kspace_size=7,
            kspace_st_dev=8,
            kspace_st_ino=9,
            kspace_st_mtime_ns=10,
            kspace_st_ctime_ns=11,
            image_size=12,
            image_st_dev=13,
            image_st_ino=14,
            image_st_mtime_ns=15,
            image_st_ctime_ns=16,
        )
        manifest = SimpleNamespace(
            root=Path('/root/Data/train'),
            directory_identities=(),
            manifest_sha256='4' * 64,
            ignored_acc4_identity_sha256='5' * 64,
            total_entries=170,
            selected_acc8_count=85,
            ignored_acc4_count=85,
            ignored_acc4_records=(ignored_record,),
            slice_count=2315,
            selected_file=record.name,
            selected_slice=0,
            records=(record,),
        )
        args = SimpleNamespace(
            model_family='fi-varnet-acc8', fi_acc8_one_step_smoke=True,
            expected_gpu_uuid='GPU-exact', GPU_NUM=0,
            net_name=Path('LOCAL_FI_ACC8_CKPT_SMOKE_R1'),
            data_path_train=Path('/root/Data/train'), input_key='kspace',
            target_key='image_label', max_key='max', seed=431,
            cascade=12, chans=18, pools=4, sens_chans=8, sens_pools=4,
            acceleration=8, lr=3e-4, weight_decay=0.0,
        )
        events = []

        def source_verifier():
            events.append('source')
            return {
                'upstream_commit': PINNED_UPSTREAM_COMMIT,
                'feature_varnet_sha256': PINNED_FEATURE_VARNET_SHA256,
                'losses_sha256': PINNED_LOSSES_SHA256,
                'license_sha256': PINNED_LICENSE_SHA256,
            }

        activation_checkpointing = {
            'enabled': True,
            'implementation': 'torch.utils.checkpoint.checkpoint',
            'use_reentrant': False,
            'preserve_rng_state': True,
            'feature_cascades': 12,
            'image_cascades': 12,
            'state_dict_unchanged': True,
        }
        caller_cwd = Path.cwd()
        publish_cwd = {}
        real_publish = fi_acc8_training_module._publish_staged_directory_no_replace

        def publish_after_caller_cwd_change(staged, final_dir, description):
            os.chdir(publish_cwd['path'])
            return real_publish(staged, final_dir, description)

        with tempfile.TemporaryDirectory(prefix='fi-dispatch-') as tmp, patch(
            'utils.learning.fi_acc8_training.verify_pinned_upstream_sources',
            side_effect=source_verifier,
            create=True,
        ), patch(
            'utils.learning.fi_acc8_training.preflight_smoke_gpu',
            side_effect=lambda *unused, **kwargs: events.append('gpu') or {
                'index': 0, 'uuid': 'GPU-exact',
                'name': 'NVIDIA GeForce GTX 1080', 'memory_mib': 8192,
                'compute_owners': [],
            },
        ), patch(
            'utils.learning.fi_acc8_training._select_smoke_device',
            return_value=torch.device('cpu'),
            create=True,
        ), patch(
            'utils.learning.fi_acc8_training.build_model',
            side_effect=lambda unused: events.append('model') or TinyFI(),
            create=True,
        ), patch(
            'utils.learning.fi_acc8_training.enable_fi_activation_checkpointing',
            side_effect=lambda unused: events.append('checkpointing')
            or activation_checkpointing,
            create=True,
        ), patch(
            'utils.learning.fi_acc8_training.build_pinned_ssim_loss',
            return_value=lambda output, target, maximum: ((output - target) ** 2).mean(),
            create=True,
        ), patch(
            'utils.learning.fi_acc8_training.inspect_acc8_training_data',
            return_value=manifest,
        ), patch(
            'utils.learning.fi_acc8_training.load_fi_acc8_sample',
            return_value=sample,
        ), patch(
            'utils.learning.fi_acc8_training._publish_staged_directory_no_replace',
            side_effect=publish_after_caller_cwd_change,
        ), patch.object(train, 'train') as legacy_train:
            temporary_root = Path(tmp)
            publish_cwd['path'] = temporary_root / 'changed-caller-cwd'
            publish_cwd['path'].mkdir()
            relative_result_root = Path(
                os.path.relpath(temporary_root / 'relative', caller_cwd)
            )
            try:
                result = train.main(args=args, result_root=relative_result_root)
            finally:
                os.chdir(caller_cwd)
            assert result is not None
            expected_output = (
                temporary_root
                / 'relative'
                / 'LOCAL_FI_ACC8_CKPT_SMOKE_R1'
                / 'fi-acc8-training-fit-smoke'
            )
            self.assertEqual(result['output_dir'], expected_output)
            self.assertTrue(result['output_dir'].is_absolute())
            output_dir_exists = result['output_dir'].is_dir()
            complete_exists = (result['output_dir'] / 'COMPLETE').is_file()
            reconstruction_exists = (
                result['output_dir'] / 'reconstruction-step-000001.pt'
            ).is_file()
            checkpoint_bytes = result['checkpoint_path'].read_bytes()

            with self.assertRaisesRegex(FileExistsError, 'already exists'):
                train.main(args=args, result_root=temporary_root / 'relative')
            self.assertEqual(result['checkpoint_path'].read_bytes(), checkpoint_bytes)

            try:
                absolute_result = train.main(
                    args=args, result_root=temporary_root / 'absolute'
                )
            finally:
                os.chdir(caller_cwd)
            assert absolute_result is not None
            self.assertTrue(absolute_result['output_dir'].is_absolute())
            self.assertEqual(
                {path.name for path in result['output_dir'].iterdir()},
                {path.name for path in absolute_result['output_dir'].iterdir()},
            )
            self.assertEqual(
                result['report']['optimizer_update_semantics'],
                absolute_result['report']['optimizer_update_semantics'],
            )

        self.assertEqual(events[:4], ['source', 'gpu', 'model', 'checkpointing'])
        legacy_train.assert_not_called()
        checkpoint = result['checkpoint']
        self.assertEqual(checkpoint['global_step'], 1)
        optimizer_steps = [
            state['step']
            for state in checkpoint['optimizer']['state'].values()
            if 'step' in state
        ]
        self.assertEqual(len(optimizer_steps), 1)
        self.assertEqual(float(optimizer_steps[0]), 1.0)
        self.assertEqual(checkpoint['scope'], 'SMOKE_ONLY')
        self.assertFalse(checkpoint['resumable'])
        self.assertFalse(checkpoint['nominal_resumable_step'])
        self.assertEqual(checkpoint['activation_checkpointing'], activation_checkpointing)
        self.assertEqual(
            checkpoint['provenance']['activation_checkpointing'],
            activation_checkpointing,
        )
        self.assertEqual(
            checkpoint['optimizer_update_semantics'],
            'SMOKE_ONLY_LR_PRIMED_FINITE_UPDATE_PROBE',
        )
        report = result['report']
        self.assertEqual(report['scope'], 'SMOKE_ONLY')
        self.assertFalse(report['resumable'])
        self.assertFalse(report['nominal_resumable_step'])
        self.assertEqual(report['activation_checkpointing'], activation_checkpointing)
        self.assertEqual(
            report['provenance']['activation_checkpointing'],
            activation_checkpointing,
        )
        self.assertEqual(report['nominal_step0_multiplier'], 0.0)
        self.assertEqual(report['nominal_step0_lr'], 0.0)
        self.assertEqual(report['smoke_applied_lr'], FI_ACC8_RECIPE.lr)
        self.assertAlmostEqual(
            report['post_step_nominal_lr'],
            FI_ACC8_RECIPE.lr * fi_lr_multiplier(1),
        )
        self.assertGreater(report['changed_parameter_count'], 0)
        self.assertRegex(report['pre_step_parameter_sha256'], r'^[0-9a-f]{64}$')
        self.assertRegex(report['post_step_parameter_sha256'], r'^[0-9a-f]{64}$')
        self.assertNotEqual(
            report['pre_step_parameter_sha256'],
            report['post_step_parameter_sha256'],
        )
        expected_mask_contract = {
            'stored_dtype': 'float32',
            'stored_shape': [8],
            'model_dtype': 'bool',
            'model_shape': [1, 1, 1, 8, 1],
            'binary': True,
            'applied_to_kspace': True,
            'masked_out_kspace_zero': True,
            'applied_mask_sha256': sample_mask_sha256,
        }
        self.assertEqual(report['mask_contract'], expected_mask_contract)
        self.assertEqual(
            checkpoint['smoke_result']['mask_contract'], expected_mask_contract
        )
        expected_data_provenance = {
            'manifest_sha256': '4' * 64,
            'organizer_total_entries': 170,
            'selected_acc8_count': 85,
            'ignored_acc4_count': 85,
            'ignored_acc4_identity_sha256': '5' * 64,
            'ignored_acc4_access': {
                'method': 'nofollow-stat-only',
                'payload_opened': False,
                'payload_hashed': False,
                'h5_read': False,
            },
            'mask_contract': expected_mask_contract,
            'slice_count': 2315,
            'selected_file': record.name,
            'selected_slice': 0,
            'root': '/root/Data/train',
            'directories': [],
            'ignored_acc4_files': [vars(ignored_record)],
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
            ],
        }
        self.assertEqual(report['provenance']['data'], expected_data_provenance)
        self.assertEqual(checkpoint['provenance']['data'], expected_data_provenance)
        scheduler_provenance = report['provenance']['scheduler']
        self.assertEqual(scheduler_provenance['ramp_steps'], 3704)
        self.assertEqual(scheduler_provenance['cosine_decay_start'], 46300)
        self.assertEqual(scheduler_provenance['max_steps'], 92600)
        self.assertEqual(
            scheduler_provenance['smoke_only_lr_priming'],
            {
                'applied_lr': 3e-4,
                'nominal_schedule_definition_modified': False,
                'optimizer_steps': 1,
                'purpose': 'finite-update-feasibility-probe',
                'resumable': False,
            },
        )
        self.assertTrue(output_dir_exists)
        self.assertTrue(complete_exists)
        self.assertTrue(reconstruction_exists)


if __name__ == "__main__":
    unittest.main()

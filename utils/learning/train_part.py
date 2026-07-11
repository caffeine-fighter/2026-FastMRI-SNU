import fcntl
import json
import uuid
import numpy as np
import torch
import time
from pathlib import Path

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.learning.resume import (
    CHECKPOINT_MANIFEST_FORMAT_VERSION,
    CHECKPOINT_MANIFEST_NAME,
    _load_checkpoint_from_handle,
    _open_anonymous_file,
    _open_checkpoint_directory,
    _open_manifest_artifact,
    _open_regular_at,
    _publish_fd_without_overwrite,
    _read_checkpoint_manifest,
    _replace_alias_from_artifact,
    _replace_external_alias_from_artifact,
    _replace_from_open_descriptor,
    _validate_val_loss_history,
    build_training_state,
    load_training_state,
    load_val_loss_history,
    preserve_best_checkpoint,
    validate_checkpoint_pair,
    validate_training_checkpoint,
)
from utils.model.varnet import VarNet

import os

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    device = next(model.parameters()).device
    non_blocking = device.type == "cuda"
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _ = data
        mask = mask.to(device=device, non_blocking=non_blocking)
        kspace = kspace.to(device=device, non_blocking=non_blocking)
        target = target.to(device=device, non_blocking=non_blocking)
        maximum = maximum.to(device=device, non_blocking=non_blocking)

        output = model(kspace, mask)
        loss = loss_type(output, target, maximum)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    device = next(model.parameters()).device
    non_blocking = device.type == "cuda"
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices = data
            kspace = kspace.to(device=device, non_blocking=non_blocking)
            mask = mask.to(device=device, non_blocking=non_blocking)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def _publish_stable_alias(exp_dir, directory_fd, artifact_name, alias_name):
    _replace_alias_from_artifact(
        exp_dir, directory_fd, artifact_name, alias_name
    )


def _publish_checkpoint_manifest(exp_dir, directory_fd, manifest):
    with _open_anonymous_file(directory_fd, "w+") as handle:
        json.dump(manifest, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        _replace_from_open_descriptor(
            handle,
            exp_dir,
            directory_fd,
            CHECKPOINT_MANIFEST_NAME,
            ".checkpoint-manifest-",
        )


def save_val_loss_history(history_path, history):
    """Atomically refresh the non-authoritative validation-history alias."""
    history_path = Path(history_path)
    directory_fd = _open_checkpoint_directory(history_path.parent)
    try:
        with _open_anonymous_file(directory_fd) as handle:
            np.save(handle, history)
            handle.flush()
            os.fsync(handle.fileno())
            _replace_from_open_descriptor(
                handle,
                history_path.parent,
                directory_fd,
                history_path.name,
                ".val-loss-alias-",
            )
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _publish_history_alias(exp_dir, directory_fd, artifact_name, history_path):
    del exp_dir
    _replace_external_alias_from_artifact(
        directory_fd, artifact_name, history_path
    )


def save_model(
    exp_dir,
    epoch,
    model,
    optimizer,
    best_val_loss,
    is_new_best,
    val_loss_history=None,
    history_path=None,
):
    """Atomically commit immutable checkpoint artifacts through one manifest."""
    exp_dir = Path(exp_dir)
    state = build_training_state(epoch, model, optimizer, best_val_loss)
    if val_loss_history is None:
        history = None
    else:
        history = _validate_val_loss_history(
            np.asarray(val_loss_history), epoch, "checkpoint generation"
        )
    if history_path is not None and history is None:
        raise ValueError("history_path requires val_loss_history")
    generation = uuid.uuid4().hex
    artifact_name = f".checkpoint-generation-{generation}-model.pt"
    history_artifact = (
        f".checkpoint-generation-{generation}-history.npy"
        if history is not None
        else None
    )
    directory_fd = _open_checkpoint_directory(exp_dir)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        previous = _read_checkpoint_manifest(exp_dir, directory_fd)
        if previous is not None:
            with _open_manifest_artifact(
                directory_fd, previous, "model"
            ) as previous_model_handle:
                previous_state = _load_checkpoint_from_handle(
                    previous_model_handle
                )
            validate_training_checkpoint(previous_state)
            if epoch <= previous_state["epoch"]:
                raise ValueError(
                    f"Checkpoint commit requires a newer epoch: got {epoch}, "
                    f"current {previous_state['epoch']}"
                )
        with _open_anonymous_file(directory_fd) as model_temporary:
            torch.save(state, model_temporary)
            model_temporary.flush()
            os.fsync(model_temporary.fileno())
            _publish_fd_without_overwrite(
                model_temporary, directory_fd, artifact_name
            )
        os.fsync(directory_fd)
        if history_artifact is not None:
            with _open_anonymous_file(directory_fd) as history_temporary:
                np.save(history_temporary, history)
                history_temporary.flush()
                os.fsync(history_temporary.fileno())
                _publish_fd_without_overwrite(
                    history_temporary, directory_fd, history_artifact
                )
            os.fsync(directory_fd)

        if is_new_best:
            best_artifact = artifact_name
        elif previous is not None:
            # Once a manifest exists, even a null best is authoritative. A
            # writable compatibility alias must never resurrect old state.
            best_artifact = previous["best"]
        else:
            try:
                legacy_best_handle = _open_regular_at(
                    directory_fd, "best_model.pt", "legacy best checkpoint"
                )
            except FileNotFoundError:
                best_artifact = None
            else:
                with legacy_best_handle:
                    retained_best_state = _load_checkpoint_from_handle(
                        legacy_best_handle
                    )
                validate_checkpoint_pair(state, retained_best_state)
                best_artifact = (
                    f".checkpoint-generation-{generation}-best.pt"
                )
                with _open_anonymous_file(directory_fd) as best_temporary:
                    torch.save(retained_best_state, best_temporary)
                    best_temporary.flush()
                    os.fsync(best_temporary.fileno())
                    _publish_fd_without_overwrite(
                        best_temporary, directory_fd, best_artifact
                    )
                os.fsync(directory_fd)

        if best_artifact is not None:
            with _open_manifest_artifact(
                directory_fd, {"best": best_artifact}, "best"
            ) as retained_best_handle:
                retained_best_state = _load_checkpoint_from_handle(
                    retained_best_handle
                )
            validate_checkpoint_pair(state, retained_best_state)
        manifest = {
            "format_version": CHECKPOINT_MANIFEST_FORMAT_VERSION,
            "generation": generation,
            "model": artifact_name,
            "best": best_artifact,
        }
        if history_artifact is not None:
            manifest["history"] = history_artifact
        _publish_checkpoint_manifest(exp_dir, directory_fd, manifest)
        os.fsync(directory_fd)

        # The manifest is the atomic commit. Stable paths are compatibility
        # aliases and can be reconstructed from it after any interruption.
        if history_artifact is not None and history_path is not None:
            _publish_history_alias(
                exp_dir,
                directory_fd,
                history_artifact,
                Path(history_path),
            )
        if best_artifact is not None:
            _publish_stable_alias(
                exp_dir, directory_fd, best_artifact, "best_model.pt"
            )
        _publish_stable_alias(
            exp_dir, directory_fd, artifact_name, "model.pt"
        )
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

        
def train(args):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.GPU_NUM}')
        torch.cuda.set_device(device)
        print('Current cuda device: ', torch.cuda.current_device())
    else:
        device = torch.device('cpu')
        print('Current device: cpu')

    model = VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans)
    model.to(device=device)

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    if args.resume_checkpoint is None:
        best_val_loss = 1.
        start_epoch = 0
        val_loss_log = np.empty((0, 2))
    else:
        start_epoch, best_val_loss = load_training_state(
            args.resume_checkpoint,
            model,
            optimizer,
            device,
            allow_inexact=args.allow_inexact_resume,
            learning_rate_override=args.resume_lr,
        )
        if start_epoch >= args.num_epochs:
            raise ValueError(
                f"Checkpoint epoch {start_epoch} must be less than "
                f"the requested total epochs {args.num_epochs}"
            )
        val_loss_log = load_val_loss_history(args.resume_checkpoint, start_epoch)
        preserve_best_checkpoint(args.resume_checkpoint, args.exp_dir)
        print(
            f"Resumed training from {args.resume_checkpoint} at epoch "
            f"{start_epoch}/{args.num_epochs}"
        )

    
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = Path(args.val_loss_dir) / "val_loss_log.npy"

        train_loss = torch.tensor(train_loss, device=device)
        val_loss = torch.tensor(val_loss, device=device)
        num_subjects = torch.tensor(num_subjects, device=device)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(
            args.exp_dir,
            epoch + 1,
            model,
            optimizer,
            best_val_loss,
            is_new_best,
            val_loss_history=val_loss_log,
            history_path=file_path,
        )
        print(f"loss file saved! {file_path}")
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )

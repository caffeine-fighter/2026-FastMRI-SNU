import ctypes
import errno
import fcntl
import hashlib
import json
import stat
import uuid
from dataclasses import dataclass
import numpy as np
import torch
import time
from pathlib import Path
from typing import Optional

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

        
def _validate_retained_reconstruction_names(reconstructions):
    for fname in reconstructions:
        if (
            not isinstance(fname, str)
            or not fname
            or Path(fname).name != fname
            or not fname.endswith(".h5")
        ):
            raise ValueError(
                "Retained reconstruction filename must be a regular .h5 "
                f"basename: {fname!r}"
            )


class PublicationIndeterminateError(OSError):
    """Publication renamed successfully but its directory durability is unknown."""


@dataclass
class _StagedDirectory:
    path: Path
    parent_fd: int
    directory_fd: int
    identity: tuple
    sealed_digest: Optional[str] = None
    published: bool = False
    closed: bool = False


def _inode_identity(file_stat):
    return (file_stat.st_dev, file_stat.st_ino)


def _directory_open_flags():
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _regular_open_flags():
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _same_open_inode(directory_fd, name, expected_identity, expected_kind):
    try:
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return expected_kind(current.st_mode) and _inode_identity(current) == expected_identity


def _fsync_directory_tree(directory_fd):
    """Fsync a symlink-free regular-file tree from its leaves to its root."""
    with os.scandir(directory_fd) as entries:
        names = sorted(entry.name for entry in entries)
    for name in names:
        entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        entry_identity = _inode_identity(entry_stat)
        if stat.S_ISREG(entry_stat.st_mode):
            file_fd = os.open(name, _regular_open_flags(), dir_fd=directory_fd)
            try:
                opened = os.fstat(file_fd)
                if not stat.S_ISREG(opened.st_mode) or _inode_identity(opened) != entry_identity:
                    raise ValueError(f"Staged regular file changed identity: {name}")
                os.fsync(file_fd)
            finally:
                os.close(file_fd)
        elif stat.S_ISDIR(entry_stat.st_mode):
            child_fd = os.open(name, _directory_open_flags(), dir_fd=directory_fd)
            try:
                opened = os.fstat(child_fd)
                if _inode_identity(opened) != entry_identity:
                    raise ValueError(f"Staged directory changed identity: {name}")
                _fsync_directory_tree(child_fd)
            finally:
                os.close(child_fd)
        else:
            raise ValueError(f"Staged output is not a regular file or directory: {name}")
    os.fsync(directory_fd)


def _seal_read_only_tree(directory_fd):
    """Remove write bits from every regular file and directory in the tree."""
    with os.scandir(directory_fd) as entries:
        names = sorted(entry.name for entry in entries)
    for name in names:
        entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        entry_identity = _inode_identity(entry_stat)
        if stat.S_ISREG(entry_stat.st_mode):
            file_fd = os.open(name, _regular_open_flags(), dir_fd=directory_fd)
            try:
                if _inode_identity(os.fstat(file_fd)) != entry_identity:
                    raise ValueError(f"Staged regular file changed identity: {name}")
                os.fchmod(file_fd, 0o444)
            finally:
                os.close(file_fd)
        elif stat.S_ISDIR(entry_stat.st_mode):
            child_fd = os.open(name, _directory_open_flags(), dir_fd=directory_fd)
            try:
                if _inode_identity(os.fstat(child_fd)) != entry_identity:
                    raise ValueError(f"Staged directory changed identity: {name}")
                _seal_read_only_tree(child_fd)
            finally:
                os.close(child_fd)
        else:
            raise ValueError(f"Staged output is not a regular file or directory: {name}")
    os.fchmod(directory_fd, 0o555)


def _tree_digest(directory_fd):
    """Return an exact names/types/modes/content digest of a staged tree."""
    digest = hashlib.sha256()

    def add_field(value):
        value = bytes(value)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    root_stat = os.fstat(directory_fd)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("Staged root is not a directory")
    add_field(b".")
    add_field(stat.S_IMODE(root_stat.st_mode).to_bytes(4, "big"))
    add_field(b"directory")

    def visit(current_fd, relative_parts):
        with os.scandir(current_fd) as entries:
            names_before = sorted(entry.name for entry in entries)
        for name in names_before:
            entry_stat = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
            identity = _inode_identity(entry_stat)
            relative = b"/".join(
                [*(os.fsencode(part) for part in relative_parts), os.fsencode(name)]
            )
            add_field(relative)
            add_field(stat.S_IMODE(entry_stat.st_mode).to_bytes(4, "big"))
            if stat.S_ISREG(entry_stat.st_mode):
                add_field(b"file")
                file_fd = os.open(name, _regular_open_flags(), dir_fd=current_fd)
                try:
                    before = os.fstat(file_fd)
                    before_snapshot = (
                        _inode_identity(before), before.st_mode, before.st_size,
                        before.st_mtime_ns, before.st_ctime_ns,
                    )
                    if not stat.S_ISREG(before.st_mode) or before_snapshot[0] != identity:
                        raise ValueError(f"Staged regular file changed identity: {name}")
                    while True:
                        chunk = os.read(file_fd, 1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                    after = os.fstat(file_fd)
                    after_snapshot = (
                        _inode_identity(after), after.st_mode, after.st_size,
                        after.st_mtime_ns, after.st_ctime_ns,
                    )
                    if after_snapshot != before_snapshot:
                        raise ValueError(f"Staged regular file mutated while validating: {name}")
                finally:
                    os.close(file_fd)
            elif stat.S_ISDIR(entry_stat.st_mode):
                add_field(b"directory")
                child_fd = os.open(name, _directory_open_flags(), dir_fd=current_fd)
                try:
                    if _inode_identity(os.fstat(child_fd)) != identity:
                        raise ValueError(f"Staged directory changed identity: {name}")
                    visit(child_fd, (*relative_parts, name))
                finally:
                    os.close(child_fd)
            else:
                raise ValueError(
                    f"Staged output is not a regular file or directory: {name}"
                )
        with os.scandir(current_fd) as entries:
            names_after = sorted(entry.name for entry in entries)
        if names_after != names_before:
            raise ValueError("Staged directory entries mutated while validating")

    visit(directory_fd, ())
    return digest.hexdigest()


def _seal_staged_directory(staged):
    """Seal, durably flush, and record the exact pre-publication tree."""
    if staged.closed or staged.published:
        raise ValueError("Staging handle is no longer sealable")
    _seal_read_only_tree(staged.directory_fd)
    _fsync_directory_tree(staged.directory_fd)
    staged.sealed_digest = _tree_digest(staged.directory_fd)


def _close_staged_directory(staged):
    if staged.closed:
        return
    staged.closed = True
    os.close(staged.directory_fd)
    os.close(staged.parent_fd)


def _cleanup_staged_directory(staged):
    """Close staging handles without deleting any potentially replaced pathname.

    A failed operation leaves at most its one clearly named ``*-orphan-*`` tree.
    Deliberately preserving that bounded orphan is safer than a check-then-delete
    sequence, which could unlink a cooperating writer's pathname replacement.
    """
    if staged.closed:
        return
    _close_staged_directory(staged)


def _open_durably_created_directory(path, description):
    """Open an absolute directory, durably creating every missing component."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    current_fd = os.open(absolute.anchor, _directory_open_flags())
    current_path = Path(absolute.anchor)
    try:
        for component in absolute.parts[1:]:
            try:
                component_stat = os.stat(
                    component, dir_fd=current_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                os.mkdir(component, mode=0o755, dir_fd=current_fd)
                try:
                    os.fsync(current_fd)
                except BaseException as exc:
                    raise OSError(
                        f"{description} parent creation is not durably confirmed: "
                        f"{current_path / component}"
                    ) from exc
                component_stat = os.stat(
                    component, dir_fd=current_fd, follow_symlinks=False
                )
            if not stat.S_ISDIR(component_stat.st_mode):
                raise NotADirectoryError(
                    errno.ENOTDIR,
                    f"{description} parent component is not a real directory: "
                    f"{current_path / component}",
                    current_path / component,
                )
            next_fd = os.open(component, _directory_open_flags(), dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
            current_path /= component
        return absolute, current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _create_staged_directory(final_dir, description):
    """Create and retain descriptor ownership of a same-parent staging inode."""
    final_dir = Path(os.path.abspath(os.fspath(final_dir)))
    parent = final_dir.parent
    parent, parent_fd = _open_durably_created_directory(parent, description)
    fcntl.flock(parent_fd, fcntl.LOCK_EX)
    try:
        try:
            os.stat(final_dir.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                errno.EEXIST,
                f"{description} output already exists: {final_dir}",
                final_dir,
            )
        staging_name = f".{final_dir.name}-unpublished-orphan-{uuid.uuid4().hex}"
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        try:
            os.fsync(parent_fd)
        except BaseException as exc:
            raise OSError(
                f"{description} staging creation is not durably confirmed: "
                f"{parent / staging_name}"
            ) from exc
        directory_fd = os.open(
            staging_name, _directory_open_flags(), dir_fd=parent_fd
        )
    except BaseException:
        os.close(parent_fd)
        raise
    return _StagedDirectory(
        path=parent / staging_name,
        parent_fd=parent_fd,
        directory_fd=directory_fd,
        identity=_inode_identity(os.fstat(directory_fd)),
    )


def _staged_directory_descriptor_path(staged):
    if staged.closed:
        raise ValueError("Staging handle is closed")
    return Path(f"/proc/self/fd/{staged.directory_fd}")


def _stage_retained_reconstructions(reconstructions, retained_root, epoch):
    _validate_retained_reconstruction_names(reconstructions)
    final_dir = Path(retained_root) / f"epoch_{epoch:04d}"
    staged = _create_staged_directory(final_dir, "Retained epoch")
    try:
        save_reconstructions(
            reconstructions, _staged_directory_descriptor_path(staged)
        )
        with os.scandir(staged.directory_fd) as entries:
            actual_names = {entry.name for entry in entries}
        expected_names = set(reconstructions)
        if actual_names != expected_names:
            raise ValueError(
                "Retained reconstruction staging coverage mismatch: "
                f"missing={sorted(expected_names - actual_names)}; "
                f"unexpected={sorted(actual_names - expected_names)}"
            )
        _seal_staged_directory(staged)
        return staged, final_dir
    except BaseException:
        _cleanup_staged_directory(staged)
        raise


def _publish_staged_directory_no_replace(staged, final_dir, description):
    """Publish a sealed tree for trusted, cooperating local writers.

    Descriptor identity, read-only sealing, and a final exact digest reject
    accidental/concurrent mutation. Python cannot exclude an arbitrary same-UID
    process that deliberately mutates the inode after the final digest and before
    ``renameat2``; callers must enforce a cooperating-writer boundary.
    """
    final_dir = Path(final_dir)
    if staged.closed or staged.published:
        raise ValueError(f"{description} staging handle is no longer publishable")
    if staged.path.parent != final_dir.parent:
        raise ValueError(f"{description} staging and destination must share a parent")
    parent_identity = _inode_identity(os.fstat(staged.parent_fd))
    try:
        named_parent = os.stat(final_dir.parent, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError(f"{description} parent changed identity") from exc
    if not stat.S_ISDIR(named_parent.st_mode) or _inode_identity(named_parent) != parent_identity:
        raise ValueError(f"{description} parent changed identity")
    if not _same_open_inode(
        staged.parent_fd, staged.path.name, staged.identity, stat.S_ISDIR
    ):
        raise ValueError(f"{description} staging path changed identity")
    if staged.sealed_digest is None:
        raise ValueError(f"{description} staging tree was not sealed")
    if _tree_digest(staged.directory_fd) != staged.sealed_digest:
        raise ValueError(f"{description} staging tree changed after sealing")

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(
            errno.ENOTSUP,
            f"Atomic no-overwrite {description.lower()} publication unavailable",
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(
        staged.parent_fd,
        os.fsencode(staged.path.name),
        staged.parent_fd,
        os.fsencode(final_dir.name),
        1,  # RENAME_NOREPLACE
    ) != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(
                error_number,
                f"{description} output already exists: {final_dir}",
                final_dir,
            )
        raise OSError(error_number, os.strerror(error_number), final_dir)

    staged.published = True
    staged.path = final_dir
    if not _same_open_inode(
        staged.parent_fd, final_dir.name, staged.identity, stat.S_ISDIR
    ):
        _close_staged_directory(staged)
        raise PublicationIndeterminateError(
            f"{description} publication committed with indeterminate identity: {final_dir}"
        )
    try:
        os.fsync(staged.parent_fd)
    except BaseException as exc:
        _close_staged_directory(staged)
        raise PublicationIndeterminateError(
            f"{description} publication committed but parent fsync failed: {final_dir}"
        ) from exc
    _close_staged_directory(staged)


def _publish_retained_epoch(staged, final_dir):
    _publish_staged_directory_no_replace(staged, final_dir, "Retained epoch")


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
        staged_epoch_dir = retained_epoch_dir = None
        if getattr(args, "retain_val_epochs", False):
            staged_epoch_dir, retained_epoch_dir = (
                _stage_retained_reconstructions(
                    reconstructions, args.val_epochs_dir, epoch + 1
                )
            )
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = Path(args.val_loss_dir) / "val_loss_log.npy"

        train_loss = torch.tensor(train_loss, device=device)
        val_loss = torch.tensor(val_loss, device=device)
        num_subjects = torch.tensor(num_subjects, device=device)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        try:
            if staged_epoch_dir is not None:
                _publish_retained_epoch(staged_epoch_dir, retained_epoch_dir)
                staged_epoch_dir = None
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
        finally:
            if staged_epoch_dir is not None:
                _cleanup_staged_directory(staged_epoch_dir)
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

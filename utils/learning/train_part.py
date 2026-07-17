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
from utils.common.metrics import ScoreAlignedLoss
from utils.learning.resume import (
    CHECKPOINT_MANIFEST_FORMAT_VERSION,
    CHECKPOINT_MANIFEST_NAME,
    _load_checkpoint_from_handle,
    _manifest_artifact_sha256,
    _open_anonymous_file,
    _open_checkpoint_directory,
    _open_manifest_artifact,
    _open_regular_at,
    _publish_fd_without_overwrite,
    _read_checkpoint_manifest,
    _replace_alias_from_artifact,
    _replace_external_alias_from_artifact,
    _replace_from_open_descriptor,
    _validate_checkpoint_manifest_payload,
    _validate_retention_policy_transition,
    _validate_val_loss_history,
    build_training_state,
    load_training_state,
    load_val_loss_history,
    preserve_best_checkpoint,
    recover_checkpoint_publication,
    validate_checkpoint_pair,
    validate_training_checkpoint,
)
from utils.model.varnet import VarNet

import os

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type, scaler=None):
    model.train()
    device = next(model.parameters()).device
    non_blocking = device.type == "cuda"
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        if getattr(args, "score_aligned_loss", False):
            mask, kspace, target, maximum, _, _, score_metadata = data
        else:
            mask, kspace, target, maximum, _, _ = data
        mask = mask.to(device=device, non_blocking=non_blocking)
        kspace = kspace.to(device=device, non_blocking=non_blocking)
        target = target.to(device=device, non_blocking=non_blocking)
        maximum = maximum.to(device=device, non_blocking=non_blocking)

        is_promptmr = getattr(args, "model_family", "varnet") == "promptmr_plus"
        if is_promptmr:
            optimizer.zero_grad()
        output = model(kspace, mask)
        if is_promptmr:
            from utils.promptmr.data import align_promptmr_output_target

            output, target = align_promptmr_output_target(output, target)
        if getattr(args, "score_aligned_loss", False):
            loss = loss_type(output, target, maximum, score_metadata)
        else:
            loss = loss_type(output, target, maximum)
        if is_promptmr:
            if scaler is None:
                raise RuntimeError("PromptMR+ training requires an AMP scaler state")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), getattr(args, "gradient_clip_val", 0.01)
            )
            scaler.step(optimizer)
            scaler.update()
        else:
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
            if getattr(args, "model_family", "varnet") == "promptmr_plus":
                from utils.promptmr.data import align_promptmr_output_target

                target = target.to(device=device, non_blocking=non_blocking)
                output, target = align_promptmr_output_target(output, target)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].cpu().numpy()

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


def _sha256_open_handle(handle):
    position = handle.tell()
    handle.seek(0)
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    handle.seek(position)
    return digest.hexdigest()


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
    scheduler=None,
    scaler=None,
    model_contract=None,
    generation=None,
    retained_epoch_record=None,
    staged_retained=None,
    retained_destination=None,
):
    """Atomically commit immutable checkpoint artifacts through one manifest."""
    exp_dir = Path(exp_dir)
    state = build_training_state(
        epoch,
        model,
        optimizer,
        best_val_loss,
        scheduler=scheduler,
        scaler=scaler,
        model_contract=model_contract,
    )
    if val_loss_history is None:
        history = None
    else:
        history = _validate_val_loss_history(
            np.asarray(val_loss_history), epoch, "checkpoint generation"
        )
    if history_path is not None and history is None:
        raise ValueError("history_path requires val_loss_history")
    generation = generation or uuid.uuid4().hex
    if (
        not isinstance(generation, str)
        or len(generation) != 32
        or any(character not in "0123456789abcdef" for character in generation)
    ):
        raise ValueError(
            "Checkpoint generation must be 32 lowercase hexadecimal characters"
        )
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
                    previous_model_handle,
                    _manifest_artifact_sha256(previous, "model"),
                )
            validate_training_checkpoint(previous_state)
            if epoch <= previous_state["epoch"]:
                raise ValueError(
                    f"Checkpoint commit requires a newer epoch: got {epoch}, "
                    f"current {previous_state['epoch']}"
                )
        _validate_retention_policy_transition(previous, retained_epoch_record)
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
                expected_best_sha256 = (
                    _manifest_artifact_sha256(previous, "best")
                    if previous is not None and best_artifact == previous["best"]
                    else None
                )
                retained_best_state = _load_checkpoint_from_handle(
                    retained_best_handle, expected_best_sha256
                )
            validate_checkpoint_pair(state, retained_best_state)
        manifest = {
            "format_version": CHECKPOINT_MANIFEST_FORMAT_VERSION,
            "generation": generation,
            "model": artifact_name,
            "best": best_artifact,
        }
        retained_epochs = [] if previous is None else list(
            previous.get("retained_epochs", [])
        )
        if retained_epoch_record is not None:
            record = dict(retained_epoch_record)
            if record.get("generation") != generation or record.get("epoch") != epoch:
                raise ValueError(
                    "Retained epoch provenance does not match checkpoint generation"
                )
            if any(
                item.get("epoch") == epoch
                or item.get("generation") == generation
                for item in retained_epochs
            ):
                raise ValueError("Duplicate retained epoch generation")
            retained_epochs.append(record)
        if retained_epochs:
            manifest["retained_epochs"] = retained_epochs
        if history_artifact is not None:
            manifest["history"] = history_artifact
        artifact_hashes = {}
        for name in {artifact_name, history_artifact, best_artifact} - {None}:
            with _open_regular_at(
                directory_fd, name, "checkpoint manifest artifact"
            ) as artifact_handle:
                artifact_hashes[name] = _sha256_open_handle(artifact_handle)
        manifest["artifacts"] = artifact_hashes

        if (staged_retained is None) != (retained_destination is None):
            raise ValueError(
                "Retained staging handle and destination must be supplied together"
            )
        if staged_retained is not None:
            if retained_epoch_record is None:
                raise ValueError("Prepared retained publication requires provenance")
            parent_bytes = (
                json.dumps(
                    previous,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                if previous is not None
                else b""
            )
            publication = {
                "format_version": 1,
                "generation": generation,
                "epoch": epoch,
                "parent_generation": (
                    None if previous is None else previous["generation"]
                ),
                "parent_manifest_sha256": hashlib.sha256(parent_bytes).hexdigest(),
                "manifest": manifest,
                "artifacts": artifact_hashes,
                "retained": {
                    "name": Path(retained_destination).name,
                    "digest": staged_retained.sealed_digest,
                },
            }
            publication_name = (
                f".checkpoint-generation-{generation}-publication.json"
            )
            with _open_anonymous_file(directory_fd, "w+") as publication_handle:
                json.dump(
                    publication,
                    publication_handle,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                publication_handle.write("\n")
                publication_handle.flush()
                os.fsync(publication_handle.fileno())
                _publish_fd_without_overwrite(
                    publication_handle, directory_fd, publication_name
                )
            os.fsync(directory_fd)
            _publish_retained_epoch(staged_retained, retained_destination)

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
    generation: Optional[str] = None
    epoch: Optional[int] = None
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


def _stage_retained_reconstructions(
    reconstructions, retained_root, epoch, generation=None
):
    _validate_retained_reconstruction_names(reconstructions)
    generation = generation or uuid.uuid4().hex
    if len(generation) != 32 or any(
        character not in "0123456789abcdef" for character in generation
    ):
        raise ValueError(
            "Retained generation must be 32 lowercase hexadecimal characters"
        )
    final_dir = Path(os.path.abspath(os.fspath(
        Path(retained_root) / f"epoch_{epoch:04d}"
    )))
    staged = _create_staged_directory(final_dir, "Retained epoch")
    staged.generation = generation
    staged.epoch = epoch
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



def _collect_retained_epoch_records(retained_root):
    retained_root = Path(retained_root)
    if not retained_root.exists():
        return [], []
    if not retained_root.is_dir() or retained_root.is_symlink():
        raise RuntimeError(f"Retained root is not a real directory: {retained_root}")
    records = []
    partial = []
    for entry in sorted(retained_root.iterdir(), key=lambda path: path.name):
        if "-unpublished-orphan-" in entry.name:
            partial.append(entry.name)
            continue
        if not entry.name.startswith("epoch_") or not entry.is_dir() or entry.is_symlink():
            partial.append(entry.name)
            continue
        try:
            epoch = int(entry.name.removeprefix("epoch_"))
        except ValueError:
            partial.append(entry.name)
            continue
        if entry.name != f"epoch_{epoch:04d}":
            partial.append(entry.name)
            continue
        directory_fd = os.open(entry, _directory_open_flags())
        try:
            digest = _tree_digest(directory_fd)
        finally:
            os.close(directory_fd)
        records.append({"epoch": epoch, "digest": digest})
    return records, partial


def reconcile_retained_checkpoint_state(exp_dir, retained_root, history_path=None):
    exp_dir = Path(exp_dir)
    records, partial = _collect_retained_epoch_records(retained_root)
    ambiguous_entries = [
        name for name in partial if "-unpublished-orphan-" not in name
    ]
    if ambiguous_entries:
        raise RuntimeError(
            "Ambiguous partial retained publication: "
            + ", ".join(sorted(ambiguous_entries))
        )
    if not exp_dir.is_dir():
        if records:
            raise RuntimeError("Retained output exists without a checkpoint directory")
        return False

    directory_fd = _open_checkpoint_directory(exp_dir)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        manifest = _read_checkpoint_manifest(exp_dir, directory_fd)
        expected = [] if manifest is None else list(manifest.get("retained_epochs", []))
        expected_by_epoch = {item["epoch"]: item for item in expected}
        actual_by_epoch = {item["epoch"]: item for item in records}
        missing = sorted(set(expected_by_epoch) - set(actual_by_epoch))
        changed = sorted(
            epoch
            for epoch in set(expected_by_epoch) & set(actual_by_epoch)
            if expected_by_epoch[epoch]["digest"]
            != actual_by_epoch[epoch]["digest"]
        )
        extra = sorted(set(actual_by_epoch) - set(expected_by_epoch))
        if missing or changed or len(extra) > 1:
            raise RuntimeError(
                "Retained/checkpoint publication is ambiguous: "
                f"missing={missing}, changed={changed}, extra={extra}"
            )
        if not extra:
            return manifest is not None

        epoch = extra[0]
        if expected:
            expected_next_epoch = expected[-1]["epoch"] + 1
            if epoch != expected_next_epoch:
                raise RuntimeError(
                    f"Prepared retained epoch must be {expected_next_epoch}, got {epoch}"
                )
        elif epoch <= 0:
            raise RuntimeError(f"Prepared retained epoch must be positive, got {epoch}")
        parent_bytes = (
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            if manifest is not None
            else b""
        )
        parent_hash = hashlib.sha256(parent_bytes).hexdigest()
        parent_generation = None if manifest is None else manifest["generation"]
        candidates = []
        with os.scandir(directory_fd) as entries:
            names = sorted(entry.name for entry in entries)
        for name in names:
            if not (
                name.startswith(".checkpoint-generation-")
                and name.endswith("-publication.json")
            ):
                continue
            with _open_regular_at(
                directory_fd, name, "checkpoint publication provenance"
            ) as handle:
                try:
                    publication = json.load(handle)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Invalid checkpoint publication: {name}"
                    ) from exc
            if not isinstance(publication, dict) or set(publication) != {
                "format_version", "generation", "epoch", "parent_generation",
                "parent_manifest_sha256", "manifest", "artifacts", "retained",
            }:
                raise RuntimeError(f"Invalid checkpoint publication schema: {name}")
            publication_version = publication["format_version"]
            publication_epoch = publication["epoch"]
            if (
                not isinstance(publication_version, int)
                or isinstance(publication_version, bool)
                or publication_version != 1
                or not isinstance(publication_epoch, int)
                or isinstance(publication_epoch, bool)
                or publication_epoch <= 0
            ):
                raise RuntimeError(
                    f"Invalid checkpoint publication provenance values: {name}"
                )
            retained = publication.get("retained", {})
            if not isinstance(retained, dict) or set(retained) != {"name", "digest"}:
                raise RuntimeError(f"Invalid retained provenance schema: {name}")
            if (
                publication.get("format_version") == 1
                and publication.get("epoch") == epoch
                and publication.get("parent_generation") == parent_generation
                and publication.get("parent_manifest_sha256") == parent_hash
                and retained.get("name") == f"epoch_{epoch:04d}"
                and retained.get("digest") == actual_by_epoch[epoch]["digest"]
            ):
                candidates.append((name, publication))
        if len(candidates) != 1:
            raise RuntimeError(
                "Prepared retained publication requires exactly one provenance "
                f"candidate, found {[name for name, _ in candidates]}"
            )
        publication_name, publication = candidates[0]
        candidate_manifest = publication.get("manifest")
        generation = publication.get("generation")
        if (
            not isinstance(candidate_manifest, dict)
            or candidate_manifest.get("format_version")
            != CHECKPOINT_MANIFEST_FORMAT_VERSION
            or not isinstance(generation, str)
            or len(generation) != 32
            or any(character not in "0123456789abcdef" for character in generation)
            or publication_name
            != f".checkpoint-generation-{generation}-publication.json"
            or candidate_manifest.get("model")
            != f".checkpoint-generation-{generation}-model.pt"
            or candidate_manifest.get("history")
            != f".checkpoint-generation-{generation}-history.npy"
            or set(candidate_manifest)
            != {
                "format_version", "generation", "model", "best",
                "history", "retained_epochs", "artifacts",
            }
        ):
            raise RuntimeError("Prepared publication schema mismatch")
        try:
            _validate_checkpoint_manifest_payload(candidate_manifest, publication_name)
        except ValueError as exc:
            raise RuntimeError("Prepared publication manifest is invalid") from exc
        ledger = candidate_manifest.get("retained_epochs", [])
        expected_ledger = expected + [{
            "epoch": epoch,
            "generation": generation,
            "digest": actual_by_epoch[epoch]["digest"],
        }]
        if (
            ledger != expected_ledger
            or candidate_manifest.get("generation") != generation
        ):
            raise RuntimeError("Prepared publication manifest ledger mismatch")
        artifact_hashes = publication.get("artifacts")
        expected_artifacts = {
            candidate_manifest.get("model"),
            candidate_manifest.get("history"),
            candidate_manifest.get("best"),
        } - {None}
        if (
            not isinstance(artifact_hashes, dict)
            or artifact_hashes != candidate_manifest.get("artifacts")
            or set(artifact_hashes) != expected_artifacts
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(
                    character not in "0123456789abcdef" for character in value
                )
                for value in artifact_hashes.values()
            )
        ):
            raise RuntimeError("Prepared publication artifact set mismatch")
        for artifact_name, expected_hash in artifact_hashes.items():
            with _open_regular_at(
                directory_fd, artifact_name, "prepared checkpoint artifact"
            ) as handle:
                if _sha256_open_handle(handle) != expected_hash:
                    raise RuntimeError(
                        "Prepared publication artifact digest mismatch: "
                        f"{artifact_name}"
                    )
        with _open_regular_at(
            directory_fd,
            candidate_manifest["model"],
            "prepared model artifact",
        ) as handle:
            state = _load_checkpoint_from_handle(handle)
        validate_training_checkpoint(state)
        if state["epoch"] != epoch:
            raise RuntimeError("Prepared publication model epoch mismatch")
        with _open_regular_at(
            directory_fd,
            candidate_manifest["history"],
            "prepared history artifact",
        ) as handle:
            history = np.load(handle, allow_pickle=False)
        _validate_val_loss_history(history, epoch, candidate_manifest["history"])
        best_artifact = candidate_manifest.get("best")
        if best_artifact is not None:
            with _open_regular_at(
                directory_fd, best_artifact, "prepared best artifact"
            ) as handle:
                best_state = _load_checkpoint_from_handle(handle)
            validate_checkpoint_pair(state, best_state)

        _publish_checkpoint_manifest(exp_dir, directory_fd, candidate_manifest)
        os.fsync(directory_fd)
        history_artifact = candidate_manifest.get("history")
        if history_artifact is not None and history_path is not None:
            _publish_history_alias(
                exp_dir, directory_fd, history_artifact, Path(history_path)
            )
        best_artifact = candidate_manifest.get("best")
        if best_artifact is not None:
            _publish_stable_alias(
                exp_dir, directory_fd, best_artifact, "best_model.pt"
            )
        _publish_stable_alias(
            exp_dir,
            directory_fd,
            candidate_manifest["model"],
            "model.pt",
        )
        os.fsync(directory_fd)
        return True
    finally:
        os.close(directory_fd)


def train(args):
    reconciled = False
    if (
        getattr(args, "model_family", "varnet") == "promptmr_plus"
        and getattr(args, "retain_val_epochs", False)
    ):
        reconciled = reconcile_retained_checkpoint_state(
            args.exp_dir,
            args.val_epochs_dir,
            Path(args.val_loss_dir) / "val_loss_log.npy",
        )
        if reconciled:
            if not recover_checkpoint_publication(args.exp_dir):
                raise RuntimeError(
                    "Reconciled PromptMR+ checkpoint has no authoritative manifest"
                )
            args.resume_checkpoint = Path(args.exp_dir) / "model.pt"
            args.resume_checkpoint_sha256 = None
    required_cuda_name = getattr(args, "require_cuda_device_name", None)
    if required_cuda_name is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Required CUDA device {required_cuda_name!r} is unavailable"
            )
        actual_cuda_name = torch.cuda.get_device_name(args.GPU_NUM)
        if actual_cuda_name != required_cuda_name:
            raise RuntimeError(
                f"Required CUDA device {required_cuda_name!r}, found {actual_cuda_name!r}"
            )
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.GPU_NUM}')
        torch.cuda.set_device(device)
        print('Current cuda device: ', torch.cuda.current_device())
    else:
        device = torch.device('cpu')
        print('Current device: cpu')

    if getattr(args, "model_family", "varnet") == "promptmr_plus":
        from utils.promptmr.runtime import (
            build_promptmr_plus_loss,
            build_promptmr_plus_model,
        )

        model = build_promptmr_plus_model()
        model.to(device=device)
        loss_type = build_promptmr_plus_loss().to(device=device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-2,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=35,
            gamma=0.1,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        checkpoint_contract = args.model_contract
    else:
        model = VarNet(num_cascades=args.cascade,
                       chans=args.chans,
                       sens_chans=args.sens_chans)
        model.to(device=device)

        if getattr(args, "score_aligned_loss", False):
            loss_type = ScoreAlignedLoss().to(device=device)
        else:
            loss_type = SSIMLoss().to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        scheduler = scaler = checkpoint_contract = None

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
            expected_sha256=getattr(args, "resume_checkpoint_sha256", None),
            scheduler=scheduler,
            scaler=scaler,
            expected_model_contract=getattr(
                args, "model_contract", {"model_family": "varnet"}
            ),
        )
        if start_epoch >= args.num_epochs:
            if reconciled and start_epoch == args.num_epochs:
                print(
                    "Recovered final committed PromptMR+ epoch; "
                    "requested training is already complete"
                )
                return
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

    
    if getattr(args, "model_family", "varnet") == "promptmr_plus":
        from utils.promptmr.data import create_promptmr_data_loaders

        train_loader, val_loader = create_promptmr_data_loaders(args)
    else:
        train_loader = create_data_loaders(
            data_path=args.data_path_train,
            args=args,
            shuffle=True,
            score_aligned=getattr(args, "score_aligned_loss", False),
        )
        val_loader = create_data_loaders(data_path=args.data_path_val, args=args)
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(
            args, epoch, model, train_loader, optimizer, loss_type, scaler=scaler
        )
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        if scheduler is not None:
            scheduler.step()
        generation = uuid.uuid4().hex
        staged_epoch_dir = retained_epoch_dir = retained_epoch_record = None
        if getattr(args, "retain_val_epochs", False):
            staged_epoch_dir, retained_epoch_dir = (
                _stage_retained_reconstructions(
                    reconstructions,
                    args.val_epochs_dir,
                    epoch + 1,
                    generation=generation,
                )
            )
            retained_epoch_record = {
                "epoch": epoch + 1,
                "generation": generation,
                "digest": staged_epoch_dir.sealed_digest,
            }
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = Path(args.val_loss_dir) / "val_loss_log.npy"

        train_loss = torch.tensor(train_loss, device=device)
        val_loss = torch.tensor(val_loss, device=device)
        num_subjects = torch.tensor(num_subjects, device=device)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        try:
            save_model(
                args.exp_dir,
                epoch + 1,
                model,
                optimizer,
                best_val_loss,
                is_new_best,
                val_loss_history=val_loss_log,
                history_path=file_path,
                scheduler=scheduler,
                scaler=scaler,
                model_contract=checkpoint_contract,
                generation=generation,
                retained_epoch_record=retained_epoch_record,
                staged_retained=staged_epoch_dir,
                retained_destination=retained_epoch_dir,
            )
            staged_epoch_dir = None
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

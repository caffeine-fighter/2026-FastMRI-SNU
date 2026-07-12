import h5py
import json
import random
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np


def _acceleration_from_name(fname):
    matches = [
        token for token in Path(fname).stem.lower().split("_")
        if token in {"acc4", "acc8"}
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Unknown acceleration in {Path(fname).name}: expected one acc4/acc8 token"
        )
    return int(matches[0][3:])


def _parse_score_annotations(raw, num_slices, fname):
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        annotations = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as exc:
        raise ValueError(f"Malformed annotations in {fname}: {exc}") from exc
    if not isinstance(annotations, dict):
        raise ValueError(f"Malformed annotations in {fname}: expected JSON object")

    boxes_by_slice = {slice_idx: [] for slice_idx in range(num_slices)}
    required = ("x", "y", "width", "height")
    int64_min, int64_max = -(1 << 63), (1 << 63) - 1
    for key, boxes in annotations.items():
        valid_key = (
            isinstance(key, str)
            and key.isascii()
            and key.isdigit()
            and key == str(int(key))
            and 0 <= int(key) < num_slices
        )
        if not valid_key or not isinstance(boxes, list):
            raise ValueError(f"Malformed annotation slice {key!r} in {fname}")
        for box in boxes:
            valid_box = (
                isinstance(box, dict)
                and all(field in box for field in required)
                and all(type(box[field]) is int for field in required)
                and box["width"] > 0
                and box["height"] > 0
            )
            if not valid_box:
                raise ValueError(f"Malformed annotation box in {fname}: {box!r}")
            if any(
                not int64_min <= box[field] <= int64_max
                for field in required
            ):
                raise ValueError(
                    f"Annotation coordinate outside int64 in {fname}: {box!r}"
                )
            boxes_by_slice[int(key)].append(tuple(box[field] for field in required))
    return boxes_by_slice


def _box_is_evaluator_accepted(box, image_shape, win_size=7):
    x, y, width, height = box
    image_height, image_width = image_shape
    x0, y0 = max(0, x), max(0, y)
    x1 = min(image_width, x + width)
    y1 = min(image_height, y + height)
    return (x1 - x0) >= win_size and (y1 - y0) >= win_size


def derive_score_weights(slice_counts, box_counts):
    if set(slice_counts) != {4, 8} or set(box_counts) != {4, 8}:
        raise ValueError("Score weights require acc4 and acc8 counts")
    if any(type(count) is not int or count <= 0 for count in slice_counts.values()):
        raise ValueError("Score slice counts must be positive integers")
    if any(type(count) is not int or count <= 0 for count in box_counts.values()):
        raise ValueError("Score box counts must be positive integers")
    total_slices = sum(slice_counts.values())
    return {
        "full": {
            acceleration: total_slices / (4 * slice_counts[acceleration])
            for acceleration in (4, 8)
        },
        "box": {
            acceleration: total_slices / (4 * box_counts[acceleration])
            for acceleration in (4, 8)
        },
    }

class SliceData(Dataset):
    def __init__(
        self, root, transform, input_key, target_key, forward=False,
        score_aligned=False,
    ):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.score_aligned = score_aligned and not forward
        self.image_examples = []
        self.kspace_examples = []
        self.score_counts = None
        self.score_weights = None
        self._score_volumes = {}
        self._score_shapes = {}
        self._score_max_boxes = 0

        if not forward:
            image_files = list(Path(root / "image").iterdir())
            if self.score_aligned:
                kspace_files = list(Path(root / "kspace").iterdir())
                image_names = {fname.name for fname in image_files}
                kspace_names = {fname.name for fname in kspace_files}
                if image_names != kspace_names:
                    raise ValueError(
                        "Score-aligned image and kspace filename sets must match"
                    )
                kspace_by_name = {fname.name: fname for fname in kspace_files}

            for fname in sorted(image_files):
                if self.score_aligned:
                    num_slices = self._get_required_slice_count(
                        fname, self.target_key, "image"
                    )
                    kspace_slices = self._get_required_slice_count(
                        kspace_by_name[fname.name], self.input_key, "kspace"
                    )
                    if kspace_slices != num_slices:
                        raise ValueError(
                            f"Score-aligned slice count mismatch for {fname.name}"
                        )
                else:
                    num_slices = self._get_metadata(fname)

                if self.score_aligned:
                    acceleration = _acceleration_from_name(fname)
                    with h5py.File(fname, "r") as hf:
                        if "annotations" not in hf.attrs:
                            raise ValueError(f"Missing annotations in {fname.name}")
                        image_shape = tuple(hf[self.target_key].shape[1:])
                        if len(image_shape) != 2 or any(
                            type(size) is not int or size <= 0
                            for size in image_shape
                        ):
                            raise ValueError(
                                f"Malformed target image shape in {fname.name}"
                            )
                        boxes_by_slice = _parse_score_annotations(
                            hf.attrs["annotations"], num_slices, fname.name
                        )
                    self._score_volumes[fname.name] = (
                        acceleration, boxes_by_slice
                    )
                    self._score_shapes[fname.name] = image_shape
                    self._score_max_boxes = max(
                        self._score_max_boxes,
                        *(len(boxes) for boxes in boxes_by_slice.values()),
                    )

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

            if self.score_aligned:
                slice_counts = {4: 0, 8: 0}
                box_counts = {4: 0, 8: 0}
                for name, (acceleration, boxes_by_slice) in self._score_volumes.items():
                    slice_counts[acceleration] += len(boxes_by_slice)
                    box_counts[acceleration] += sum(
                        sum(
                            _box_is_evaluator_accepted(
                                box, self._score_shapes[name]
                            )
                            for box in boxes
                        )
                        for boxes in boxes_by_slice.values()
                    )
                self.score_counts = {
                    "slices": slice_counts,
                    "boxes": box_counts,
                }
                self.score_weights = derive_score_weights(
                    slice_counts, box_counts
                )

        kspace_files = list(Path(root / "kspace").iterdir())
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)

            self.kspace_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]


    @staticmethod
    def _get_required_slice_count(fname, dataset_key, side):
        with h5py.File(fname, "r") as hf:
            if dataset_key not in hf:
                raise ValueError(
                    f"Missing {side} dataset {dataset_key!r} in {Path(fname).name}"
                )
            dataset = hf[dataset_key]
            if not isinstance(dataset, h5py.Dataset) or not dataset.shape:
                raise ValueError(
                    f"Malformed {side} dataset {dataset_key!r} in {Path(fname).name}"
                )
            num_slices = dataset.shape[0]
        if type(num_slices) is not int or num_slices <= 0:
            raise ValueError(
                f"Nonpositive {side} slice count in {Path(fname).name}"
            )
        return num_slices

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]
        if not self.forward and image_fname.name != kspace_fname.name:
            raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
            
        if not self.score_aligned:
            return self.transform(
                mask, input, target, attrs, kspace_fname.name, dataslice
            )
        acceleration, boxes_by_slice = self._score_volumes[kspace_fname.name]
        score_metadata = {
            "acceleration": acceleration,
            "boxes": boxes_by_slice[dataslice],
            "max_boxes": self._score_max_boxes,
            "full_weight": self.score_weights["full"][acceleration],
            "box_weight": self.score_weights["box"][acceleration],
        }
        return self.transform(
            mask, input, target, attrs, kspace_fname.name, dataslice,
            score_metadata=score_metadata,
        )


def create_data_loaders(
    data_path, args, shuffle=False, isforward=False, score_aligned=False
):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(
            isforward, max_key_, score_aligned=score_aligned
        ),
        input_key=args.input_key,
        target_key=target_key_,
        forward=isforward,
        score_aligned=score_aligned,
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader

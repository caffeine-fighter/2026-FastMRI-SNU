import contextlib
import copy
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as torch_functional

MODEL_UTILS = Path(__file__).resolve().parents[1] / "utils" / "model"
if str(MODEL_UTILS) not in sys.path:
    sys.path.insert(1, str(MODEL_UTILS))

from utils.model.fi_varnet_adapter import (
    FI_DETERMINISTIC_REFLECT_PAD_CONTRACT,
    deterministic_reflect_pad2d,
    install_deterministic_reflect_pad_adapter,
    load_pinned_fi_varnet_class,
    validate_deterministic_reflect_pad_receipt,
)


@contextlib.contextmanager
def _strict_determinism():
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(
            previous_enabled, warn_only=previous_warn_only
        )


@pytest.mark.parametrize("shape", [(1, 1, 3, 4), (2, 3, 7, 9), (1, 2, 16, 18)])
@pytest.mark.parametrize("pad", [(0, 0, 0, 0), (1, 1, 1, 1), (2, 1, 0, 2), (0, 3, 2, 0)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
def test_deterministic_reflect_pad2d_is_native_forward_exact(shape, pad, dtype):
    left, right, top, bottom = pad
    if left >= shape[-1] or right >= shape[-1] or top >= shape[-2] or bottom >= shape[-2]:
        pytest.skip("native reflect padding rejects this shape/pad pair")
    values = torch.arange(torch.tensor(shape).prod().item(), dtype=dtype).reshape(shape)

    expected = torch_functional.pad(values, pad, mode="reflect")
    actual = deterministic_reflect_pad2d(values, pad)

    assert torch.equal(actual, expected)
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device


def test_deterministic_reflect_pad2d_backward_is_strict_and_repeat_exact():
    gradients = []
    with _strict_determinism():
        for _ in range(2):
            value = torch.arange(2 * 3 * 7 * 9, dtype=torch.float32).reshape(2, 3, 7, 9)
            value.requires_grad_(True)
            output = deterministic_reflect_pad2d(value, (2, 3, 1, 2))
            output.square().sum().backward()
            gradients.append(value.grad.detach().clone())
    assert torch.equal(gradients[0], gradients[1])


def test_reflect_adapter_is_pinned_module_local_idempotent_and_state_transparent():
    fi_class = load_pinned_fi_varnet_class()
    module = __import__(fi_class.__module__, fromlist=["F"])
    global_pad = torch_functional.pad
    model = fi_class(
        num_cascades=1,
        chans=2,
        pools=1,
        sens_chans=2,
        sens_pools=1,
        acceleration=4,
    )
    before = copy.deepcopy(model.state_dict())

    with _strict_determinism():
        first = install_deterministic_reflect_pad_adapter(model)
        second = install_deterministic_reflect_pad_adapter(model)

    assert first == second == FI_DETERMINISTIC_REFLECT_PAD_CONTRACT
    assert validate_deterministic_reflect_pad_receipt(first) == first
    assert torch_functional.pad is global_pad
    assert module.F is not torch_functional
    assert module.F.softmax is torch_functional.softmax
    assert list(model.state_dict()) == list(before)
    assert all(torch.equal(model.state_dict()[key], value) for key, value in before.items())

    value = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    assert torch.equal(
        module.F.pad(value, (1, 2, 1, 0), mode="reflect"),
        torch_functional.pad(value, (1, 2, 1, 0), mode="reflect"),
    )
    assert torch.equal(
        module.F.pad(value, (1, 0, 0, 0), mode="constant", value=7),
        torch_functional.pad(value, (1, 0, 0, 0), mode="constant", value=7),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda receipt: receipt.pop("implementation"),
        lambda receipt: receipt.update(version=True),
        lambda receipt: receipt.update(native_forward_exact=False),
        lambda receipt: receipt.update(state_dict_unchanged=False),
        lambda receipt: receipt.update(strict_deterministic_algorithms=False),
        lambda receipt: receipt.update(scope="global-torch-functional"),
    ],
)
def test_reflect_adapter_receipt_schema_v2_is_closed(mutation):
    malformed = dict(FI_DETERMINISTIC_REFLECT_PAD_CONTRACT)
    mutation(malformed)
    with pytest.raises(ValueError, match="reflect-padding adapter receipt"):
        validate_deterministic_reflect_pad_receipt(malformed)


@pytest.mark.parametrize(
    "tensor,pad,message",
    [
        (torch.ones(3, 4), (1, 1, 1, 1), "4D"),
        (torch.ones(1, 1, 3, 4), (1, 1), "four integers"),
        (torch.ones(1, 1, 3, 4), (True, 0, 0, 0), "four integers"),
        (torch.ones(1, 1, 3, 4), (-1, 0, 0, 0), "nonnegative"),
        (torch.ones(1, 1, 3, 4), (4, 0, 0, 0), "smaller than input"),
        (torch.ones(1, 1, 3, 4), (0, 0, 3, 0), "smaller than input"),
    ],
)
def test_deterministic_reflect_pad2d_validation_is_closed(tensor, pad, message):
    with pytest.raises(ValueError, match=message):
        deterministic_reflect_pad2d(tensor, pad)

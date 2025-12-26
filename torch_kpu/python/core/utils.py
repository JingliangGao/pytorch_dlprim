import torch
import torch_kpu

from torch._utils import _get_device_index as _torch_get_device_index
from typing import Any, Optional


class device(object):
    current_device = 0

    def __init__(self, device):
        pass

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        return False


def _get_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a KPU device. Note that for a KPU device without a specified index,
    i.e., ``torch.device('kpu')``, this will return the current default KPU
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default KPU
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if allow_cpu:
            if device.type not in ["kpu", "cpu"]:
                raise ValueError(
                    "Expected a kpu or cpu device, but got: {}".format(device)
                )
        elif device.type != "kpu":
            raise ValueError("Expected a kpu device, but got: {}".format(device))
    if not torch.jit.is_scripting():
        if isinstance(device, torch.kpu.device):
            return device.idx
    return _torch_get_device_index(device, optional, allow_cpu)


def set_device(device):
    device_id = _get_device_index(device, optional=True)
    if device_id >= 0:
        torch_kpu._C._kpu_setDevice(device_id)

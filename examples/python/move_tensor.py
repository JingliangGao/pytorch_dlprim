import torch
import sys
import os

python_path_env = os.environ.get("PYTHONPATH", "")
paths = python_path_env.split(":")
for p in paths:
    pt_ocl_path = os.path.join(p, "pytorch_ocl")
    if pt_ocl_path and pt_ocl_path not in sys.path: 
        sys.path.append(pt_ocl_path)
from pt_ocl import *

def _device_index(device):
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if device.index is None:
            return 0
        return device.index
    return -1


class _OCL:
    class profile:
        """ Enables profiling for a ocl device and saves result log"""
        def __init__(self,device,path = None):
            """ if path is not None profiling is enabled and result is saved in csv format to path"""
            self._device_id = _device_index(device)
            self._path = path

        def __enter__(self):
            if self._path is not None:
                impl_start_profiling(self._device_id)
        
        def __exit__(self, type, value, traceback):
            if self._path is not None:
                impl_stop_profiling(self._device_id,self._path)

    def enable_profiling(device):
        impl_enable_profiling(_device_index(device))
        
    class device:
        current_device = 0
        def __init__(self, device):
            pass
            #self.idx = _device_index(device)
            #self.prev_idx = -1

        def __enter__(self):
            pass
            #print("Enter:",self.idx)
            #self.prev_idx = self.current_device
            #self.current_device = self.idx
        
        def __exit__(self, type, value, traceback):
            #print("Leave:",self.idx,"->",self.current_device)
            #self.current_device=self.prev_device
            return False

    @staticmethod
    def synchronize(dev = None):
        if dev is None:
            impl_synchronize_device(-1)
        else:
            impl_synchronize_device(_device_index(dev))

    @staticmethod
    def manual_seed_all(seed:int):
        impl_seed_all(seed)

    @staticmethod
    def _is_in_bad_fork():
        return impl_is_bad_fork()

    @staticmethod
    def empty_cache():
        impl_empty_cache()

def synchronize(dev):
    _OCL.synchronize(Dev)

def manual_seed_all(seed):
    _OCL.manual_seed_all(seed)

def empty_cache():
    _OCL.empty_cache()


torch.utils.rename_privateuse1_backend('ocl')
torch._register_device_module("ocl", _OCL)

t = torch.tensor([0.5, 0.3])
t2 = torch.tensor([0.2, 0.4])

print("Creating ocl")
dev = torch.device('ocl')
print("Moving to ocl")
o = t.to(dev)
print("Result : ", o)
import torch
from .pt_ocl import *
from pathlib import Path
import os, shutil
import json

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



def merge_json_files(folder, remove_cache=False):
    json_files = [f for f in os.listdir(folder) if f.endswith(".json")]
    if len(json_files) != 2:
        raise RuntimeError(f"[ERROR] Can not find correct JSON file in {folder}, but find : {json_files} .")

    # check JSON file if exist
    kylin_file = None
    torch_file = None
    for f in json_files:
        if f == "kylin.TEMP1174003943.pt.trace.json":
            kylin_file = os.path.join(folder, f)
        else:
            torch_file = os.path.join(folder, f)
    if not kylin_file or not torch_file:
        raise RuntimeError("[ERROR] Can not find kylin or torch JSON file .")

    # read JSON context
    with open(kylin_file, "r", encoding="utf-8") as f:
        kylin_data = json.load(f)
    with open(torch_file, "r", encoding="utf-8") as f:
        torch_data = json.load(f)

    # merge JSON context
    torch_traceEvents = torch_data["traceEvents"]
    torch_data["traceEvents"] = torch_traceEvents + kylin_data

    # write into file
    output_file = torch_file.replace(".pt.trace.json", ".merged.pt.trace.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(torch_data, f, ensure_ascii=False, indent=4)

    # remove cache
    if (remove_cache):
        os.remove(kylin_file)
        os.remove(torch_file)

class _OCL:
    class profile:
        """ Enables profiling for a ocl device and saves result log"""
        def __init__(self,device, path = None):
            """ if path is not None profiling is enabled and result is saved in csv format to path"""
            self._device_id = _device_index(device)
            self._path = path
            self.prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self._path),
                record_shapes=True,
                profile_memory=True,
                with_modules=True,
                with_stack=True)

        def __enter__(self):
            if self._path is not None:
                # remove old profile folder 
                if os.path.isdir(self._path):
                    shutil.rmtree(self._path)
                os.makedirs(self._path)
                self.prof.start()
                impl_start_profiling(self._device_id)
        
        def __exit__(self, type, value, traceback):
            if self._path is not None:
                fullpath = str(Path(self._path) / "kylin.TEMP1174003943.pt.trace.json")
                impl_stop_profiling(self._device_id, fullpath)
                self.prof.stop()
                merge_json_files(self._path)

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


try: 
    torch.utils.rename_privateuse1_backend('ocl')
    torch._register_device_module("ocl", _OCL)
except:
    pass

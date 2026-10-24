"""Track total device memory (including CUDA allocations outside Torch) via NVML."""
import ctypes as C
import threading
import time
import os
from pathlib import Path

class HostMemory:
    """Sample the entire capped scope, including loop backing-file cache."""
    def __init__(self,path,limit=None):
        limit=int(os.environ.get('QWEN_RAM_LIMIT_BYTES','16000000000')) if limit is None else limit
        self.path=Path(path); self.limit=limit; self.peak=0; self.running=True
        cap=(self.path/'memory.max').read_text().strip()
        assert cap!='max' and int(cap)<=limit, f'Uncapped global RAM scope: {self.path}'
        assert (self.path/'memory.swap.max').read_text().strip()=='0'
        self.thread=threading.Thread(target=self._run,daemon=True); self.thread.start()
    def sample(self):
        used=int((self.path/'memory.current').read_text()); self.peak=max(self.peak,used)
        return used
    def _run(self):
        while self.running:
            if self.sample()>self.limit:
                print(f'RAM LIMIT EXCEEDED: {self.peak} > {self.limit}',flush=True)
                os._exit(87)
            time.sleep(0.02)
    def close(self):
        self.running=False; self.thread.join(); self.sample()

class DeviceMemory:
    class Info(C.Structure):
        _fields_=[('total',C.c_ulonglong),('free',C.c_ulonglong),('used',C.c_ulonglong)]
    def __init__(self,limit=16_000_000_000):
        self.lib=C.CDLL('libnvidia-ml.so.1')
        assert self.lib.nvmlInit_v2()==0
        self.handle=C.c_void_p()
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(0,C.byref(self.handle))==0
        self.limit=limit; self.peak=0; self.running=True
        self.thread=threading.Thread(target=self._run,daemon=True); self.thread.start()
    def sample(self):
        info=self.Info()
        assert self.lib.nvmlDeviceGetMemoryInfo(self.handle,C.byref(info))==0
        self.peak=max(self.peak,info.used)
        return info.used
    def _run(self):
        while self.running:
            if self.sample()>self.limit:
                print(f'VRAM LIMIT EXCEEDED: {self.peak} > {self.limit}',flush=True)
                import sys
                torch=sys.modules.get('torch')
                if torch is not None and torch.cuda.is_initialized():
                    print(f'TORCH MEMORY AT EXIT: allocated={torch.cuda.memory_allocated()} '
                          f'reserved={torch.cuda.memory_reserved()}',flush=True)
                os._exit(86)
            time.sleep(0.02)
    def close(self):
        self.running=False; self.thread.join(); self.sample()

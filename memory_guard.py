"""Track total device memory (including CUDA allocations outside Torch) via NVML."""
import ctypes as C
import threading
import time
import os
from pathlib import Path

class HostMemory:
    """Sample cgroup RAM, including file cache, and enforce the runtime limit."""
    def __init__(self,path,limit=None):
        limit=int(os.environ.get('QWEN_RAM_LIMIT_BYTES','16000000000')) if limit is None else limit
        self.path=Path(path); self.limit=limit; self.peak=0; self.running=True; self.thread=None
        # A preconfigured kernel RAM/swap cap is not required for inference.
        # The sampler below still enforces self.limit.
        if not (self.path/'memory.current').exists():
            # e.g. WSL's root cgroup, or a desktop session without a delegated memory controller: nothing to
            # measure, so the RAM guard stays off.
            print(f'RAM guard off: no memory.current in {self.path} (start inside a memory cgroup to enable it)',flush=True)
            self.running=False; return
        self.thread=threading.Thread(target=self._run,daemon=True); self.thread.start()
    def sample(self):
        if not (self.path/'memory.current').exists(): return 0
        used=int((self.path/'memory.current').read_text()); self.peak=max(self.peak,used)
        return used
    def _run(self):
        while self.running:
            if self.sample()>self.limit:
                print(f'RAM LIMIT EXCEEDED: {self.peak} > {self.limit}',flush=True)
                os._exit(87)
            time.sleep(0.02)
    def close(self):
        self.running=False
        if self.thread: self.thread.join()
        self.sample()

def device_memory(uuid):
    """(total, used) VRAM of the GPU with this UUID ('GPU-...') as NVML counts it, every process included (what
    DeviceMemory guards), or None without NVML."""
    try:
        lib=C.CDLL('libnvidia-ml.so.1')
    except OSError:
        return None
    if lib.nvmlInit_v2()!=0:
        return None
    handle=C.c_void_p(); info=DeviceMemory.Info()
    if lib.nvmlDeviceGetHandleByUUID(uuid.encode(),C.byref(handle))!=0 or \
       lib.nvmlDeviceGetMemoryInfo(handle,C.byref(info))!=0:
        return None
    return info.total,info.used

class DeviceMemory:
    class Info(C.Structure):
        _fields_=[('total',C.c_ulonglong),('free',C.c_ulonglong),('used',C.c_ulonglong)]
    def __init__(self,limit=None):
        # Device-wide (NVML) cap, other processes included: QWEN_VRAM_GUARD_BYTES, else QWEN_VRAM_LIMIT_BYTES
        if limit is None:
            limit=int(os.environ.get('QWEN_VRAM_GUARD_BYTES',os.environ.get('QWEN_VRAM_LIMIT_BYTES','16000000000')))
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

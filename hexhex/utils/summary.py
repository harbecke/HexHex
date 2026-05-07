import threading
import time

from torch.utils.tensorboard import SummaryWriter

_LAYOUT = {
    'training': {
        'loss': ['Multiline', ['train/train_loss', 'train/val_loss']],
        'grad norm': ['Multiline', ['train/grad_norm']],
    },
    'timing': {
        'rst breakdown': ['Multiline', ['time/data_generation', 'time/training', 'time/evaluation']],
        'rst total': ['Multiline', ['time/rst_iteration']],
    },
    'gpu': {
        'utilization': ['Multiline', ['gpu/utilization', 'gpu/memory_utilization']],
        'memory': ['Multiline', ['gpu/memory_used_gb', 'gpu/memory_used_pct']],
    },
}


class _WriterProxy:
    """Lazy proxy so `from hexhex.utils.summary import writer` works while still
    letting the training entry point pin a per-experiment log_dir via init()."""

    def __init__(self):
        self._w: SummaryWriter | None = None

    def init(self, log_dir: str) -> None:
        self._w = SummaryWriter(log_dir=log_dir)
        self._w.add_custom_scalars(_LAYOUT)

    def _ensure(self) -> SummaryWriter:
        if self._w is None:
            self._w = SummaryWriter()
            self._w.add_custom_scalars(_LAYOUT)
        return self._w

    def __getattr__(self, name):
        return getattr(self._ensure(), name)


writer = _WriterProxy()


def start_gpu_monitor(interval_seconds: float = 2.0) -> None:
    """Spawn a daemon thread polling NVML and writing GPU stats as TB scalars.

    Step axis = elapsed seconds since monitor start (so curves plot vs wallclock).
    No-op when CUDA is unavailable or NVML can't be loaded — local mac runs are
    unaffected.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return

    start = time.time()

    def loop():
        while True:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                step = int(time.time() - start)
                writer.add_scalar('gpu/utilization', util.gpu, step)
                writer.add_scalar('gpu/memory_utilization', util.memory, step)
                writer.add_scalar('gpu/memory_used_gb', mem.used / 1e9, step)
                writer.add_scalar('gpu/memory_used_pct', 100 * mem.used / mem.total, step)
            except Exception:
                pass
            time.sleep(interval_seconds)

    threading.Thread(target=loop, daemon=True).start()

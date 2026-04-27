from torch.utils.tensorboard import SummaryWriter

_LAYOUT = {
    'training': {
        'loss': ['Multiline', ['train/train_loss', 'train/val_loss']],
        'grad norm': ['Multiline', ['train/grad_norm']],
    },
    'timing': {
        'rst breakdown': ['Multiline', ['time/data_generation', 'time/training', 'time/evaluation', 'time/elo_tournament']],
        'rst total': ['Multiline', ['time/rst_iteration']],
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

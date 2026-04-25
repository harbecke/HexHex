from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

layout = {
    'training': {
        'loss': ['Multiline', ['train/train_loss', 'train/val_loss']],
        'grad norm': ['Multiline', ['train/grad_norm']],
    },
    'timing': {
        'rst breakdown': ['Multiline', ['time/data_generation', 'time/training', 'time/evaluation']],
        'rst total': ['Multiline', ['time/rst_iteration']],
    },
}
writer.add_custom_scalars(layout)
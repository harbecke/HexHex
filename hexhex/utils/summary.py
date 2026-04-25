from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

layout = {
    'training': {
        'loss': ['Multiline', ['train/train_loss', 'train/val_loss']],
        'grad norm': ['Multiline', ['train/grad_norm']],
    },
}
writer.add_custom_scalars(layout)
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(flush_secs=5)

layout = {'training': {'val/train loss': ['Multiline', ['train/train_loss', 'train/val_loss']]}}
writer.add_custom_scalars(layout)
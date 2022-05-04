import torch
import torch.optim as optim
import config
import argparse
import models.data as data
from tensorboardX import SummaryWriter
import os
from utils.checkpoint import CheckpointIO
import numpy as np

# Config
cfg = config.load_config('default.yaml')
is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")
out_dir = cfg['training']['out_dir']
# Dataset
train_dataset = config.get_dataset('train', cfg)
val_dataset = config.get_dataset('val', cfg)


#print(train_dataset.models)
#assert(0)

# Shorthands
batch_size = cfg['training']['batch_size']
batch_size_val = cfg['training']['batch_size_val']
model_selection_sign = -1
# model_selection_metric = cfg['training']['model_selection_metric']
# if cfg['training']['model_selection_mode'] == 'maximize':
#     model_selection_sign = 1
# elif cfg['training']['model_selection_mode'] == 'minimize':
#     model_selection_sign = -1
# else:
#     raise ValueError('model_selection_mode must be '
#                      'either maximize or minimize.')

lr = cfg['training']['learning_rate']

# Dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True,
                                            collate_fn=data.collate_remove_none,
                                            worker_init_fn=data.worker_init_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_val, num_workers=0, shuffle=False,
                                        collate_fn=data.collate_remove_none,
                                        worker_init_fn=data.worker_init_fn)

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)
model = model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Load pre-trained model is existing
kwargs = {
    'model': model,
    'optimizer': optimizer,
}
checkpoint_io = CheckpointIO(
    out_dir, initialize_from=cfg['model']['initialize_from'],
    initialization_file_name=cfg['model']['initialization_file_name'],
    **kwargs)
try:
    load_dict = checkpoint_io.load('model.pt')
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

# Trainer
trainer = config.get_trainer(model, optimizer, cfg, device)

# Training Info
epoch_it = -1
it = -1

logger = SummaryWriter(os.path.join(out_dir, 'logs'))
print_every = cfg['training']['print_every']
validate_every = cfg['training']['validate_every']

# Training loop

while True:
    epoch_it += 1

    for batch in train_loader:
        it += 1
        
        loss = trainer.train_step(batch)
        logger.add_scalar('train/loss', loss, it)
        
        if print_every > 0 and (it % print_every) == 0:
            print('[Epoch %02d] it=%03d, loss=%.4f'
                  % (epoch_it, it, loss))
            

        if it > 0 and validate_every > 0 and (it % validate_every) == 0:
            # print("EVALUATE")
            eval_dict = trainer.evaluate(val_loader)
            metric_val = eval_dict["loss"]
            print('Validation %s: %.4f'
                  % ("loss", metric_val))

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)


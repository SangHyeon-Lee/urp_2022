import torch
import torch.optim as optim
import config
import argparse
import models.data as data

# Config
cfg = config.load_config('default.yaml')
is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")

# Dataset
train_dataset = config.get_dataset('train', cfg)
val_dataset = config.get_dataset('val', cfg)


# Shorthands
batch_size = cfg['training']['batch_size']
batch_size_val = cfg['training']['batch_size_val']


# Dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
                                            collate_fn=data.collate_remove_none,
                                            worker_init_fn=data.worker_init_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False,
                                        collate_fn=data.collate_remove_none,
                                        worker_init_fn=data.worker_init_fn)

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

# Optimizer
optimizer = optim.Adam(model.parameters())

# Trainer
trainer = config.get_trainer(model, optimizer, cfg, device)

# Training Info
epoch_it = -1
it = -1

print_every = cfg['training']['print_every']


# Training loop

while True:
    epoch_it += 1

    print(epoch_it)

    for batch in train_loader:
        it += 1

        loss = trainer.train_step(batch)

        if print_every > 0 and (it % print_every) == 0:
            print('[Epoch %02d] it=%03d, loss=%.4f'
                  % (epoch_it, it, loss))



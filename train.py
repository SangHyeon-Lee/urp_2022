import torch
import torch.optim as optim
import config
import argparse

# Config
cfg = config.load_config('configs/default.yaml')
is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")

# Dataset
train_dataset = config.get_dataset('train', cfg)
val_dataset = config.get_dataset('val', cfg)

# Dataloader
train_loader = torch.utils.data.DataLoader(train_dataset)
val_loader = torch.utils.data.DataLoader(val_dataset)

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

# Optimizer
optimizer = optim.Adam(model.parameters())

# Trainer
trainer = config.get_trainer(model, optimizer)

# Training loop
while True:
    for batch in train_loader:
        loss = trainer.train_step(batch)


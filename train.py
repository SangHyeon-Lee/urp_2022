import torch
import torch.optim as optim

# Dataset
train_dataset = get_dataset()
val_dataset = get_dataset()

# Dataloader
train_loader = torch.utils.data.DataLoader(train_dataset)
val_loader = torch.utils.data.DataLoader(val_dataset)

# Model
model = get_model()

# Optimizer
optimizer = optim.Adam(model.parameters())

# Trainer
trainer = get_trainer(model, optimizer)

# Training loop
while True:
    for batch in train_loader:
        loss = trainer.train_step(batch)


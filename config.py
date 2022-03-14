import yaml
import models

# General config
def load_config(path):

    with open(path, 'r') as f:
        cfg = yaml.load(f)

    return cfg

# Models
def get_model(cfg, device=None, dataset=None):
    model = models.OccupancyFlow()
    return model


# Trainer
def get_trainer(model, optimizer, cfg, device):
    
    return trainer


# Generator for final mesh extraction
def get_generator(model, cfg, device):
    
    return generator


# Datasets
def get_dataset(mode, cfg, return_idx=False, return_category=False):

    return dataset


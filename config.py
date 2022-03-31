import yaml
import models, training
import torch
import torch.distributions as dist
from torch import nn
import os

# General config
def load_config(path):

    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return cfg

def get_decoder(cfg, device, dim=3, c_dim=0, z_dim=0):
    
    decoder = models.decoder.Decoder()
    return decoder

def get_velocity_field(cfg, device, dim=3, c_dim=0, z_dim=0):
    
    velocity_field = models.velocity_field.VelocityField()
    return velocity_field

def get_encoder(cfg, device, dataset=None, c_dim=0):
    
    encoder = models.pointnet.ResnetPointnet()
    return encoder

def get_encoder_latent(cfg, device, c_dim=0, z_dim=0):

    encoder_latent = models.encoder_latent.PointNet()
    return encoder_latent

def get_encoder_latent_temporal(cfg, device, c_dim=0, z_dim=0):

    encoder_latent_temporal = models.encoder_latent.PointNet()
    return encoder_latent_temporal

def get_encoder_temporal(cfg, device, dataset=None, c_dim=0, z_dim=0):
    
    encoder_temporal = models.pointnet.ResnetPointnet()
    return encoder_temporal

def get_prior_z(cfg, device, **kwargs):
    ''' Returns the prior distribution of latent code z.

    Args:
        cfg (yaml config): yaml config object
        device (device): PyTorch device
    '''
    z_dim = cfg['model']['z_dim'] #TODO
    p0_z = dist.Normal(
        torch.zeros(z_dim, device=device),
        torch.ones(z_dim, device=device)
    )

    return p0_z


# Models
def get_model(cfg, device=None, dataset=None):

    # Shortcuts
    dim = cfg['data']['dim'] # 3
    z_dim = cfg['model']['z_dim'] # 128
    c_dim = cfg['model']['c_dim'] # 0
    input_type = cfg['data']['input_type'] # pcl_seq
    ode_solver = cfg['model']['ode_solver'] # dopri5
    ode_step_size = cfg['model']['ode_step_size'] # ?
    use_adjoint = cfg['model']['use_adjoint'] # true
    rtol = cfg['model']['rtol'] # 0.001
    atol = cfg['model']['atol'] # 0.00001

    decoder = get_decoder(cfg, device, dim, c_dim, z_dim)
    velocity_field = get_velocity_field(cfg, device, dim, c_dim, z_dim)
    # ~~~~~~~~~~~~~~~#
    encoder = get_encoder(cfg, device, dataset, c_dim)
    encoder_latent = get_encoder_latent(cfg, device, c_dim, z_dim)
    encoder_latent_temporal = get_encoder_latent_temporal(
        cfg, device, c_dim, z_dim)
    encoder_temporal = get_encoder_temporal(cfg, device, dataset, c_dim, z_dim)
    p0_z = get_prior_z(cfg, device)


    model = models.OccupancyFlow(decoder=decoder, encoder=encoder, encoder_latent=encoder_latent,
        encoder_latent_temporal=encoder_latent_temporal,
        encoder_temporal=encoder_temporal, vector_field=velocity_field,
        ode_step_size=ode_step_size, use_adjoint=use_adjoint,
        rtol=rtol, atol=atol, ode_solver=ode_solver,
        p0_z=p0_z, device=device, input_type=input_type)
    return model


# Trainer
def get_trainer(model, optimizer, cfg, device):

    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']
    loss_corr = cfg['model']['loss_corr']
    loss_recon = cfg['model']['loss_recon']
    loss_corr_bw = cfg['model']['loss_corr_bw']
    eval_sample = cfg['training']['eval_sample']
    vae_beta = cfg['model']['vae_beta']

    trainer = training.Trainer(
        model, optimizer, device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=eval_sample, loss_corr=loss_corr,
        loss_recon=loss_recon, loss_corr_bw=loss_corr_bw,
        vae_beta=vae_beta)

    return trainer


# Generator for final mesh extraction
def get_generator(model, cfg, device):

    #TODO

    return generator


# Datasets
def get_dataset(mode, cfg, return_idx=False, return_category=False):

    #TODO
    dataset = []

    return dataset


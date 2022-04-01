import yaml
import models, training
from models import velocity_color_field
import torch
import torch.distributions as dist
from torch import nn
import os
import models.data
from torchvision import transforms


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

#############################################
#### For new field (VelocityColorField) #####
#############################################

def get_encoder_color(cfg, device, dataset=None, c_dim=0):
    pass

def get_decoder_color(cfg, device, dim, c_dim, z_dim):
    pass

def get_velocity_color_field(cfg, device, dim, c_dim, z_dim):
    
    velocity_color_field = models.velocity_color_field.VelocityColorField()
    return velocity_color_field

#############################################

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


# from im2mesh/oflow/config.py
def get_transforms(cfg):
    ''' Returns transform objects.

    Args:
        cfg (yaml config): yaml config object
    '''
    n_pcl = cfg['data']['n_training_pcl_points']
    n_pt = cfg['data']['n_training_points']
    n_pt_eval = cfg['training']['n_eval_points']

    transf_pt = models.data.SubsamplePoints(n_pt)
    transf_pt_val = models.data.SubsamplePointsSeq(n_pt_eval, random=False)
    transf_pcl_val = models.data.SubsamplePointcloudSeq(n_pt_eval, random=False)
    transf_pcl = models.data.SubsamplePointcloudSeq(n_pcl, connected_samples=True)

    return transf_pt, transf_pt_val, transf_pcl, transf_pcl_val


# from im2mesh/oflow/config.py
def get_data_fileds(mode, cfg):
    ''' Returns data fields.

    Args:
        mode (str): mode (train|val|test)
        cfg (yaml config): yaml config object
    '''
    fields = {}
    seq_len = cfg['data']['length_sequence']
    p_folder = cfg['data']['points_iou_seq_folder']
    pcl_folder = cfg['data']['pointcloud_seq_folder']
    mesh_folder = cfg['data']['mesh_seq_folder']
    generate_interpolate = cfg['generation']['interpolate']
    correspondence = cfg['generation']['correspondence']
    unpackbits = cfg['data']['points_unpackbits']

    # Transformation
    transf_pt, transf_pt_val, transf_pcl, transf_pcl_val = get_transforms(cfg)

    # Fields
    pts_iou_field = models.data.PointsSubseqField
    pts_corr_field = models.data.PointCloudSubseqField

    if mode == 'train':
        if cfg['model']['loss_recon']:
            fields['points'] = pts_iou_field(p_folder, transform=transf_pt,
                                             seq_len=seq_len,
                                             fixed_time_step=0,
                                             unpackbits=unpackbits)
            fields['points_t'] = pts_iou_field(p_folder,
                                               transform=transf_pt,
                                               seq_len=seq_len,
                                               unpackbits=unpackbits)
        # Connectivity Loss:
        if cfg['model']['loss_corr']:
            fields['pointcloud'] = pts_corr_field(pcl_folder,
                                                  transform=transf_pcl,
                                                  seq_len=seq_len)
    elif mode == 'val':
        fields['points'] = pts_iou_field(p_folder, transform=transf_pt_val,
                                         all_steps=True, seq_len=seq_len,
                                         unpackbits=unpackbits)
        fields['points_mesh'] = pts_corr_field(pcl_folder,
                                               transform=transf_pcl_val,
                                               seq_len=seq_len)
    elif mode == 'test' and (generate_interpolate or correspondence):
        fields['mesh'] = models.data.MeshSubseqField(mesh_folder, seq_len=seq_len,
                                              only_end_points=True)
    return fields


# Datasets
def get_dataset(mode, cfg, return_idx=False, return_category=False):

    #TODO
    # temp config vars
    dataset_type = 'Faces'


    # load config vars
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }
    split = splits[mode]



    if dataset_type == 'Faces':
        fields = get_data_fileds(mode, cfg)
        inputs_field = get_inputs_field(mode, cfg)

        if inputs_field is not None:
            fields['inputs'] = inputs_field
        
        if return_idx:
            fields['idx'] = models.data.IndexField()
        
        if return_category:
            fields['category'] = models.data.CategoryField()
        
        dataset = models.data.FacesDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            length_sequence=cfg['data']['length_sequence'],
            n_files_per_sequence=cfg['data']['n_files_per_sequence'],
            offset_sequence=cfg['data']['offset_sequence'],
            ex_folder_name=cfg['data']['pointcloud_seq_folder'])

    else:
        raise ValueError ('Invalid datase')

    return dataset


def get_inputs_field(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    input_type = cfg['data']['input_type']

    if input_type is None:
        inputs_field = None
    elif input_type == 'img_seq':
        if mode == 'train' and cfg['data']['img_augment']:
            resize_op = transforms.RandomResizedCrop(
                cfg['data']['img_size'], (0.75, 1.), (1., 1.))
        else:
            resize_op = transforms.Resize((cfg['data']['img_size']))

        transform = transforms.Compose([
            resize_op, transforms.ToTensor(),
        ])

        if mode == 'train':
            random_view = True
        else:
            random_view = False

        inputs_field = models.data.ImageSubseqField(
            cfg['data']['img_seq_folder'],
            transform, random_view=random_view)
    elif input_type == 'pcl_seq':
        connected_samples = cfg['data']['input_pointcloud_corresponding']
        transform = transforms.Compose([
            models.data.SubsamplePointcloudSeq(
                cfg['data']['input_pointcloud_n'],
                connected_samples=connected_samples),
            models.data.PointcloudNoise(cfg['data']['input_pointcloud_noise'])
        ])
        inputs_field = models.data.PointCloudSubseqField(
            cfg['data']['pointcloud_seq_folder'],
            transform, seq_len=cfg['data']['length_sequence'])
    elif input_type == 'end_pointclouds':
        transform = models.data.SubsamplePointcloudSeq(
            cfg['data']['input_pointcloud_n'],
            connected_samples=cfg['data']['input_pointcloud_corresponding'])

        inputs_field = models.data.PointCloudSubseqField(
            cfg['data']['pointcloud_seq_folder'],
            only_end_points=True, seq_len=cfg['data']['length_sequence'],
            transform=transform)
    elif input_type == 'idx':
        inputs_field = models.data.IndexField()
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_field
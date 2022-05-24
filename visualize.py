import torch
import torch.optim as optim
import config
from models.data.core import worker_init_fn
from utils.checkpoint import CheckpointIO
from tqdm import tqdm
import models.data as data
import numpy as np


import open3d as o3d

# Visualizing result
# Generate pointcloud for particular time t from t0 data

# Config
cfg = config.load_config('default.yaml')
is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")
out_dir = cfg['training']['out_dir']
batch_size = cfg['visualize']['batch_size']

# Device Setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("### Device Check list ###")
print("GPU available?:", torch.cuda.is_available())
if torch.cuda.is_available():
    device_number = torch.cuda.current_device()
    print("Device number:", device_number)
    print("Is device?:", torch.cuda.device(device_number))
    print("Device count?:", torch.cuda.device_count())
    print("Device name?:", torch.cuda.get_device_name(device_number))
    print("### ### ### ### ### ###\n\n")


# Dataset
# TODO
vis_dataset = config.get_dataset('vis', cfg)

# Dataloader
# TODO
vis_loader= torch.utils.data.DataLoader(vis_dataset, batch_size=batch_size, num_workers=0, shuffle=False,
                                        collate_fn=data.collate_remove_none,
                                        worker_init_fn=data.worker_init_fn)

# Model
model = config.get_model(cfg, device=device, dataset=vis_dataset)
model = model.to(device)

# Optimizer
lr = cfg['training']['learning_rate']
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
    load_dict = checkpoint_io.load('model_best.pt')
except FileExistsError:
    load_dict = dict()

model.eval()



for it, data in enumerate(tqdm(vis_loader)):
    # TODO
    #print("debug: ", data)
    #print(data.keys())

    #print(data['colored_points'].shape)
    

    # Shape should be (batch x num_pts x 6)
    #colored_points_t0 = torch.index_select(data['colored_points'], 1, torch.Tensor([0]).int()).to(device)
    colored_points_t0 = data.get('colored_points', torch.empty(1, 1, 0)).to(device)

    c_s, c_s_color, c_t, c_t_color = model.encode_inputs(colored_points_t0)
    q_z, q_z_color, q_z_t, q_z_t_color = model.infer_z(colored_points_t0, c=c_t, c_color=c_t_color, data=data)
    z, z_t = q_z.rsample(), q_z_t.rsample()
    z_color, z_t_color = q_z_color.rsample(), q_z_t_color.rsample()




    #print(colored_points_t0.shape)
    #colored_points_t0 = colored_points_t0[:, 0, :, :]
    #num_batch, _, num_pts, dim_pts = colored_points_t0.shape
    #colored_points_t0.reshape((num_batch, num_pts, dim_pts))
    #colored_points_t0 = data.get('').to(device)
    # Shape should be 1D tensor
    time = torch.linspace(0, 1, 100).to(device)

    # Output shape
    #   point_pred: (batch x time x num_pts x 3)
    #   color_pred: (batch x time x num_pts x 6)
    # color_pred[:,:,:,3:] to get only color
    point_pred, color_pred = model.transform_to_t(time, colored_points_t0[:, 0], c_t=c_t, z=z_t, z_color=z_color, c_t_color=c_t_color)


    
    # Save to npz?
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    geometry = o3d.geometry.PointCloud()
    vis.add_geometry(geometry)
    
    save_image = True
    
    for target_idx in tqdm(range(100)):
        points = point_pred[0, target_idx].detach().cpu().numpy()
        colors = color_pred[0, target_idx, :, 3:].detach().cpu().numpy()

        for i, c in enumerate(colors):
            if c[0] < 0.1 and c[1] < 0.1 and c[2] < 0.1:
                points[i,:] = np.array([200., 200., 200.])

        #points = np.array([points[i] for i, x in enumerate(colors) if colors[i][2] > 20])
        #colors = np.array([colors[i] for i, x in enumerate(colors) if colors[i][2] > 20])

        geometry.points = o3d.utility.Vector3dVector(points)
        geometry.colors = o3d.utility.Vector3dVector(colors*2)
        geometry.normals = o3d.utility.Vector3dVector(points / np.expand_dims(np.linalg.norm(points, axis=1), axis=1))

        #o3d.visualization.draw_geometries([geometry])
        
        vis.add_geometry(geometry)
        vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        if save_image:
            vis.capture_screen_image("temp/temp_%04d.jpg" % target_idx)
    
    vis.destroy_window()
    
    
    #o3d.visualization.draw_geometries([pcd])  
    break
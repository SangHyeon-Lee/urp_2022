import torch
import torch.optim as optim
import config
from utils.checkpoint import CheckpointIO
from tqdm import tqdm

# Visualizing result
# Generate pointcloud for particular time t from t0 data

# Config
cfg = config.load_config('default.yaml')
is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")
out_dir = cfg['training']['out_dir']
# Dataset
# TODO
vis_dataset = None

# Dataloader
# TODO
vis_loader= None

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
    load_dict = checkpoint_io.load('model_best_2.pt')
except FileExistsError:
    load_dict = dict()

model.eval()

for it, data in enumerate(tqdm(vis_loader)):
    # TODO
    # Shape should be (batch x num_pts x 6)
    colored_points_t0 = data.get('').to(device)
    # Shape should be 1D tensor
    time = None

    # Output shape
    #   point_pred: (batch x time x num_pts x 3)
    #   color_pred: (batch x time x num_pts x 6)
    # color_pred[:,:,:,3:] to get only color
    point_pred, color_pred = model.transform_to_t(time, colored_points_t0)

    # Save to npz?
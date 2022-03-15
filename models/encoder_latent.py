import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.layers import ResnetBlockFC


def maxpool(x, dim=-1, keepdim=False):
    ''' Performs a maximum pooling operation.

    Args:
        x (tensor): input tensor
        dim (int): dimension of which the pooling operation is performed
        keepdim (bool): whether to keep the dimension
    '''
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

class PointNet(nn.Module):
    ''' Latent PointNet-based encoder class.

    It maps the inputs together with an (optional) conditioned code c
    to means and standard deviations.

    Args:
        dim (int): dimension of input points
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_dim (int): dimension of hidden size
        n_blocks (int): number of ResNet-based blocks
    '''

    def __init__(self, z_dim=128, c_dim=128, dim=51, hidden_dim=128,
                 n_blocks=3, **kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.dim = dim
        self.n_blocks = n_blocks

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)])

        if self.c_dim != 0:
            self.c_layers = nn.ModuleList(
                [nn.Linear(c_dim, 2*hidden_dim) for i in range(n_blocks)])

        self.actvn = nn.ReLU()
        self.pool = maxpool

        self.fc_mean = nn.Linear(hidden_dim, z_dim)
        self.fc_logstd = nn.Linear(hidden_dim, z_dim)

    def forward(self, inputs, c=None, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            inputs (tensor): inputs
            c (tensor): latent conditioned code c
        '''
        batch_size, n_t, T, _ = inputs.shape

        # Reshape input is necessary
        if self.dim == 3:
            inputs = inputs[:, 0]
        else:
            inputs = inputs.transpose(
                1, 2).contiguous().view(batch_size, T, -1)
        # output size: B x T X F
        net = self.fc_pos(inputs)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net_c = self.c_layers[i](c).unsqueeze(1)
                net = net + net_c

            net = self.blocks[i](net)
            if i < self.n_blocks - 1:
                pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
                net = torch.cat([net, pooled], dim=2)

        net = self.pool(net, dim=1)

        mean = self.fc_mean(net)
        logstd = self.fc_logstd(net)

        return mean, logstd

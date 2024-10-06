import torch
from torch_geometric.nn import AGNNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F


class Model(torch.nn.Module):
    """
    The graph attentional propagation layer from the "Attention-based
    Graph Neural Network for Semi-Supervised Learning"
    <https://arxiv.org/abs/1803.03735>
    """

    def __init__(self, args, num_features= 1280):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(num_features, 512)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.fc2 = torch.nn.Linear(512, 256)

        # self.fc3 = torch.nn.Linear(2 * 16, 1)
        self.fc3 = torch.nn.Linear(2 * 512, args['num_classes'])

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.fc1(x))
        x = self.prop1(x, edge_index)
        x = self.prop2(x, edge_index)
        x = F.relu(self.fc2(x))

        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # x = torch.sigmoid(self.fc3(x)).squeeze(1)
        x = self.fc3(x)

        return x
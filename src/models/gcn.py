import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np





class MyChannelAttention(nn.Module):
    def __init__(self,channels_num):
        super(MyChannelAttention, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(channels_num, 4),
            nn.Tanh(),
            nn.Linear(4, channels_num),
        )

    def forward(self, inputs):
        inputs = inputs.permute(0, 1, 3, 2)
        cha_attention = self.channel_attention(inputs)
        cha_attention = torch.mean(cha_attention, dim=0)
        return cha_attention

class MyGraphConvolution(nn.Module):
    def __init__(self,channels_num,graph_convolution_kernel=5,is_channel_attention=True):
        super(MyGraphConvolution, self).__init__()
        self.is_channel_attention = is_channel_attention
        # 导入邻接矩阵
        adjacency = np.zeros((32, 32))  # solve the adjacency matrix (N*N, eg. 64*64)
        edges = np.load('/home/eeg-music-challenge-icassp-2025-baselines-master/src/models/edges.npy')
        for x, y in edges:
            adjacency[x][y] = 1
            adjacency[y][x] = 1
        adjacency = np.sign(adjacency + np.eye(channels_num))
        adjacency = np.sum(adjacency, axis=0) * np.eye(64) - adjacency
        e_vales, e_vets = np.linalg.eig(adjacency)

        # 计算模型需要的参数
        self.adj = None
        self.e_vales = torch.tensor(e_vales, dtype=torch.float32)
        self.e_vets = torch.tensor(e_vets, dtype=torch.float32)

        # 计算 图卷积 的卷积核
        self.graph_kernel = nn.Parameter(torch.randn(graph_convolution_kernel, 1, channels_num))
        self.graph_kernel = self.graph_kernel * torch.eye(channels_num)
        self.graph_kernel = torch.matmul(torch.matmul(self.e_vets, self.graph_kernel), self.e_vets.t())
        self.graph_kernel = self.graph_kernel.unsqueeze(0)

        # 添加 注意力 机制
        self.graph_channel_attention = MyChannelAttention(channels_num=channels_num) if is_channel_attention else None

    def forward(self, x):
        adj = self.graph_kernel

        # 通道注意力网络
        if self.is_channel_attention:
            cha_attention = self.graph_channel_attention(x)
            adj = cha_attention * adj

        # 卷积过程
        x = torch.matmul(adj, x)
        x = F.relu(x)

        return x

class Model(nn.Module):
    def __init__(self, args, channels_num=32,sample_len=1280,graph_layer_num=10,is_channel_attention=True):
        super(Model, self).__init__()
        self.permute = lambda x: x.permute(0, 2, 1)
        self.reshape = lambda x: x.view(x.size(0), 1, channels_num, sample_len)
        self.batch_norm1 = nn.BatchNorm2d(1)
        self.graph_convs = nn.ModuleList([MyGraphConvolution(channels_num,is_channel_attention=is_channel_attention) for _ in range(graph_layer_num)])
        self.batch_norm2 = nn.ModuleList([nn.BatchNorm2d(1) for _ in range(graph_layer_num)])
        self.permute2 = lambda x: x.permute(0, 1, 3, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, sample_len))
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(channels_num, 8)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(8, args['num_classes'])

    def forward(self, x):
        x = self.permute(x)
        x = self.reshape(x)
        x = self.batch_norm1(x)
        for graph_conv, batch_norm in zip(self.graph_convs, self.batch_norm2):
            x = graph_conv(x)
            x = batch_norm(x)
        x = self.permute2(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = torch.tanh(self.fc1(x))
        x = self.dropout2(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x

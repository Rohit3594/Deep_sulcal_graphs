# Author: Rohit YADAV

import os
import networkx as nx
import numpy as np
import pandas as pd
import random
from torch_geometric.utils.convert import from_networkx
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.nn.functional import one_hot
from sklearn.preprocessing import OneHotEncoder
from torch.nn import Linear
import torch.nn.functional as F
from math import ceil
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
import torch_geometric as pyg
from torch_geometric.nn import GraphConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import TopKPooling
from torch_geometric.data import Data
from torch_geometric.loader import DenseDataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.nn import GCNConv, DenseGraphConv



class mincutnet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=16):
        super(mincutnet, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        num_of_centers =  20
        self.pool1 = Linear(hidden_channels, num_of_centers) # The degree of the node belonging to any of the centers
        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch): # x torch.Size([661, 3]),  data.batch  torch.Size([783])
        node_emb = F.relu(self.conv1(x, edge_index))  #x torch.Size([661, 32])
        x, mask = to_dense_batch(node_emb, batch) #now x torch.Size([1, 661, 32]) ; mask torch.Size([20, 122])
        adj = to_dense_adj(edge_index, batch) # adj torch.Size([1, 661, 661])
        s = self.pool1(x) # s torch.Size([1, 661, 20])
        x, adj, mincut_loss, ortho_loss = dense_mincut_pool(x, adj, s, mask) # x torch.Size([1, 20, 32]),  adj torch.Size([1, 20, 20])
        x = self.conv2(x, adj) #x torch.Size([1, 20, 32])
        g_emb = x.mean(dim=1) # x torch.Size([1, 32])
        x = F.relu(self.lin1(x)) # x torch.Size([1, 32])
        out = self.lin2(x) #x torch.Size([1, 2])
        
        return out, g_emb, node_emb, s

model = mincutnet(num_node_features, 2)
print(model)

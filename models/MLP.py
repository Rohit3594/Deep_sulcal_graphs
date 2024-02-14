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


# Create MLP Model

class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=16):        
        super(MLP, self).__init__()
        torch.manual_seed(12345)
        
        
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)
        

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        
        emb = self.lin1(x)
        x = emb.relu()
        emb = self.lin2(x)
        x = emb.relu()
        g_emb = global_mean_pool(x, batch)
        out = self.lin3(g_emb)
        
        return out, g_emb, emb, x

    
model = MLP(num_node_features, 2)
print(model)

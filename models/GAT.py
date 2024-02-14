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


# Create GAT Model

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):        
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.in_head = 2
        self.out_head = 1
        
        self.emb_dim = 16
        
        
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=self.in_head)
        self.conv2 = GATConv(hidden_channels*self.in_head, hidden_channels*2,concat=False)
        self.conv3 = GATConv(hidden_channels*2, self.emb_dim, concat=False, dropout=0.6)
        self.lin = Linear(self.emb_dim, 2)
        

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x,a1 = self.conv1(x, edge_index,return_attention_weights=True)
        x = x.relu()
        x,a2 = self.conv2(x, edge_index,return_attention_weights=True)
        x = x.relu()
        emb,a3 = self.conv3(x, edge_index,return_attention_weights=True)

        # 2. Readout layer
        g_emb = global_mean_pool(emb, batch)  # [batch_size, hidden_channels] TopKPooling
        #g_emb = TopKPooling(emb, batch)

        # 3. Apply a final classifier
        out = F.dropout(g_emb, p=0.5, training=self.training)
        out = self.lin(out)
        
        return out, g_emb, emb, [a1,a2,a3]

model = GAT(hidden_channels=16)
print(model)

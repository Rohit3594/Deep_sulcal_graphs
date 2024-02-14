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



class topk(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, embed_dim=16):
        super(topk, self).__init__()
        
        self.in_head = 1
        
        #self.conv1 = GATConv(num_node_features, embed_dim, heads=self.in_head,dropout=0.2)
        
        self.conv1 = GraphConv(num_node_features, embed_dim)
        self.pool1 = TopKPooling(embed_dim, ratio=0.5,nonlinearity=torch.sigmoid)
        self.conv2 = GraphConv(embed_dim, embed_dim)
        self.pool2 = TopKPooling(embed_dim, ratio=0.5,nonlinearity=torch.sigmoid)

        self.lin1 = torch.nn.Linear(embed_dim * 2, embed_dim)
        self.bn1 = torch.nn.BatchNorm1d(embed_dim)

        self.lin2 = torch.nn.Linear(embed_dim, out_channels)


    def forward(self, x, edge_index, batch):
        

        x = F.relu(self.conv1(x, edge_index))
        
#         x, attn_weights = self.conv1(x, edge_index,return_attention_weights=True)
#         x = x.relu()
        
        x, edge_index_1, edge_attr_1, batch_1, perm_1, score_1 = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch_1), gap(x, batch_1)], dim=1)

        x = F.relu(self.conv2(x, edge_index_1))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index_1, None, batch_1)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)


        #g_emb = torch.cat([x1,x2], dim=1)
        
        g_emb32 = x1 + x2
        g_emb = self.bn1(F.relu(self.lin1(g_emb32)))
        x = F.dropout(g_emb, p=0.5, training=self.training)
        
        x= F.dropout(x, p=0.5, training=self.training)
        out = F.log_softmax(self.lin2(x), dim=-1)
        
        return out, g_emb, self.pool1.weight, score_1
        

model = topk(num_node_features, 2)
print(model)

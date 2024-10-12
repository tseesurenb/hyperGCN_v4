'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import degree

# HyperGCN Convolutional Layer
class hyperGCN(MessagePassing):
    def __init__(self, edge_weight_func = 'exp', **kwargs):  
        super().__init__(aggr='add')
        
        self.edge_weight_func = edge_weight_func
        self.graph_norms = None
        self.edge_attrs = None
            
    def forward(self, x, edge_index, edge_attrs):
        
        if self.graph_norms is None:

            # Compute normalization
            from_, to_ = edge_index
            deg = degree(to_, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
          
            self.graph_norms = norm
            
            if self.edge_weight_func == 'exp' and edge_attrs != None:
              self.edge_attrs = torch.exp(edge_attrs)
            elif self.edge_weight_func == 'sigmoid' and edge_attrs != None:
              self.edge_attrs = torch.sigmoid(edge_attrs)
            elif self.edge_weight_func == 'tanh' and edge_attrs != None:
              self.edge_attrs = torch.tanh(edge_attrs)
            else:
              self.edge_attrs = None
        
        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=self.graph_norms, attr = self.edge_attrs)

    def message(self, x_j, norm, attr):
        if attr != None:
            return norm.view(-1, 1) * (x_j * attr.view(-1, 1))
        else:
            return norm.view(-1, 1) * x_j   

# LightGCN Convolutional Layer     
class LightGCNConv(MessagePassing):
    def __init__(self, **kwargs):  
        super().__init__(aggr='add')
            
    def forward(self, x, edge_index, edge_attrs):
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
      
# NGCF Convolutional Layer
class NGCFConv(MessagePassing):
  def __init__(self, emb_dim, dropout, bias=True, **kwargs):  
    super(NGCFConv, self).__init__(aggr='add', **kwargs)

    self.dropout = dropout

    self.lin_1 = nn.Linear(emb_dim, emb_dim, bias=bias)
    self.lin_2 = nn.Linear(emb_dim, emb_dim, bias=bias)

    self.init_parameters()

  def init_parameters(self):
    nn.init.xavier_uniform_(self.lin_1.weight)
    nn.init.xavier_uniform_(self.lin_2.weight)

  def forward(self, x, edge_index, edge_attrs):
    # Compute normalization
    from_, to_ = edge_index
    deg = degree(to_, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

    # Start propagating messages
    out = self.propagate(edge_index, x=(x, x), norm=norm)

    # Perform update after aggregation
    out += self.lin_1(x)
    out = F.dropout(out, self.dropout, self.training)
    return F.leaky_relu(out)

  def message(self, x_j, x_i, norm):
    return norm.view(-1, 1) * (self.lin_1(x_j) + self.lin_2(x_j * x_i)) 

  
class RecSysGNN(nn.Module):
  def __init__(
      self,
      emb_dim, 
      n_layers,
      n_users,
      n_items,
      model, # 'NGCF' or 'LightGCN' or 'hyperGCN'
      dropout=0.1, # Only used in NGCF
      edge_weight_func = None
  ):
    super(RecSysGNN, self).__init__()

    assert (model == 'NGCF' or model == 'lightGCN') or model == 'hyperGCN', 'Model must be NGCF or LightGCN or hyperGCN'
    self.model = model
    self.n_users = n_users
    self.n_items = n_items
    self.n_layers = n_layers
    self.emb_dim = emb_dim
    
    self.embedding = nn.Embedding(self.n_users + self.n_items, self.emb_dim)
        
    if self.model == 'NGCF':
      self.convs = nn.ModuleList(NGCFConv(self.emb_dim, dropout=dropout) for _ in range(self.n_layers))
    elif self.model == 'lightGCN':
      self.convs = nn.ModuleList(LightGCNConv() for _ in range(self.n_layers))
    elif self.model == 'hyperGCN':
      self.convs = nn.ModuleList(hyperGCN(edge_weight_func=edge_weight_func) for _ in range(self.n_layers))
    else:
      raise ValueError('Model must be NGCF, LightGCN or hyperGCN')

    self.init_parameters()

  def init_parameters(self):
    
    if self.model == 'NGCF':
      nn.init.xavier_uniform_(self.embedding.weight, gain=1)
    else:
      # Authors of LightGCN report higher results with normal initialization
      nn.init.normal_(self.embedding.weight, std=0.1)

  def forward(self, edge_index, edge_attrs):
    emb0 = self.embedding.weight
    embs = [emb0]

    emb = emb0
    for conv in self.convs:
      emb = conv(x=emb, edge_index=edge_index, edge_attrs=edge_attrs)
      embs.append(emb)
      
    out = (
      torch.cat(embs, dim=-1) if self.model == 'NGCF' 
      else torch.mean(torch.stack(embs, dim=0), dim=0)
    )
        
    return emb0, out


  def encode_minibatch(self, users, pos_items, neg_items, edge_index, edge_attrs):
    emb0, out = self(edge_index, edge_attrs)
    
    return (
        out[users], 
        out[pos_items], 
        out[neg_items],
        emb0[users],
        emb0[pos_items],
        emb0[neg_items]
    )
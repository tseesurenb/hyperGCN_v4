'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import torch
import torch.nn.functional as F

import numpy as np

from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import degree
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from world import config

def edge_attr_drop(edge_index, edge_attr, modify_prob=0.2, mode=1):

    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) < modify_prob

    # Modify the selected edge attributes to 1
    new_edge_attr = edge_attr.clone()
    if mode == 1:
      new_edge_attr[mask] = 1.0
    else:
      new_edge_attr[mask] = 0.0

    return new_edge_attr

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class HyperGCNAttention(MessagePassing):
    def __init__(self, edge_attr_mode='exp', attr_drop=0.2, self_loop=False, hidden_dim=64, **kwargs):
        super().__init__(aggr='add')  # Aggregation mode: add
        
        self.edge_attr_mode = edge_attr_mode
        self.attr_drop = attr_drop
        self.add_self_loops = self_loop
        self.edge_attrs = None
        
        # Trainable attention weights
        #self.attn_weight = torch.nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.attn_weight = torch.nn.Parameter(torch.Tensor(hidden_dim, hidden_dim).double())

        self.attn_bias = torch.nn.Parameter(torch.Tensor(1).double())
        torch.nn.init.xavier_uniform_(self.attn_weight)  # Xavier initialization
        torch.nn.init.zeros_(self.attn_bias)

    def forward(self, x, edge_index, edge_attrs=None, scale=1.0):
        # Process edge attributes
        if edge_attrs is not None:
            if self.edge_attr_mode == 'exp':
                self.edge_attrs = torch.exp(scale * edge_attrs)
            elif self.edge_attr_mode == 'sig':
                self.edge_attrs = torch.sigmoid(edge_attrs)
            elif self.edge_attr_mode == 'tan':
                self.edge_attrs = torch.tanh(edge_attrs)
            else:
                self.edge_attrs = edge_attrs
        else:
            self.edge_attrs = None

        # Start propagating messages
        return self.propagate(edge_index, x=x, edge_attr=self.edge_attrs)

    def message(self, x_i, x_j, edge_index, edge_attr):
        """
        x_i: Node features of target nodes
        x_j: Node features of source nodes
        edge_index: Edge indices
        edge_attr: Edge attributes
        """
        # Compute attention scores
        attn_score = (x_i @ self.attn_weight) * x_j
        if edge_attr is not None:
            attn_score = attn_score + edge_attr.view(-1, 1)
        attn_score = attn_score.sum(dim=-1) + self.attn_bias
        attn_score = F.leaky_relu(attn_score, negative_slope=0.2)

        # Normalize attention scores using softmax
        attn_score = softmax(attn_score, edge_index[0])

        # Apply attention scores
        return attn_score.view(-1, 1) * x_j

    def update(self, aggr_out):
        # Optional: Add additional transformations after aggregation
        return aggr_out


# HyperGCN Convolutional Layer
class hyperGCN(MessagePassing):
    def __init__(self, edge_attr_mode = 'exp', attr_drop = 0.2, self_loop = False, **kwargs):  
        super().__init__(aggr='add')
        
        self.edge_attr_mode = edge_attr_mode
        self.graph_norms = None
        self.edge_attrs = None
        self.add_self_loops = self_loop
        self.attr_drop = attr_drop # make edge_attr 0 for 20% of edges
            
    def forward(self, x, edge_index, edge_attrs, scale):
        
        if self.graph_norms is None:
          # Compute normalization  
          self.edge_index_norm = gcn_norm(edge_index=edge_index, add_self_loops=self.add_self_loops)
          self.graph_norms = self.edge_index_norm[1]
              
          if self.edge_attr_mode == 'exp' and edge_attrs != None:            
            self.edge_attrs = torch.exp(scale * edge_attrs)
            #self.edge_attrs = torch.exp(edge_attrs)
          elif self.edge_attr_mode == 'sig' and edge_attrs != None:
            self.edge_attrs = torch.sigmoid(edge_attrs)
          elif self.edge_attr_mode == 'tan' and edge_attrs != None:
            self.edge_attrs = torch.tanh(edge_attrs)
                
        #self.edge_attrs = edge_attr_drop(edge_index, self.edge_attrs, self.attr_drop)
        
        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=self.graph_norms, attr = self.edge_attrs)

    def message(self, x_j, norm, attr):
        if attr != None:
            return norm.view(-1, 1) * (x_j * attr.view(-1, 1))
        else:
            return norm.view(-1, 1) * x_j
          
          
# HyperGCN Convolutional Layer
class hyperGAT(MessagePassing):
    def __init__(self, edge_attr_mode = 'exp', attr_drop = 0.0, self_loop = False, device = 'cpu', **kwargs):  
        super().__init__(aggr='add')
        
        self.edge_attr_mode = edge_attr_mode
        self.graph_norms = None
        self.edge_attrs = None
        self.add_self_loops = self_loop
        self.attr_drop = attr_drop # make edge_attr 0 for 20% of edges
        num_heads = 4
        self.num_heads = num_heads
        self.head_dim = 32
        self.device = device
        
        self.edge_attr_linears = nn.ModuleList([
            nn.Linear(64, self.head_dim) for _ in range(num_heads)
        ]).double().to(self.device)
        
    def compute_multi_head_attention(self, edge_attrs):
      # Compute attention for each head
      attr_heads = []
      for linear in self.edge_attr_linears:
          attr_heads.append(F.softmax(linear(edge_attrs), dim=0))  # Apply softmax per head
      return torch.stack(attr_heads, dim=1)  # Shape: [num_edges, num_heads, head_dim]

    def forward(self, x, edge_index, edge_attrs, scale):
        
        if self.graph_norms is None:
          # Compute normalization  
          self.edge_index_norm = gcn_norm(edge_index=edge_index, add_self_loops=self.add_self_loops)
          self.graph_norms = self.edge_index_norm[1]

          self.edge_attrs = F.leaky_relu(torch.exp(scale * edge_attrs), negative_slope=0.2)
          #self.edge_attrs = softmax(scale * edge_attrs, edge_index[0])

        
        #if self.attr_drop >= 0.0:
        #  edge_attrs = edge_attr_drop(edge_index, edge_attrs, self.attr_drop, mode=config['drop_mode'])
        
        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=self.graph_norms, attr = self.edge_attrs)

    def message(self, x_j, norm, attr):
        # Attended message passing      
        if attr != None:
            return norm.view(-1, 1) * (x_j * attr.view(-1, 1))
        else:
            return norm.view(-1, 1) * x_j

class hyperGAT3(MessagePassing):
    def __init__(self,edge_attr_mode='exp', heads=8, attr_drop=0.0, self_loop = False,  **kwargs):
        super().__init__(aggr='add', **kwargs)

        in_channels = 64
        out_channels = 64
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = attr_drop

        self.weight = nn.Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))


        self.edge_attr_mode = edge_attr_mode

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.att)


    def forward(self, x, edge_index, edge_attrs, scale):
        # Node feature transformation
        x = torch.matmul(x, self.weight).view(-1, self.heads, self.out_channels)

        # Edge feature transformation (optional, based on edge_attr_mode)
        if self.edge_attr_mode == 'exp':
             edge_attrs = torch.exp(edge_attrs)
        # elif self.edge_attr_mode == 'linear':
        #     # ... (apply linear transformation)

        # Attention mechanism
        edge_attrs = F.dropout(edge_attrs, p=self.dropout, training=self.training)
        edge_attrs = (edge_attrs * self.att[:, :, 0]).sum(dim=-1) - (edge_attrs * self.att[:, :, 1]).sum(dim=-1)
        edge_attrs = F.softmax(edge_attrs, dim=1)

        # Message passing
        return self.propagate(edge_index, x=x, edge_attr=edge_attrs)

    def message(self, x_j, edge_attr):
        # Attended message passing
        return edge_attr.view(-1, 1) * x_j
      
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
  

# NGCF Convolutional Layer
class knnNGCFConv(MessagePassing):
  def __init__(self, emb_dim, dropout, bias=True, edge_attr_mode= 'exp', **kwargs):  
    #super(knnNGCFConv, self).__init__(aggr='add', **kwargs)
    super(knnNGCFConv, self).__init__(aggr='add', **kwargs)

    self.dropout = dropout
    self.edge_attr_mode = edge_attr_mode
    self.norm = None
    self.edge_attrs = None

    self.lin_1 = nn.Linear(emb_dim, emb_dim, bias=bias)
    self.lin_2 = nn.Linear(emb_dim, emb_dim, bias=bias)

    self.init_parameters()

  def init_parameters(self):
    nn.init.xavier_uniform_(self.lin_1.weight)
    nn.init.xavier_uniform_(self.lin_2.weight)

  def forward(self, x, edge_index, edge_attrs):
    # Compute normalization
    if self.norm is None:
      from_, to_ = edge_index
      deg = degree(to_, x.size(0), dtype=x.dtype)
      deg_inv_sqrt = deg.pow(-0.5)
      deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
      self.norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

      if self.edge_attr_mode == 'exp':
        self.edge_attrs = torch.exp(edge_attrs)
      elif self.edge_attr_mode == 'sig':
        self.edge_attrs = torch.sigmoid(edge_attrs)
      elif self.edge_attr_mode == 'tan':
        self.edge_attrs = torch.tanh(edge_attrs)
      else:
        self.edge_attrs = None

    # Start propagating messages
    out = self.propagate(edge_index, x=(x, x), norm=self.norm, attr = self.edge_attrs)

    # Perform update after aggregation
    out += self.lin_1(x)
    out = F.dropout(out, self.dropout, self.training)
    return F.leaky_relu(out)

  def message(self, x_j, x_i, norm, attr):
    return norm.view(-1, 1) * (self.lin_1(x_j) + self.lin_2(x_j * x_i)) * attr.view(-1, 1) 


# LightGCN Convolutional Layer     
class LightGCNConv(MessagePassing):
    def __init__(self, **kwargs):  
        super().__init__(aggr='add')
        
        self.norm = None
            
    def forward(self, x, edge_index, edge_attrs, scale):
      
        if self.norm is None:
          # Compute normalization
          from_, to_ = edge_index
          deg = degree(to_, x.size(0), dtype=x.dtype)
          deg_inv_sqrt = deg.pow(-0.5)
          deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
          self.norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=self.norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

      
class RecSysGNN(nn.Module):
  def __init__(
      self,
      emb_dim, 
      n_layers,
      n_users,
      n_items,
      model, # 'NGCF' or 'LightGCN' or 'hyperGCN'
      dropout=0.1, # Only used in NGCF
      attr_drop = 0.0, # Only used in hyperGCN
      edge_attr_mode = None,
      scale = 1.0,
      device = 'cpu',
      self_loop = False
  ):
    super(RecSysGNN, self).__init__()

    assert (model == 'NGCF' or model == 'lightGCN') or model == 'hyperGCN' or model == 'knnNGCF' or model == 'hyperGAT' or model == 'HyperGCNAttention', 'Model must be NGCF or LightGCN or hyperGCN or hyperGAN'
    self.model = model
    self.n_users = n_users
    self.n_items = n_items
    self.n_layers = n_layers
    self.emb_dim = emb_dim
    
    # Initialize scale parameters for users and items
    self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float64))
  
    self.embedding = nn.Embedding(self.n_users + self.n_items, self.emb_dim, dtype=torch.float64)
        
    if self.model == 'NGCF':
      self.convs = nn.ModuleList(NGCFConv(self.emb_dim, dropout=dropout) for _ in range(self.n_layers))
    elif self.model == 'knnNGCF':
      self.convs = nn.ModuleList(knnNGCFConv(self.emb_dim, dropout=dropout, edge_attr_mode=edge_attr_mode) for _ in range(self.n_layers))
    elif self.model == 'lightGCN':
      self.convs = nn.ModuleList(LightGCNConv() for _ in range(self.n_layers))
    elif self.model == 'hyperGCN':
      self.convs = nn.ModuleList(hyperGCN(edge_attr_mode=edge_attr_mode, attr_drop=attr_drop, self_loop=self_loop) for _ in range(self.n_layers))
    elif self.model == 'hyperGAT':
      self.convs = nn.ModuleList(hyperGAT(edge_attr_mode=edge_attr_mode, attr_drop=attr_drop, self_loop=self_loop, device=device) for _ in range(self.n_layers))
    elif self.model == 'HyperGCNAttention':
      self.convs = nn.ModuleList(HyperGCNAttention(edge_attr_mode=edge_attr_mode, attr_drop=attr_drop, self_loop=self_loop) for _ in range(self.n_layers))
    else:
      raise ValueError('Model must be NGCF, LightGCN or hyperGCN')

    self.init_parameters()

  def init_parameters(self):
        
    if self.model == 'NGCF':
      nn.init.xavier_uniform_(self.embedding.weight, gain=1)
    else:
      nn.init.normal_(self.embedding.weight, std=0.1)

  def forward(self, edge_index, edge_attrs):
    
    emb0 = self.embedding.weight
    embs = [emb0]
     
    emb = emb0
    for conv in self.convs:
      emb = conv(x=emb, edge_index=edge_index, edge_attrs=edge_attrs, scale = self.scale)
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
        emb0[neg_items],
    )
    
  def predict(self, users, items, edge_index, edge_attrs):
    emb0, out = self(edge_index, edge_attrs)    
    return torch.matmul(out[users], out[items].t())


# define a function that compute all users scoring for all items and then save it to a file. later, I can be able to get top-k for a user by user_id
def get_all_predictions(model, edge_index, edge_attrs, device):
    model.eval()
    users = torch.arange(model.n_users).to(device)
    items = torch.arange(model.n_items).to(device)
    predictions = model.predict(users, items, edge_index, edge_attrs)
    return predictions.cpu().detach().numpy()
  
# define a function that get top-k items for a user by user_id after sorting the predictions
def get_top_k(user_id, predictions, k):
    return np.argsort(predictions[user_id])[::-1][:k]
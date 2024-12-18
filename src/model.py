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
from torch_geometric.utils import softmax
import sys

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
        
    def forward(self, x, edge_index, edge_attrs, scale):
        
        if self.graph_norms is None:
          # Compute normalization  
          #self.edge_index_norm = gcn_norm(edge_index=edge_index, add_self_loops=self.add_self_loops)
          
          from_, to_ = edge_index
          deg = degree(to_, x.size(0), dtype=x.dtype)
          deg_inv_sqrt = deg.pow(-0.5)
          deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
          norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
          self.graph_norms = norm
        
          if config['e_attr_mode'] == 'exp' and edge_attrs != None:  
            self.edge_attrs = torch.exp(scale * edge_attrs)
          elif config['e_attr_mode'] == 'smax' and edge_attrs != None:
            self.edge_attrs = softmax(edge_attrs, to_) # sofmax over all edges from the source nodes to the target nodes
            norm = norm = deg_inv_sqrt[from_] # only normalize the source nodes
            self.graph_norms = norm
          elif config['e_attr_mode'] == 'raw' and edge_attrs != None:
            self.edge_attrs = edge_attrs
          elif config['e_attr_mode'] == 'none' and edge_attrs != None:
            self.edge_attrs = None
          else:
            print('Invalid edge_attr_mode')
                  
        if self.attr_drop >= 0.0 and self.edge_attrs != None:
          self.edge_attrs = edge_attr_drop(edge_index, self.edge_attrs, self.attr_drop, mode=config['drop_mode'])
        
        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=self.graph_norms, attr = self.edge_attrs)

    def message(self, x_j, norm, attr):
        # Attended message passing      
        if attr != None:
            return norm.view(-1, 1) * (x_j * attr.view(-1, 1))
        else:
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

  def forward(self, x, edge_index, edge_attrs, scale):
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
class hyperNGCF(MessagePassing):
  def __init__(self, emb_dim, dropout, bias=True, edge_attr_mode= 'exp', **kwargs):
    super(hyperNGCF, self).__init__(aggr='add', **kwargs)

    self.dropout = dropout
    self.edge_attr_mode = edge_attr_mode
    self.norm = None
    self.edge_attrs = None

    self.lin_1 = nn.Linear(emb_dim, emb_dim, bias=bias, dtype=torch.float32)
    self.lin_2 = nn.Linear(emb_dim, emb_dim, bias=bias, dtype=torch.float32)

    self.init_parameters()

  def init_parameters(self):
    nn.init.xavier_uniform_(self.lin_1.weight)
    nn.init.xavier_uniform_(self.lin_2.weight)

  def forward(self, x, edge_index, edge_attrs, scale):
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
      elif self.edge_attr_mode == 'raw':
        self.edge_attrs = edge_attrs
      else:
        print('Invalid edge_attr_mode')

    # Start propagating messages
    out = self.propagate(edge_index, x=(x, x), norm=self.norm, attr = self.edge_attrs)

    # Perform update after aggregation
    out += self.lin_1(x)
    out = F.dropout(out, self.dropout, self.training)
    return F.leaky_relu(out)

  def message(self, x_j, x_i, norm, attr):    
    return norm.view(-1, 1) * (self.lin_1(x_j) + self.lin_2(x_j * x_i)) * attr.view(-1, 1) 


# LightGCN Convolutional Layer     
class lightGCN(MessagePassing):
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

    assert (model == 'NGCF' or model == 'lightGCN') or model == 'hyperGCN' or model == 'hyperNGCF' or model == 'hyperGAT', 'Model must be NGCF or LightGCN or hyperGCN or hyperGAT'
    self.model = model
    self.n_users = n_users
    self.n_items = n_items
    self.n_layers = n_layers
    self.emb_dim = emb_dim
    
    # Initialize scale parameters for users and items
    self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
  
    self.embedding = nn.Embedding(self.n_users + self.n_items, self.emb_dim, dtype=torch.float32)
        
    if self.model == 'NGCF':
      self.convs = nn.ModuleList(NGCFConv(self.emb_dim, dropout=dropout) for _ in range(self.n_layers))
    elif self.model == 'hyperNGCF':
      self.convs = nn.ModuleList(hyperNGCF(self.emb_dim, dropout=dropout, edge_attr_mode=edge_attr_mode) for _ in range(self.n_layers))
    elif self.model == 'lightGCN':
      self.convs = nn.ModuleList(lightGCN() for _ in range(self.n_layers))
    elif self.model == 'hyperGCN':
      self.convs = nn.ModuleList(hyperGCN(edge_attr_mode=edge_attr_mode, attr_drop=attr_drop, self_loop=self_loop) for _ in range(self.n_layers))
    elif self.model == 'hyperGAT':
      self.convs = nn.ModuleList(hyperGAT(edge_attr_mode=edge_attr_mode, attr_drop=attr_drop, self_loop=self_loop, device=device) for _ in range(self.n_layers))
    else:
      raise ValueError('Model must be NGCF, LightGCN or hyperGCN or hyperGAT')
    
    # Attention mechanism for aggregation
    self.attention_weights = nn.Parameter(torch.ones(self.n_layers + 1, dtype=torch.float32))
    self.softmax = nn.Softmax(dim=0)

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
      
    
    if self.model == 'NGCF':
      out = torch.cat(embs, dim=-1)
    else:
      #out = torch.mean(torch.stack(embs, dim=0), dim=0)
      # Compute attention scores
      attention_scores = self.softmax(self.attention_weights)
      out = torch.stack(embs, dim=0)  # Shape: [n_layers+1, num_nodes, emb_dim]
      #print('\nbefore out:\n', out)
      out = torch.sum(out * attention_scores[:, None, None], dim=0)  # Weighted sum
      
      #print('Attention scores:', attention_scores)
      #print('\nafter out:\n', out)
      
      #sys.exit()
        
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
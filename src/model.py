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

def edge_attr_drop(edge_index, edge_attr, modify_prob=0.2):
    """
    Randomly modifies edge attributes to 1.

    Parameters:
        edge_index (torch.Tensor): The edge index tensor of shape [2, num_edges].
        edge_attr (torch.Tensor): The edge attributes tensor of shape [num_edges, ...].
        modify_prob (float): Probability of modifying an edge attribute to 1.

    Returns:
        new_edge_attr (torch.Tensor): Modified edge attributes.
    """

    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) < modify_prob

    # Modify the selected edge attributes to 1
    new_edge_attr = edge_attr.clone()
    new_edge_attr[mask] = 1.0

    return new_edge_attr

# HyperGCN Convolutional Layer
class hyperGCN(MessagePassing):
    def __init__(self, edge_attr_mode = 'exp', self_loop = False, **kwargs):  
        super().__init__(aggr='add')
        
        self.edge_attr_mode = edge_attr_mode
        self.graph_norms = None
        self.edge_attrs = None
        self.add_self_loops = self_loop
        self.edge_drop = 0.2 # make edge_attr 0 for 20% of edges
        
        # define the scale parameter for edge attributes and set it to 1.0
        #self.scale = nn.Parameter(torch.tensor(4.0))
            
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
          else:
            self.edge_attrs = None
          
          
        self.edge_attrs = edge_attr_drop(edge_index, self.edge_attrs, 0.2)
        
        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=self.graph_norms, attr = self.edge_attrs)

    def message(self, x_j, norm, attr):
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
            
    def forward(self, x, edge_index, edge_attrs, scale):
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

      
class RecSysGNN(nn.Module):
  def __init__(
      self,
      emb_dim, 
      n_layers,
      n_users,
      n_items,
      model, # 'NGCF' or 'LightGCN' or 'hyperGCN'
      dropout=0.1, # Only used in NGCF
      edge_attr_mode = None,
      scale = 1.0,
      self_loop = False
  ):
    super(RecSysGNN, self).__init__()

    assert (model == 'NGCF' or model == 'lightGCN') or model == 'hyperGCN' or model == 'knnNGCF', 'Model must be NGCF or LightGCN or hyperGCN'
    self.model = model
    self.n_users = n_users
    self.n_items = n_items
    self.n_layers = n_layers
    self.emb_dim = emb_dim
    
    # Initialize scale parameters for users and items
    self.scale = nn.Parameter(torch.tensor(scale))
    #self.scale = nn.Parameter(torch.ones(self.n_users + self.n_items) * scale)

    self.embedding = nn.Embedding(self.n_users + self.n_items, self.emb_dim, dtype=torch.float32)
        
    if self.model == 'NGCF':
      self.convs = nn.ModuleList(NGCFConv(self.emb_dim, dropout=dropout) for _ in range(self.n_layers))
    elif self.model == 'knnNGCF':
      self.convs = nn.ModuleList(knnNGCFConv(self.emb_dim, dropout=dropout, edge_attr_mode=edge_attr_mode) for _ in range(self.n_layers))
    elif self.model == 'lightGCN':
      self.convs = nn.ModuleList(LightGCNConv() for _ in range(self.n_layers))
    elif self.model == 'hyperGCN':
      self.convs = nn.ModuleList(hyperGCN(edge_attr_mode=edge_attr_mode, self_loop=self_loop) for _ in range(self.n_layers))
    else:
      raise ValueError('Model must be NGCF, LightGCN or hyperGCN')

    self.init_parameters()

  def init_parameters(self):
    
    if self.model == 'NGCF':
      nn.init.xavier_uniform_(self.embedding.weight, gain=1)
    else:
      # Authors of LightGCN report higher results with normal initialization
      nn.init.normal_(self.embedding.weight, std=0.1)
      #nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1).to(torch.float32)


  def forward(self, edge_index, edge_attrs):
    emb0 = self.embedding.weight
    embs = [emb0]
    
    edge_drop_ratio = 0.2
    
    edge_attrs = edge_attr_drop(edge_index, edge_attrs, edge_drop_ratio)
    
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
        emb0[neg_items]
    )
    
  def predict(self, users, items, edge_index, edge_attrs):
    emb0, out = self(edge_index, edge_attrs)
    
    #return torch.matmul(out[users], emb0[items].t())
    #return torch.matmul(out[users].float(), emb0[items].t())
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
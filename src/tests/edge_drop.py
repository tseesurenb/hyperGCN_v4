import torch
from torch_geometric.data import Data

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

# Example usage
# Create a simple graph with 4 nodes and 6 edges
edge_index = torch.tensor([[0, 1, 2, 3, 0, 2], [1, 0, 3, 2, 2, 0]], dtype=torch.long)

# Optional edge attributes
edge_attr = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float)

# Drop edges with 30% probability
drop_prob = 1.0
new_edge_attr = edge_attr_drop(edge_index, edge_attr, drop_prob, mode=1)

print("Original Edge Index:\n", edge_attr)
print("New Edge Index:\n", new_edge_attr)
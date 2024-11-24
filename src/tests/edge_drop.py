import torch
from torch_geometric.data import Data

def edge_drop(edge_index, edge_attr=None, drop_prob=0.2):
    """
    Randomly drop edges from a graph.
    
    Parameters:
        edge_index (torch.Tensor): The edge index tensor of shape [2, num_edges].
        edge_attr (torch.Tensor, optional): The edge attributes tensor of shape [num_edges, ...]. Default is None.
        drop_prob (float): Probability of dropping an edge. Default is 0.2.
    
    Returns:
        new_edge_index (torch.Tensor): Edge index after edge drop.
        new_edge_attr (torch.Tensor, optional): Edge attributes after edge drop.
    """
    # Generate a mask to decide which edges to keep
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) > drop_prob

    # Filter the edges
    new_edge_index = edge_index[:, mask]
    new_edge_attr = edge_attr[mask] if edge_attr is not None else None

    return new_edge_index, new_edge_attr

# Example usage
# Create a simple graph with 4 nodes and 6 edges
edge_index = torch.tensor([[0, 1, 2, 3, 0, 2],
                           [1, 0, 3, 2, 2, 0]], dtype=torch.long)

# Optional edge attributes
edge_attr = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float)

# Drop edges with 30% probability
drop_prob = 0.3
new_edge_index, new_edge_attr = edge_drop(edge_index, edge_attr, drop_prob)

print("Original Edge Index:\n", edge_index)
print("New Edge Index:\n", new_edge_index)
print("Original Edge Attr:\n", edge_attr)
print("New Edge Attr:\n", new_edge_attr)

import numpy as np
import matplotlib.pyplot as plt

# Simple Hyperbolic Geometry Example with a Network

# Example: Visualizing connections in hyperbolic space (e.g., a tree-like structure)

def plot_hyperbolic_tree(depth=4, branching=3):
    """
    Visualize a hyperbolic-like structure (tree network).

    Args:
        depth: How many levels deep the tree grows.
        branching: Number of child nodes per parent.
    """
    nodes = [(0, 0)]  # Start with the root node at (0, 0) (center of the disk)
    edges = []
    radius_increment = 1  # How far apart each level is

    # Generate positions for nodes in hyperbolic-like space
    for level in range(1, depth + 1):
        radius = level * radius_increment
        angle_step = 2 * np.pi / (branching ** level)
        for i in range(branching ** level):
            angle = i * angle_step
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            nodes.append((x, y))

            # Connect to a parent node
            parent_index = (i // branching) + sum(branching ** d for d in range(level - 1))
            edges.append((parent_index, len(nodes) - 1))

    # Plot the tree in hyperbolic space (simplified as a disk)
    fig, ax = plt.subplots(figsize=(8, 8))
    for edge in edges:
        x_values = [nodes[edge[0]][0], nodes[edge[1]][0]]
        y_values = [nodes[edge[0]][1], nodes[edge[1]][1]]
        ax.plot(x_values, y_values, 'k-', alpha=0.5)

    for node in nodes:
        ax.plot(node[0], node[1], 'ro', markersize=3)

    ax.set_aspect('equal')
    ax.set_title("Hyperbolic Tree Visualization")
    ax.set_xlim(-depth * radius_increment, depth * radius_increment)
    ax.set_ylim(-depth * radius_increment, depth * radius_increment)
    plt.show()

# Run the example
plot_hyperbolic_tree(depth=8, branching=6)

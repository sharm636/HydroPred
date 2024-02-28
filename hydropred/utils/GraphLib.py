import torch
import numpy as np

# Define a function to create adjacency matrix for a 2D grid
def create_adjacency_matrix(num_atoms_row, num_atoms_col):
    adjacency_matrix = np.zeros((num_atoms_row * num_atoms_col, num_atoms_row * num_atoms_col))

    for i in range(num_atoms_row):
        for j in range(num_atoms_col):
            idx = i * num_atoms_col + j

            # Check neighbors (up, down, left, right, and diagonals)
            neighbors = [
                ((i - 1) * num_atoms_col + j) if i > 0 else None,
                ((i + 1) * num_atoms_col + j) if i < num_atoms_row - 1 else None,
                (i * num_atoms_col + j - 1) if j > 0 else None,
                (i * num_atoms_col + j + 1) if j < num_atoms_col - 1 else None,
                ((i - 1) * num_atoms_col + j - 1) if i > 0 and j > 0 else None,
                ((i - 1) * num_atoms_col + j + 1) if i > 0 and j < num_atoms_col - 1 else None,
                ((i + 1) * num_atoms_col + j - 1) if i < num_atoms_row - 1 and j > 0 else None,
                ((i + 1) * num_atoms_col + j + 1) if i < num_atoms_row - 1 and j < num_atoms_col - 1 else None,
            ]

            for neighbor in neighbors:
                if neighbor is not None:
                    adjacency_matrix[idx, neighbor] = 1
                    adjacency_matrix[neighbor, idx] = 1

    return torch.FloatTensor(adjacency_matrix)


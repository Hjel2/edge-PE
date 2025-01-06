"""
In this file, I explicitly construct two pairs of isospectral graphs.
I investigate whether the eigenvectors of their the Laplacians of their adjoints or duals also coincide
Constructions taken from "A geometric construction of isospectral magnetic graphs"
"""
import torch
from torch_geometric.utils import get_laplacian
from make_encoding import adjoint_eigenvectors_and_eigenvalues, hypergraph_eigenvectors_and_eigenvalues

graph1 = torch.tensor([
    [0, 1, 1, 1, 2, 2, 2, 3, 3, 4],
    [1, 2, 4, 5, 3, 4, 5, 4, 5, 5]
])

graph2 = torch.tensor([
    [0, 0, 1, 1, 1, 2, 2, 2, 3, 4],
    [1, 5, 2, 4, 5, 3, 4, 5, 4, 5]
])


def eig_from_edge_index(edge_index):
    adj = torch.zeros((15, 15))
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1
    laplacian = torch.eye(15) - adj
    eigval, eigvec = torch.linalg.eigh(laplacian)
    order = torch.argsort(eigval)
    return eigvec[order, :]


if __name__ == '__main__':
    print(torch.round(eig_from_edge_index(graph1) - eig_from_edge_index(graph2), decimals=2))
    print(torch.round(adjoint_eigenvectors_and_eigenvalues(graph1)[1] - adjoint_eigenvectors_and_eigenvalues(graph2)[1], decimals=2))
    print(torch.round(hypergraph_eigenvectors_and_eigenvalues(graph1)[1] - hypergraph_eigenvectors_and_eigenvalues(graph2)[1], decimals=2))

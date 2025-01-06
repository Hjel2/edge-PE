"""
Implementation of utility functions for edge encodings
"""
import torch


def adjoint_graph_edge_index(edge_index):
    adj = (edge_index[0] == edge_index[1].reshape(-1, 1))
    edge_index = torch.argwhere(adj).transpose(0, 1)
    return edge_index


def adjoint_eigenvectors_and_eigenvalues(edge_index):
    """
    Computes the eigenvectors and eigenvalues of the adjoint graph assuming undirected edges
    """
    diag = torch.eye(edge_index.size(1))
    adj = (edge_index[0] == edge_index[1].reshape(-1, 1))
    adj = (adj | adj.transpose(0, 1)).int()
    laplace = diag - adj

    eigenvalues, eigenvectors = torch.linalg.eigh(laplace)

    order = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[order]

    return eigenvalues, eigenvectors


def hypergraph_eigenvectors_and_eigenvalues(edge_index):
    adj = (edge_index[0] == edge_index[1].reshape(-1, 1))
    adj = (adj | adj.transpose(0, 1)).int()
    diag = torch.eye(edge_index.size(1)) * adj.sum(-1)
    laplace = diag - adj

    eigenvalues, eigenvectors = torch.linalg.eigh(laplace)

    order = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[order]

    return eigenvalues, eigenvectors

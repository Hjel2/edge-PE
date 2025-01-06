"""
In this file, I perform theory.
I:
- show line graph laplacian eigenvectors can differentiate 1-WL indistinguishable graphs
"""
import torch
from make_encoding import adjoint_eigenvectors_and_eigenvalues, hypergraph_eigenvectors_and_eigenvalues


def one_wl_indistinguishable(eigenfn):
    edge_index_1 = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    edge_index_2 = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]])
    evl1, evec1 = eigenfn(edge_index_1)
    evl2, evec2 = eigenfn(edge_index_2)
    print(torch.round(evec1, decimals=3))
    print(torch.round(evec2, decimals=3))


def one_wl_indistinguishable_no_coincide(eigenfn):
    edge_index_1 = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 1, 3, 5, 7],
        [1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11],
    ])
    edge_index_2 = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 1, 3, 5, 7],
                                 [1, 2, 3, 0, 5, 6, 7, 4, 8, 9, 10, 11]])
    evl1, evec1 = eigenfn(edge_index_1)
    evl2, evec2 = eigenfn(edge_index_2)
    print(torch.round(evec1, decimals=3))
    print(torch.round(evec2, decimals=3))


if __name__ == '__main__':
    one_wl_indistinguishable_no_coincide(adjoint_eigenvectors_and_eigenvalues)
    one_wl_indistinguishable_no_coincide(
        hypergraph_eigenvectors_and_eigenvalues)

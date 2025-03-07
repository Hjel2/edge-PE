# How do positional encodings generalise to edges?

Official codebase for the miniproject [_How do positional encodings generalise to edges?_](https://hjel.me/assets/pdf/grl.pdf)

Positional encodings have recently become ubiquitous in the graph learning ecosystem: initially invented for transformers, they have found use even in message-passing neural networks. However, all current positional encodings augment node features to some extent. This project rethinks that.

---

# Contents

`experiments` contains the code for the empirical investigation which found promising evidence for edge encodings acting as a useful inductive bias.

`theory` contains the code used to prove the theoretical results in the paper. This directory proves that edge positional encodings can increase the expressivity of networks.

```
project
│   README.md
│
└───experiments
│   │   gat_csl.ipynb
│   │   gat_zinc.ipynb
│   
└───theory
    │   1-wl.py
    │   isospectral.py
    │   make_encoding.py
```

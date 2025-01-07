[![codecov](https://codecov.io/gh/handdl/btcnn/graph/badge.svg?token=GCC9XW04VB)](https://codecov.io/gh/handdl/btcnn)

**`TL;DR`**  A PyTorch-like implementation of 1-D convolutional layers for binary trees, designed for efficient hierarchical data encoding.

**⚡ Key Features:**
- Efficiently encodes hierarchical structures using 1-D tree convolutions
- Seamless integration with PyTorch, making it easy to use and extend
- Modular design for flexible customization and quick experimentation
- Custom `InstanceNormalization` layer for improved performance

# 💡 Concept

Binary Tree Convolutional Neural Networks (BTCNNs) use specialized 1-D convolutions tailored for binary trees, inspired by graph convolutional networks. This method leverages the natural binary tree structure, where nodes have at most two children, to apply efficient convolution techniques from image processing. By respecting the inherent geometry of tree data, BTCNNs can generalize effectively and produce richer, more meaningful representations.


# 📦 Setup

```bash
python -m pip install --upgrade pip
python3 -v venv venv
source venv/bin/activate
pip install -e .
pytest --cov=. --cov-report=term-missing
```

# 🚀 How To

How to create a `Dataset` / `DataLoader`, configure the architecture, run training and manage trained models is demonstrated in the notebook.

[![example.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/handdl/btcnn/blob/main/src/btcnn/example.ipynb)

# 🧩 Interface

<details>
  <summary><strong>Layer</strong></summary>

Our layers process objects using the following representation:
- `vertices` - 3D tensor of shape `[batch_size, n_channels, max_length_in_batch]`
- `edges` - 4D tensor of shape `[batch_size, 1, max_length_in_batch, 3]`, where the last dimension contains three indices representing the node’s 1-hop neighborhood (`[parent_id, left_child_id, right_child_id]`)

```python3
def forward(self, vertices: "Tensor", edges: "Tensor") -> "Tensor":
    ...
```

_P.S. Currently implemented layers are: `BinaryTreeActivation`, `BinaryTreeAdaptivePooling`, `BinaryTreeConv`, `BinaryTreeLayerNorm`, `BinaryTreeInstanceNorm`_

_P.P.S. To work with this format, zero padding is used to handle a) missing children and b) aligning the tree lengths._

</details>

<details>
  <summary><strong>Stacking</strong></summary>

Since layers must always remember the structure behind the `vertices` (which is stored in `edges`), we decided to build module for layer stacking `BinaryTreeSequential`:

```python
class BinaryTreeSequential(nn.Module):
    def forward(self, vertices: "Tensor", edges: "Tensor") -> "Tensor":
        for layer in self.layers:
            vertices = layer(vertices, edges)
        return vertices
```

</details>

<details>
  <summary><strong>Regressor</strong></summary>

By combining CNN block with FCNN, it is possible to solve prediction problems. In fact, the whole inference
is broken down into two parts - encoding into a vector taking into account the tree structure (`btcnn` part), and then running a fully-connected network (`fcnn` part). 
This is put together in the `BinaryTreeRegressor` module:

```python3
class BinaryTreeRegressor(nn.Module):
    def forward(self, vertices: "Tensor", edges: "Tensor") -> "Tensor":
        return self.fcnn(self.btcnn(vertices=vertices, edges=edges))
```



</details>


# 🔢 Pipeline

<details>
  <summary><strong>Step 1. Vectorize the binary tree.</strong></summary>

```python3
                [1.0, 1.0]
                 /       \
        [1.0, -1.0]     *None*
            /    \                 
  [-1.0, -1.0]   [1.0, 1.0]
```

</details>

<details>
  <summary><strong>Step 2. Add padding nodes for all incomplited nodes.</strong></summary>
  
```python3
                [1.0, 1.0]
                 /       \
        [1.0, -1.0]   [0.0, 0.0]   # padding node
            /    \                 
  [-1.0, -1.0]   [1.0, 1.0]
```
  
</details>
  

<details>
  <summary><strong>Step 3. Construct tensors for <code>vertices</code> and <code>edges</code> using a tree traversal.</strong></summary>
  
```python3
# vertices 
[[0, 0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]]

# edges in the form `[node_id, left_child_id, right_child_id]`
[[1, 2, 0], [2, 3, 4], [3, 0, 0], [4, 0, 0]]
```

</details>


<details>
  <summary><strong>Step 4. Convolve over the binary tree neighborhoods.</strong></summary>
  
To account for the binary tree structure, we’ll convolve over the parent, left child, and right child nodes. This can be visualized as a filter moving across the tree structure:

```python3
       [θ_11, θ_12]
         /       \
[θ_21, θ_22]   [θ_31, θ_32]
```

**🪄 Trick:** the knowledge that each node has either zero or two children allows us to stretch the entire tree into a tensor of size `3 * tree_length`, a one-dimensional CNN with a 
`stride=3` can then capture the tree’s neighborhood, leveraging efficient convolution implementations while maintaining the tree’s geometry.

</details>

<details>
  <summary><strong>Step 5. Apply point-wise Activation and Adaptive Pooling.</strong></summary>
  
After applying several convolutional layers (along with point-wise non-linear functions and normalization layers), we can use a adaptive pooling method to reduce the tree to a _fixed-size_ vector.

```python3
                [a, e]
                /    \
            [b, f]   [e, k]
             /  \                 
        [c, g]  [d, h]

# after `AdaptiveMaxPooling` layer, the tree becomes a vector which size is equal to the number of channels in the tree
vector = [max(a, b, c, d, e), max(e, f, g, h, k)]
```
</details>

<details>
<summary><b> 🔥 Putting It All Together </b></summary>

<b>First</b>, a convolution with the filter is performed independently for each neighbourhood. An example of neighbourhood convolution on the root:

```python3
# tree
                [1.0,1.0]
                 /     \
        [1.0, -1.0]   *None*
            /  \                 
 [-1.0,-1.0]   [1.0,1.0]

# filter
       [1.0,-1.0]
         /      \
[-1.0,-1.0]   [1.0,1.0]

# root's neighborhood convolution
                [1.0,1.0]                [1.0,-1.0]
                 /     \        *         /      \            =      [0.0]
        [1.0,-1.0]   [0.0,0.0]    [-1.0,-1.0]   [1.0,1.0]
# (1.0 * 1.0 + 1.0 * -1.0) + (1.0 * -1.0 + -1.0 * -1.0) + (0.0 * 1.0 + 0.0 * 1.0) = 0.0
```

<b>In second</b>, normalization and activation layers are applied. <b>In third</b>, dynamic pooling layer maps tree to fixed-length vector. 
Considering the structure of the tree, the following happens to the tree throughout the process:

```python3
                 # tree                  # filter                # after Conv            # after Norm & ReLU    # after AdaptiveMaxPooling

                [1.0,1.0]                                             [0.0]                      [0.0]                                     
                 /     \                [1.0,-1.0]                   /     \                     /   \
        [1.0,-1.0]   *None*  *           /      \        ->       [6.0]     *None*  ->     [1.73]  *None*  ->  [1.73]
            /  \                [-1.0,-1.0]   [1.0,1.0]           /  \                       /  \                                       
 [-1.0,-1.0]   [1.0,1.0]                                     [0.0]   [0.0]              [0.0]  [0.0]
```

**👁️⃤ Intuition.** After normalizing and applying the ReLU activation, the left child of the root becomes prominent. This happens because its values closely match the filter weights. This prominence indicates the similarity of the substructure to the filter. When training multiple filters simultaneously and combining convolutional blocks, we begin to capture more complex structures, such as subtrees of height 2, 3, and beyond. `BTCNN` effectively identifies key substructures in the tree, and then a `FCNN` assesses their presence.

</details>

# ✨ [new] Normalization Layer

To simplify the optimisation problem, it is useful to use normalization layers within the convolution blocks. 
Among all the options tried by us, `InstanceNormalization` worked best of all.

<details>
    
<summary><b>Descriptions</b></summary>

<br>

**Batch Normalization.** Aggregation is performed across all trees in the batch.
    
```python3
               [10000]                       [100]
               /     \                      /     \
            [100]   [100]               [10]     [10]
            /  \     /  \               /  \     
         [10] [10] [10] [10],        [2]   [5] 

batch_vertices = [
    [[.0], [10000], [100], [100], [10], [10], [10], [10]],
    [[.0], [100],   [10],  [2],   [5],  [10], [.0], [.0]],
]
batch_edges = [[[1,2,5], [2,3,4], [3,0,0], [4,0,0], [5,0,0]]]
batch_vertices_mean = mean(batch_vertices)  # [5050, 105, 101, 7.5, 10, 5, 5]
```

The Batch Normalization does not suit us in a similar way to any NN over sequence reason - objects in a batch may have representations responsible for completely different information at the same position. As a result, aggregation by objects in the batches will lead to the fact that we will mix, for example, statistics of tree roots of different heights (which, given the semantics of statistics, is _inappropriate_ - characteristic orders of magnitude of cardinalities grow with tree height). 

**Layer Normalization.** Aggregation is performed independently for each tree.
    
```python3
                [1.0,1.0]
                 /     \
        [1.0, -1.0]   *None*
            /  \                 
 [-1.0,-1.0]   [1.0,1.0]

tree_mean = mean([1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0])  # 0.25
tree_std = std([1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.,0 0., 0.])  # 0.9682458365518543
```

**Instance Normalization.** Aggregation is performed independently for each tree and each channel.
    
```python3
                [1.0,1.0]
                 /     \
        [1.0, -1.0]   *None*
            /  \                 
 [-1.0,-1.0]   [1.0,1.0]

tree_mean = [mean([1.0, 1.0, -1.0, 1.0]), mean([1.0, -1.0, -1.0, 1.0])]  # [0.5, .0]
tree_std = [std([1.0, 1.0, -1.0, 1.0]),  std([1.0, -1.0, -1.0, 1.0])]  # [0.8660254037844386, 1.0]
```

</details>

# 📚 References

- **Original Paper**: [Mou et al., 2015 - Convolutional Neural Networks over Tree Structures for Programming Language Processing](https://arxiv.org/pdf/1409.5718)
- **Inspired Implementation**: [TreeConvolution](https://github.com/RyanMarcus/TreeConvolution)

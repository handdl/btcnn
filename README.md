**`TL;DR`** Here is `torch`-like implementation of layers designed for constructing convolutional blocks on structured data, specifically binary trees, these layers can be particularly 
useful for tasks such as vectorizing Abstract Syntax Trees (ASTs), query execution plans in databases, and extracting significant
information from the structure of objects. 

P.S. This type of convolution lies _between_ conventional CNNs used for images and graph-based CNNs.  The constraint that each node in the binary tree has at most two neighbors
allows the data to be formatted in a way that a one-dimensional CNN can efficiently process while considering the tree’s structure.

# Interface

**Layer.** Our layers process the next structures:
- `vertices` - 3D tensor of shape `[batch_size, n_channels, batch_length]`
- `edges` - 4D tensor of shape `[batch_size, 1, batch_length, 3]`, where the last dimension contains three indices representing the node’s 1-hop neighborhood (`[parent_id, left_child_id, right_child_id]`)

```python3
def forward(self, vertices: "Tensor", edges: "Tensor") -> "Tensor":
    ...
```

_P.S. To work with this format, zero padding is used to handle a) missing children and b) aligning the tree lengths._

**Stacking.** Since layers working with the tree must always remember the structure behind the `vertices` (which is stored in `edges`), we decided to build our own module for layer stacking `BinaryTreeSequential`:

```python3
class BinaryTreeSequential(nn.Module)
    def forward(self, vertices: "Tensor", edges: "Tensor") -> "Tensor":
        for layer in self.layers:
            vertices = layer(vertices, edges)
        return vertices
```

**Regressor.** By combining convolutional block tree vectorisation and fully connected networks, it is possible to solve prediction problems. In fact, the whole inference
is broken down into two parts - encoding into a vector taking into account the tree structure (`btcnn` part), and then running a fully-connected network (`fcnn` part). 
This is put together in the `BinaryTreeRegressor` module:

```python3
class BinaryTreeRegressor(nn.Module):
    def forward(self, vertices: "Tensor", edges: "Tensor") -> "Tensor":
        return self.fcnn(self.btcnn(vertices=vertices, edges=edges))
```

# BTCNN Pipeline

Here, we assume that we have a vectorized binary tree (which can be obtained, for instance, via node-vectorization from **X**).

**Step 1. Vectorize the binary tree.**
```python3
                [1.0, 1.0]
                 /       \
        [1.0, -1.0]     *None*
            /    \                 
  [-1.0, -1.0]   [1.0, 1.0]
```

**Step 2. Add Padding Nodes for all incomplited nodes.**
```python3
                [1.0, 1.0]
                 /       \
        [1.0, -1.0]   [0.0, 0.0]   # padding node
            /    \                 
  [-1.0, -1.0]   [1.0, 1.0]
```

**Step 3. Construct tensors for `vertices` and `edges` using a left-first preorder traversal.**

```python3
# vertices 
[[0, 0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]]
# edges in the form `[node_id, left_child_id, right_child_id]`
[[1, 2, 0], [2, 3, 4], [3, 0, 0], [4, 0, 0]]
```

**Step 4. Convolve over the binary tree neighborhoods.**
To account for the binary tree structure, we’ll convolve over the parent, left child, and right child nodes. This can be visualized as a filter moving across the tree structure:

```python3
       [θ_11, θ_12]
         /       \
[θ_21, θ_22]   [θ_31, θ_32]
```

The knowledge that each node has either zero or two children allows us to stretch the entire tree into a tensor of size `3 * tree_length`. A one-dimensional CNN with a 
`stride=3` can then capture the tree’s neighborhood, leveraging efficient convolution implementations while maintaining the tree’s geometry.

**Step 5. Apply Adaptive Pooling.** 
After applying several convolutional layers (along with point-wise non-linear functions and normalization layers), we can use a adaptive pooling method to reduce the tree to a fixed-size vector.


```python3
                [a, e]
                /    \
            [b, f]   [e, k]
             /  \                 
        [c, g]  [d, h]

# After `AdaptiveMaxPooling` layer, the tree becomes a vector which size is equal to the number of channels in the tree
[max(a, b, c, d, e), max(e, f, g, h, k)]
```

# Normalisation Layers

To simplify the optimisation problem, it is useful to use normalisation layers within the convolution blocks. 

## Batch Normalisation (inappropriate)

Aggregation is performed across all trees in the batch.

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

The Batch Normalisation does not suit us in a similar way to any NN over sequence reason - objects in a batch may have representations responsible for completely different information at the same position.
As a result, aggregation by objects in the batches will lead to the fact that we will mix, for example, statistics of tree roots of different heights (which, given the semantics of statistics, 
is _inappropriate_ - characteristic orders of magnitude of cardinalities grow with tree height). 


## Layer Normalisation (good)
Aggregation is performed independently for each tree.

```python3
                [1.0,1.0]
                 /     \
        [1.0, -1.0]   *None*
            /  \                 
 [-1.0,-1.0]   [1.0,1.0]

tree_mean = mean([1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0])  # 0.25
tree_std = std([1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.,0 0., 0.])  # 0.9682458365518543
```

## Instance Normalisation (best)
Aggregation is performed independently for each tree and each channel.

```python3
                [1.0,1.0]
                 /     \
        [1.0, -1.0]   *None*
            /  \                 
 [-1.0,-1.0]   [1.0,1.0]

tree_mean = [mean([1.0, 1.0, -1.0, 1.0]), mean([1.0, -1.0, -1.0, 1.0])]  # [0.5, .0]
tree_std = [std([1.0, 1.0, -1.0, 1.0]),  std([1.0, -1.0, -1.0, 1.0])]  # [0.8660254037844386, 1.0]
```

# Example

Let's considere convolution of the next tree with the next filter

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
```

First, a convolution with the filter will be performed independently for each neighbourhood

```python3
# root
                [1.0,1.0]                [1.0,-1.0]
                 /     \        *         /      \            =      [0.0]
        [1.0,-1.0]   [0.0,0.0]    [-1.0,-1.0]   [1.0,1.0]
# (1.0 * 1.0 + 1.0 * -1.0) + (1.0 * -1.0 + -1.0 * -1.0) + (0.0 * 1.0 + 0.0 * 1.0) = 0.0

# left child of the root
                [1.0,-1.0]               [1.0,-1.0]
                 /     \        *         /      \            =      [6.0]
        [-1.0,-1.0]   [1.0,1.0]    [-1.0,-1.0]   [1.0,1.0]
# (1.0 * 1.0 + -1.0 * -1.0) + (-1.0 * -1.0 + -1.0 * -1.0) + (1.0 * 1.0 + 1.0 * 1.0) = 6.0

# left child of the left child of the root
                [-1.0,-1.0]                [1.0,-1.0]
                 /     \        *           /      \            =      [0.0]
        [0.0,0.0]   [0.0,0.0]       [-1.0,-1.0]   [1.0,1.0]
# (-1.0 * 1.0 + -1.0 * -1.0) + (0.0 * -1.0 + 0.0 * -1.0) + (0.0 * 1.0 + 0.0 * 1.0) = 0.0

# right child of the left child of the root
                [1.0,1.0]                 [1.0,-1.0]
                 /     \        *          /      \            =      [0.0]
        [0.0,0.0]   [0.0,0.0]      [-1.0,-1.0]   [1.0,1.0]
# (1.0 * 1.0 + 1.0 * -1.0) + (0.0 * -1.0 + 0.0 * -1.0) + (0.0 * 1.0 + 0.0 * 1.0) = 0.0
```

Normalisation and activation layers are then applied. Given a structure over the vertices, the following happens to the tree:

```python3
              #  tree                     # filter                # after Convolution       # after Normalization & ReLU    # after AdaptiveMaxPooling

                [1.0,1.0]                                               [0.0]                      [0.0]                                     
                 /     \                   [1.0,-1.0]                  /     \                     /   \
        [1.0,-1.0]   *None*       *         /      \          ->    [6.0]     *None*    ->   [1.73..]  *None*   ->       [1.73]
            /  \                    [-1.0,-1.0]   [1.0,1.0]         /  \                       /  \                                       
 [-1.0,-1.0]   [1.0,1.0]                                        [0.0]   [0.0]              [0.0]  [0.0]
```

After normalising and applying the `ReLU` activation, we see that the value of the left child of the root _stands out_ - and this is expected, because the values of its neighborhood are same to the filter weights.
In fact, this value indicates how similar to the filter the substructure pattern was found in the tree. When we train several filters at once, and combine convolutional blocks at the same time, we start learning 
more filters with more complex structure - subtrees of height 2, 3 and so on are considered. 

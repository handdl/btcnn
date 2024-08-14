import torch
from torch import nn, Tensor
from layers import BinaryTreeSequential


class BinaryTreeRegressor(nn.Module):
    def __init__(
        self,
        btcnn: "BinaryTreeSequential",
        fcnn: "nn.Sequential",
        name: "str" = "unknown",
        device: "torch.device" = torch.device("cpu"),
    ):
        super().__init__()
        self.btcnn: "BinaryTreeSequential" = btcnn
        self.fcnn: "nn.Sequential" = fcnn
        self.device: "torch.device" = device
        self.name: "str" = name
        self.fcnn.to(self.device)
        self.btcnn.to(self.device)

    def forward(self, vertices: "Tensor", edges: "Tensor") -> "Tensor":
        return self.fcnn(self.btcnn(vertices=vertices, edges=edges))

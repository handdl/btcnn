import torch
from torch import nn, Tensor
from layers import (
    BinaryTreeSequential,
    BinaryTreeActivation,
    BinaryTreeConv,
    BinaryTreeInstanceNorm,
    BinaryTreeAdaptivePooling,
)


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


def get_big_btcnn_and_instance_norm(in_channels: "int") -> "BinaryTreeSequential":
    return BinaryTreeSequential(
        BinaryTreeConv(in_channels, 64),
        BinaryTreeActivation(torch.nn.functional.leaky_relu),
        BinaryTreeConv(64, 128),
        BinaryTreeInstanceNorm(128),
        BinaryTreeActivation(torch.nn.functional.leaky_relu),
        BinaryTreeConv(128, 256),
        BinaryTreeInstanceNorm(256),
        BinaryTreeActivation(torch.nn.functional.leaky_relu),
        BinaryTreeConv(256, 512),
        BinaryTreeAdaptivePooling(torch.nn.AdaptiveMaxPool1d(1)),
    )


def get_big_fcnn() -> "nn.Sequential":
    return nn.Sequential(
        nn.Linear(512, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 1),
        nn.Softplus(),
    )


def get_bt_regressor(in_channels: "int", name: "str", device: "torch.device") -> "BinaryTreeRegressor":
    return BinaryTreeRegressor(get_big_btcnn_and_instance_norm(in_channels), get_big_fcnn(), name=name, device=device)

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import lr_scheduler
from regressor import get_bt_regressor
from train_utils import weighted_train_loop, calculate_loss, load_model, set_seed
from dataset import WeightedBinaryTreeDataset, weighted_binary_tree_collate


def test_train(vertices: "Tensor", edges: "Tensor"):
    test_time = torch.tensor(42.0)
    batch_size, lr, epochs = 8, 3e-4, 500
    device = torch.device("cpu")
    dataloader = DataLoader(
        dataset=WeightedBinaryTreeDataset(
            [vertices] * batch_size, [edges] * batch_size, [test_time] * batch_size, device
        ),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda el: weighted_binary_tree_collate(el, 10),
        drop_last=False,
    )

    model = get_bt_regressor(in_channels=2, name="test_model", device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)

    set_seed(42)
    weighted_train_loop(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(reduction="none"),
        scheduler=scheduler,
        train_dataloader=dataloader,
        num_epochs=epochs,
        ckpt_period=epochs,
        path_to_save=f"/tmp/{model.name}.pth",
    )

    final_loss = calculate_loss(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(reduction="none"),
        dataloader=dataloader,
        train_mode=False,
    )

    assert final_loss < 1e-3, "Problems with fitting"

    model = load_model(model, f"/tmp/{model.name}.pth", device)
    final_loss_after_reloading = calculate_loss(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(reduction="none"),
        dataloader=dataloader,
        train_mode=False,
    )
    assert abs(final_loss - final_loss_after_reloading) < 1e-3, "Inconsistency after reloading"

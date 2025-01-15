import torch.nn as nn

def get_loss(loss_config):
    loss_name = loss_config["name"]

    if loss_name == "bce":
        loss_fn = nn.BCEWithLogitsLoss()  # For multilabel classification
    elif loss_name == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()  # For multiclass classification
    else:
        raise ValueError(f"Loss function {loss_name} is not supported")

    return loss_fn

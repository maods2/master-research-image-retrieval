from torchvision import models
import torch.nn as nn

def get_model(model_config):
    model_name = model_config["name"]
    num_classes = model_config["num_classes"]

    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust for the custom output layer
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} is not supported")

    return model

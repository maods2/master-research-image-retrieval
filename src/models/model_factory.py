from torchvision import models
import torch.nn as nn

from models.triplet_resnet import TripletResNet
from models.triplet_vit import TripletViT
from utils.checkpoint_utils import load_checkpoint


def get_model(model_config):
    model_name = model_config['name']
    num_classes = model_config['num_classes']

    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(
            model.fc.in_features, num_classes
        )  # Adjust for the custom output layer
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features, num_classes
        )

    elif model_name == 'triplet_resnet':
        model = TripletResNet(embedding_size=model_config['embedding_size'])

    elif model_name == 'triplet_vit':
        model = TripletViT(embedding_size=model_config['embedding_size'])

    else:
        raise ValueError(f'Model {model_name} is not supported')

    if model_config['load_checkpoint']:
        load_checkpoint(model_config['checkpoint_path'], model)

    return model

from torchvision import models
import torch.nn as nn

from models.clip import CLIP
from models.triplet_resnet import TripletResNet, ResNet50
from models.dino import DINO, DINOv2
from models.uni import UNI
from models.virchow2 import Virchow2
from models.vit import ViT, TripletViT
from utils.checkpoint_utils import load_checkpoint


def get_model(model_config):
    model_name = model_config['name']
    num_classes = model_config['num_classes']

    if model_name == 'resnet50':
        model = ResNet50()  
        
    elif model_name == 'dino':
        model = DINO(model_name=model_config['model_name'])

    elif model_name == 'dinov2':
        model = DINOv2(model_name=model_config['model_name'])
        
    elif model_name == 'clip':
        model = CLIP(  model_name=model_config['model_name'])

    elif model_name == 'triplet_resnet':
        model = TripletResNet(embedding_size=model_config['embedding_size'])

    elif model_name == 'triplet_vit':
        model = TripletViT(embedding_size=model_config['embedding_size'])
        
    elif model_name == 'vit':        
        model = ViT(model_name=model_config['vit_model_name'])

    elif model_name == 'uni': # Pathology Foundation Model
        model = UNI(model_name=model_config['model_name'])

    elif model_name == 'virchow2': # Pathology Foundation Model
        model = Virchow2(model_name=model_config['model_name'])


    else:
        raise ValueError(f'Model {model_name} is not supported')
    
    print(model_config['load_checkpoint'])

    if model_config['load_checkpoint']:
        load_checkpoint(model_config['checkpoint_path'], model)

    return model

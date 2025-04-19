import os
import sys

from models.fsl_models import UNIFsl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from torchvision import models
import torch.nn as nn

from src.models.clip import CLIP
from src.models.triplet_resnet import TripletResNet, ResNet50
from src.models.dino import DINO, DINOv2
from src.models.uni import UNI
from src.models.virchow2 import Virchow2
from src.models.vit import ViT, TripletViT
from src.utils.checkpoint_utils import load_checkpoint


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
        model = ViT(model_name=model_config['model_name'])

    elif model_name == 'uni': # Pathology Foundation Model
        model = UNI(model_name=model_config['model_name'])

    elif model_name == 'uni_fsl': # Pathology Foundation Model
        model = UNIFsl(model_name=model_config['model_name'])

    elif model_name == 'virchow2': # Pathology Foundation Model
        model = Virchow2(model_name=model_config['model_name'])


    else:
        raise ValueError(f'Model {model_name} is not supported')
    
    if model_config['load_checkpoint']:
        load_checkpoint(model_config['checkpoint_path'], model)

    return model

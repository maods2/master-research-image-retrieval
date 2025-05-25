import os
import sys



sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
)

from src.models.clip import CLIP
from src.models.triplet_resnet import TripletResNet, ResNet50, ResNet
from src.models.dino import DINO, DINOv2
from src.models.uni import UNI
from src.models.virchow2 import Virchow2
from src.models.vit import ViT, TripletViT
from src.models.fsl_models import WrappedFsl
from src.models.phikon import Phikon
from src.utils.checkpoint_utils import load_checkpoint


def get_model(model_config):
    model_name = model_config['name']

    if model_name == 'resnet':
        model = ResNet(model_name=model_config['model_name'])
        
    elif model_name == 'dino':
        model = DINO(model_name=model_config['model_name'])

    elif model_name == 'dinov2':
        model = DINOv2(model_name=model_config['model_name'])

    elif model_name == 'clip':
        model = CLIP(model_name=model_config['model_name'])

    elif model_name == 'vit':
        model = ViT(model_name=model_config['model_name'])

    elif model_name == 'uni':   # Pathology Foundation Model
        model = UNI(model_name=model_config['model_name'])

    elif model_name == 'UNI2-h':   # Pathology Foundation Model
        model = UNI(model_name=model_config['model_name'])

    elif model_name == 'virchow2':   # Pathology Foundation Model
        model = Virchow2(model_name=model_config['model_name'])
                
    elif model_name == 'phikon':   # Pathology Foundation Model
        model = Phikon(model_name=model_config['model_name'])

    elif model_name == 'phikon-v2':   # Pathology Foundation Model
        model = Phikon(model_name=model_config['model_name'])
        

################### Few-Shot Learning Models ######################################
    
    elif model_name == 'dino_fsl':
        backbone = DINO(model_name=model_config['model_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config['hidden_dim'], 
            embedding_dim=model_config['embedding_dim']
            )

    elif model_name == 'dinov2_fsl':
        backbone = DINOv2(model_name=model_config['model_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config['hidden_dim'], 
            embedding_dim=model_config['embedding_dim']
            )

    elif model_name == 'clip_fsl':
        backbone = CLIP(model_name=model_config['model_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config['hidden_dim'], 
            embedding_dim=model_config['embedding_dim']
            )

    elif model_name == 'vit_fsl':
        backbone = ViT(model_name=model_config['model_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config['hidden_dim'], 
            embedding_dim=model_config['embedding_dim']
            )

    elif model_name == 'uni_fsl':   # Pathology Foundation Model
        backbone = UNI(model_name=model_config['model_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config['hidden_dim'], 
            embedding_dim=model_config['embedding_dim']
            )

    elif model_name == 'UNI2-h_fsl':   # Pathology Foundation Model
        backbone = UNI(model_name=model_config['model_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config['hidden_dim'], 
            embedding_dim=model_config['embedding_dim']
            )

    elif model_name == 'virchow2_fsl':   # Pathology Foundation Model
        backbone = Virchow2(model_name=model_config['model_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config['hidden_dim'], 
            embedding_dim=model_config['embedding_dim']
            )

    elif model_name == 'resnet_fsl':   # Pathology Foundation Model
        backbone = ResNet(model_name=model_config['model_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config['hidden_dim'], 
            embedding_dim=model_config['embedding_dim']
            )

    elif model_name == 'phikon':   # Pathology Foundation Model
        backbone = Phikon(model_name=model_config['model_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config['hidden_dim'], 
            embedding_dim=model_config['embedding_dim']
            )

    elif model_name == 'phikon-v2':   # Pathology Foundation Model
        backbone = Phikon(model_name=model_config['model_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config['hidden_dim'], 
            embedding_dim=model_config['embedding_dim']
            )
        
################### Triplet Models #################################################
    
    elif model_name == 'triplet_resnet':
        model = TripletResNet(embedding_size=model_config['embedding_size'])

    elif model_name == 'triplet_vit':
        model = TripletViT(embedding_size=model_config['embedding_size'])

    else:
        raise ValueError(f'Model {model_name} is not supported')



################## NOT USED ######################################################


    if model_config['load_checkpoint']:
        load_checkpoint(model_config['checkpoint_path'], model)

    return model

import torch.nn as nn

from losses.loss_triplet import AdaptiveTripletLoss
from losses.prototypical_loss import PrototypicalLoss
from losses.loss_contrastive import (
    NTXentLoss,
    SupConLoss,
    ProxyNCALoss,
    MultiSimilarityLoss,
    ArcFaceLoss,
    NPairLoss,
)


def get_loss(loss_config):
    loss_name = loss_config['name']

    if not loss_name:
        return None
    elif loss_name == 'bce':
        loss_fn = nn.BCEWithLogitsLoss()  # For multilabel classification
    elif loss_name == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss()  # For multiclass classification
    elif loss_name == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_name == 'adaptative_triplet':
        loss_fn = AdaptiveTripletLoss()
    elif loss_name == 'prototypical':
        loss_fn = PrototypicalLoss(loss_config)
    elif loss_name == 'ntxent':
        loss_fn = NTXentLoss(loss_config)
    elif loss_name == 'supervised_contrastive':
        loss_fn = SupConLoss(loss_config)
    elif loss_name == 'proxy_nca':
        loss_fn = ProxyNCALoss(loss_config)
    elif loss_name == 'multi_similarity':
        loss_fn = MultiSimilarityLoss(loss_config)
    elif loss_name == 'arcface':
        loss_fn = ArcFaceLoss(loss_config)
    elif loss_name == 'npair':
        loss_fn = NPairLoss(loss_config)

    else:
        raise ValueError(f'Loss function {loss_name} is not supported')

    return loss_fn

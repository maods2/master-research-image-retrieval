def get_train_function(config):
    
    if config['training']['pipeline'] == 'train_multilabel':
        from pipelines.training_pipes.multilabel_train import train_multilabel
        return train_multilabel
    
    elif config['training']['pipeline'] == 'train_triplet':
        from pipelines.training_pipes.triplet_train import TripletTrain
        return TripletTrain(config)
    
    else:
        raise ValueError(
            f'Training pipeline {config["training"]["pipeline"]} is not supported'
        )

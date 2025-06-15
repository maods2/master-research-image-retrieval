def get_train_function(config):

    if config['training']['pipeline'] == 'train_multilabel':
        from pipelines.training_pipes.multilabel_train import train_multilabel

        return train_multilabel

    elif config['training']['pipeline'] == 'train_triplet':
        from pipelines.training_pipes.triplet_train import TripletTrain

        return TripletTrain(config)

    elif config['training']['pipeline'] == 'train_few_shot_leaning':
        from pipelines.training_pipes.few_shot_train import FewShotTrain

        return FewShotTrain(config)

    elif config['training']['pipeline'] == 'supervised_constrastive_leaning':
        from pipelines.training_pipes.supcon_trainer import SupConTrainer

        return SupConTrainer(config)

    elif config['training']['pipeline'] == 'train_autoencoder':
        from pipelines.training_pipes.autoencoder_trainer import AutoencoderTrainer

        return AutoencoderTrainer(config)
    
    
    else:
        raise ValueError(
            f'Training pipeline {config["training"]["pipeline"]} is not supported'
        )

def get_train_function(config):

    if config['training']['pipeline'] == 'train_multilabel':
        from pipelines.training_pipes.multilabel_train import train_multilabel

        return train_multilabel

    elif config['training']['pipeline'] == 'train_triplet':
        from pipelines.training_pipes.triplet_train import TripletTrain

        return TripletTrain(config)

    elif config['training']['pipeline'] == 'train_few_shot_leaning':
        from pipelines.training_pipes.few_shot_train import FewShotTrain
        

    elif config['training']['pipeline'] == 'contrastive':
        from pipelines.training_pipes.contrastive_train import ContrastiveTrain 

        return ContrastiveTrain(config)

    else:
        raise ValueError(
            f'Training pipeline {config["training"]["pipeline"]} is not supported'
        )

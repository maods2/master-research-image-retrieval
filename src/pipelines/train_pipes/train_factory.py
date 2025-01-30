def get_train_function(pipeline_config):
    if pipeline_config == 'train_multilabel':
        from pipelines.train_pipes.multilabel_train import train_multilabel

        return train_multilabel
    elif pipeline_config == 'train_triplet':
        from pipelines.train_pipes.triplet_train import TripletTrain

        return TripletTrain()
    else:
        raise ValueError(
            f'Training pipeline {pipeline_config} is not supported'
        )

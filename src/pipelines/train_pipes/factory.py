def get_train_function(pipeline_config):
    if pipeline_config == "train_multilabel":
        from src.pipelines.train_pipes.multilabel_train import train_multilabel
        return train_multilabel
    else:
        raise ValueError(f"Training pipeline {pipeline_config} is not supported")

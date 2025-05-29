import os
from pathlib import Path
from typing import Dict, List
from ruamel.yaml import YAML
from config_templates import create_config

def load_template(template_path: str) -> dict:
    """Load the YAML template file."""
    yaml = YAML()
    with open(template_path, 'r') as f:
        return yaml.load(f)



def main(template_type: str = "fsl_train") -> None:
    
    from config_models import fsl_models, retrieval_backbone_models
    
    # Define output directory for configs
    output_dir = "./configs/"
    
    # Define datasets and their class mappings
    datasets = {
        "skin-cancer-splitted": {
            "class-mapping": {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3,"mel": 4, "nv": 5, "vasc": 6 },
            "train_dir": "datasets/final/skin-cancer-splitted/train",
            "test_dir": "datasets/final/skin-cancer-splitted/test"
        },
        "ovarian-cancer-splitted": {
            "class-mapping": {"Clear_Cell": 0, "Endometri": 1, "Mucinous": 2, "Non_Cancerous": 3,"Serous": 4},
            "train_dir": "datasets/final/ovarian-cancer-splitted/train",
            "test_dir": "datasets/final/ovarian-cancer-splitted/test"
        },
        "glomerulo":{
            "class-mapping": {"Crescent": 0, "Hypercellularity": 1, "Membranous": 2, "Normal": 3,"Podocytopathy": 4, "Sclerosis": 5},
            "train_dir": "datasets/final/glomerulo/train",
            "test_dir": "datasets/final/glomerulo/test"
        },
        # Add more datasets as needed
    }

    
    experiment_name = None
    if template_type == "retrieval_test":
        template_path = "./configs/templates/retrieval_test/default_model_config.yaml"
        models = retrieval_backbone_models
        experiment_name = "retr_test_backone"
        config_type_folder=f"/{experiment_name}/"

    elif template_type == "retrieval_test_norm":
        template_path = "./configs/templates/retrieval_test/default_model_config.yaml"
        models = retrieval_backbone_models
        experiment_name = "retr_test_backone_norm"
        config_type_folder=f"/{experiment_name}/"
        
    elif template_type == "fsl_train":
        template_path = "./configs/templates/fsl_train/default_train_config.yaml"
        models = fsl_models
        experiment_name = "retr_fsl_train_test"
        config_type_folder=f"/{experiment_name}/"
    else:
        raise ValueError("Invalid template type.")
    
    
    template = load_template(template_path)
    for dataset_name, dataset_config in datasets.items():
        for model_config in models:
            create_config(
                template=template,
                dataset_name=dataset_name,
                model_config=model_config,
                dataset_config=dataset_config,
                output_dir=output_dir,
                config_type_folder=config_type_folder,
                template_type=template_type,
                experiment_name=experiment_name
            )

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        template_type = sys.argv[1]
    else:
        raise ValueError("Please provide a template type as an argument.(fsl_train | retrieval_test)")
    
    main(template_type)
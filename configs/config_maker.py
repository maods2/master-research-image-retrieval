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



def main():
    from config_models import fsl_models
    
    # Define output directory for configs
    output_dir = "./configs/"
    
    # Define datasets and their class mappings
    datasets = {
        "CRC-VAL-HE-7K-splitted": {
            "class-mapping": {"ADI": 0, "BACK": 1, "DEB": 2, "LYM": 3,"MUC": 4, "MUS": 5, "NORM": 6, "STR": 7, "TUM": 8},
            "train_dir": "datasets/final/CRC-VAL-HE-7K-splitted/train",
            "test_dir": "datasets/final/CRC-VAL-HE-7K-splitted/test"
        },
        "skin-cancer-splitted": {
            "class-mapping": {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3,"mel": 4, "nv": 5, "vasc": 6 },
            "train_dir": "datasets/final/skin-cancer-splitted/train",
            "test_dir": "datasets/final/skin-cancer-splitted/test"
        },
        "bracs-resized": {
            "class-mapping": {"0_N": 0, "1_PB": 1, "2_UDH": 2, "3_FEA": 3,"4_ADH": 4, "5_DCIS": 5, "6_IC": 6},
            "train_dir": "datasets/final/bracs-resized/train",
            "test_dir": "datasets/final/bracs-resized/retriv_test"
        },
        "ovarian-cancer-splitted": {
            "class-mapping": {"Clear_Cell": 0, "Endometri": 1, "Mucinous": 2, "Non_Cancerous": 3,"Serous": 4},
            "train_dir": "datasets/final/ovarian-cancer-splitted/train",
            "test_dir": "datasets/final/ovarian-cancer-splitted/test"
        },
        # "glomerulo":{
        #     "class-mapping": {"Crescent": 0, "Hypercellularity": 1, "Membranous": 2, "Normal": 3,"Podocytopathy": 4, "Sclerosis": 5},
        #     "train_dir": "datasets/final/glomerulo/train",
        #     "test_dir": "datasets/final/glomerulo/test"
        # },
        # Add more datasets as needed
    }

    # Load template
    # template_path = "./configs/templates/retrieval_test/default_model_config.yaml" # tamplate used for retrieval
    template_path = "./configs/templates/fsl_train/default_train_config.yaml" # tamplate used for fsl training
    template = load_template(template_path)
    
    models = fsl_models
    # Generate configs for each dataset and model combination
    for dataset_name, dataset_config in datasets.items():
        for model_config in models:
            create_config(
                template=template,
                dataset_name=dataset_name,
                model_config=model_config,
                dataset_config=dataset_config,
                output_dir=output_dir,
                config_type_folder="/fsl_train/",
                template_type="fsl_train"
            )

if __name__ == "__main__":
    main()
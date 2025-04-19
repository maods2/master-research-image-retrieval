import os
from pathlib import Path
from typing import Dict, List
from ruamel.yaml import YAML

def load_template(template_path: str) -> dict:
    """Load the YAML template file."""
    yaml = YAML()
    with open(template_path, 'r') as f:
        return yaml.load(f)

def create_config(
        template: dict, 
        dataset_name: str, 
        model_config: Dict[str, str], 
        dataset_config: Dict[str, str], 
        output_dir: str,
        config_type_folder: str = "retrieval_test"
        ):
    """Create a new config file with customized parameters."""
    # Copy the template to preserve its structure
    config = template.copy()
    
    # Update paths in the 'data' section
    config['data']['train_dir'] = dataset_config['train_dir']
    config['data']['test_dir'] = dataset_config['test_dir']
    config['data']['class_mapping'] = dataset_config['class-mapping']
    
    # Update fields in the 'model' section
    config['model']['name'] = model_config['model_name']
    config['model']['experiment_name'] = f"{model_config['model_name']}_{dataset_name}"
    config['model']['num_classes'] = len(dataset_config['class-mapping'])
    config['model']['model_name'] = model_config["model_pretreined"]
    
    # Update fields in the 'testing' section
    config['testing']['embeddings_path'] = f'./artifacts/{dataset_name}/embeddings_{model_config["model_name"]}'
    config['testing']['embeddings_save_path'] = f'./artifacts/{dataset_name}/embeddings_{model_config["model_name"]}'
    
    # Update fields in the 'output' section
    config['output']['model_dir'] = f'./artifacts/{dataset_name}'
    config['output']['results_dir'] = f'./local_experiments/{dataset_name}'
    
    # Create output directory if it doesn't exist
    dataset_config_dir = os.path.join(output_dir, dataset_name + config_type_folder)
    os.makedirs(dataset_config_dir, exist_ok=True)
    
    # Save the config while preserving the structure and using inline arrays
    output_path = os.path.join(dataset_config_dir, f"{model_config['model_name']}_config.yaml")
    yaml = YAML()
    yaml.default_flow_style = None  # Use block style for objects but inline style for arrays
    with open(output_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Created config: {output_path}")

def main():
    # Define the template path
    template_path = "./configs/templates/retrieval_test/default_model_config.yaml"
    
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
        "glomerulo":{
            "class-mapping": {"Crescent": 0, "Hypercellularity": 1, "Membranous": 2, "Normal": 3,"Podocytopathy": 4, "Sclerosis": 5},
            "train_dir": "datasets/final/glomerulo/train",
            "test_dir": "datasets/final/glomerulo/test"
        },
        # Add more datasets as needed
    }
    
    # Define models to test
    models = [
        {"model_name": "resnet50", "model_pretreined": ""},
        {"model_name": "dino", "model_pretreined": "vit_small_patch16_224_dino"},
        {"model_name": "dinov2", "model_pretreined": "dinov2_vitl14"},
        {"model_name": "uni", "model_pretreined": "vit_large_patch16_224"},
        {"model_name": "clip", "model_pretreined": "openai/clip-vit-base-patch32"},
        {"model_name": "virchow2", "model_pretreined": "hf-hub:paige-ai/Virchow2"},
        {"model_name": "vit", "model_pretreined": "vit_base_patch16_224"},
    ]
    
    # Load template
    template = load_template(template_path)
    
    # Generate configs for each dataset and model combination
    for dataset_name, dataset_config in datasets.items():
        for model_config in models:
            create_config(
                template=template,
                dataset_name=dataset_name,
                model_config=model_config,
                dataset_config=dataset_config,
                output_dir=output_dir,
                config_type_folder="/retrieval_test/"
            )

if __name__ == "__main__":
    main()
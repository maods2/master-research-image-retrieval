import os
from pathlib import Path
from typing import Dict
from ruamel.yaml import YAML


def update_config_fields(config: dict, dataset_name: str, model_config: Dict[str, str], dataset_config: Dict[str, str], experiment_name: str) -> dict:
    """Update configuration fields with provided values."""
    # Update paths in the 'data' section
    config['data']['train_dir'] = dataset_config['train_dir']
    config['data']['test_dir'] = dataset_config['test_dir']
    config['data']['class_mapping'] = dataset_config['class-mapping']
    
    # Update fields in the 'model' section
    config['model']['name'] = model_config['model_name']
    config['model']['experiment_name'] = f"{model_config['model_name']}_{dataset_name}"
    config['model']['num_classes'] = len(dataset_config['class-mapping'])
    config['model']['model_name'] = model_config["model_pretreined"]
    config['model']['checkpoint_path'] = model_config["checkpoint_path"]
    config['model']['load_checkpoint'] = model_config["load_checkpoint"]
    
    # Update fields in the 'testing' section
    config['testing']['embeddings_path'] = f'./artifacts/{dataset_name}/embeddings_{model_config["model_name"]}'
    config['testing']['embeddings_save_path'] = f'./artifacts/{dataset_name}/embeddings_{model_config["model_name"]}'
    
    # Update fields in the 'output' section
    config['output']['model_dir'] = f'./artifacts/{experiment_name}_{dataset_name}'
    config['output']['results_dir'] = f'./local_experiments/{experiment_name}_{dataset_name}'
    
    return config

def update_config_fields_fsl_train(config: dict, dataset_name: str, model_config: Dict[str, str], dataset_config: Dict[str, str]) -> dict:
    """Update configuration fields with provided values."""
    # Update paths in the 'data' section
    config['data']['train_dir'] = dataset_config['train_dir']
    config['data']['test_dir'] = dataset_config['test_dir']
    config['data']['class_mapping'] = dataset_config['class-mapping']
    
    # Update fields in the 'model' section
    config['model']['name'] = model_config['model_name']
    config['model']['experiment_name'] = f"{model_config['model_name']}_{dataset_name}"
    config['model']['num_classes'] = len(dataset_config['class-mapping'])
    config['model']['n_way'] = len(dataset_config['class-mapping'])
    config['model']['model_name'] = model_config["model_pretreined"]
    config['model']['k_queries'] = 128 # used to be 32 but changed to 64 in better GPU
    
    # Update fields in the 'testing' section
    config['testing']['embeddings_path'] = f'./artifacts/{dataset_name}/embeddings_{model_config["model_name"]}'
    config['testing']['embeddings_save_path'] = f'./artifacts/{dataset_name}/embeddings_{model_config["model_name"]}'
    config['testing']['normalize_embeddings'] = True
    config['testing']['enabled'] = True
    
    # Update fields in the 'output' section
    config['output']['model_dir'] = f'./artifacts/{dataset_name}'
    config['output']['results_dir'] = f'./local_experiments/{dataset_name}'
    
    return config

def create_config(
        template: dict, 
        dataset_name: str, 
        model_config: Dict[str, str], 
        dataset_config: Dict[str, str], 
        output_dir: str,
        config_type_folder: str = "/retrieval_test/",
        template_type: str = "retrieval_test",
        experiment_name: str = None
        ):
    """Create a new config file with customized parameters."""
    # Copy the template to preserve its structure
    config = template.copy()
    

    # In the original function, replace the placeholder with:
    if template_type == "retrieval_test":
        config = update_config_fields(config, dataset_name, model_config, dataset_config, experiment_name) 
        
    elif template_type == "fsl_train":
        config = update_config_fields_fsl_train(config, dataset_name, model_config, dataset_config)
    
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
import torch

def save_artifacts(model, results, output_config):
    model_path = output_config["model_path"]
    results_path = output_config["results_path"]
    
    # Save the model state dict
    torch.save(model.state_dict(), model_path)
    
    # Save any results you want, e.g., test performance or retrieval results
    with open(results_path, "w") as f:
        f.write(str(results))

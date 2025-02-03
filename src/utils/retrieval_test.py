import sys
sys.path.append("/workspaces/master-research-image-retrieval/src")
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from metrics.precision_at_k import PrecisionAtK

# Define dataset and transformations
def get_dataloader(dataset_name="CIFAR10", train=True, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset_class = getattr(datasets, dataset_name, None)
    if dataset_class is None:
        raise ValueError(f"Dataset {dataset_name} not found in torchvision.datasets")
    
    dataset = dataset_class(root="./data", train=train, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return dataloader

# Load a pre-trained model for feature extraction
def get_feature_extractor():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 512)
    model.eval()
    return model

# Extract features for image retrieval
def extract_features(dataloader, model, device="cuda"): 
    model.to(device)
    features, labels = [], []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            output = model(images).squeeze(-1).squeeze(-1)  # Flatten the features
            features.append(output.cpu())
            labels.append(targets.cpu())
    
    return torch.cat(features), torch.cat(labels)

if __name__ == "__main__":
    repository_dataloader = get_dataloader("CIFAR10", train=True)
    queries_dataloader = get_dataloader("CIFAR10", train=False)
    
    model = get_feature_extractor()
    # repository_features, repository_labels = extract_features(repository_dataloader, model)
    # queries_features, queries_labels = extract_features(queries_dataloader, model)
    
    print("Feature extraction completed.")
    class Log:
        def info(self, message):
            print(message)
            
    precision_at_k = PrecisionAtK([1, 5, 10])
    precision_at_k(model, repository_dataloader, queries_dataloader, None,  Log())


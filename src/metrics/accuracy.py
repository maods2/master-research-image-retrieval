from typing import Dict, Any
import torch

class Accuracy:
    def __init__(self, **kwargs):
        """
        Initialize the F1Score metric.
        
        Args:
            **kwargs: Additional properties that are not explicitly required by this class.
        """
        pass
    
    def __call__(self, model: Any, train_loader: Any, test_loader: Any, config: Dict[str, Any], logger: Any) -> Dict[str, float]:
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels.argmax(1)).sum().item()

        accuracy = correct / total
        logger.info(f"Accuracy: {accuracy:.4f}")
        return {"accuracy": accuracy}

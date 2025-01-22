from typing import Dict, Any
import torch
from sklearn.metrics import precision_recall_fscore_support

class F1Score:
    def __call__(self, model: Any, train_loader: Any, test_loader: Any, config: Dict[str, Any], logger: Any) -> Dict[str, float]:
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, dim=1)
                all_labels.extend(labels.argmax(1).cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        logger.info(f"F1 Score: {f1:.4f}")
        return {"f1_score": f1}

# Extending the Framework

This guide explains how to extend the framework by adding custom training logic, datasets, loss functions, models, and testing pipelines. Follow the steps below to integrate your custom components.

## Summary of Steps

1. **Prepare Dataset**: Ensure the dataset is in a compatible format and place it in the `data/` directory.  
2. **Create Dataset Loader**: Implement a dataset class inheriting `StandardImageDataset` in `src/data_loaders/`.  
3. **Register Dataset**: Add the dataset to `dataset_factory.py`.  
4. **Create Model File**: Define a model class in `models/` with the desired architecture.  
5. **Register Model**: Add the model to `factory.py`.  
6. **Create Loss Function**: Implement a custom loss function in `losses/`.  
7. **Register Loss**: Add the loss function to `loss_factory.py`.  
8. **Define Training Logic**: Extend `BaseTrainer` and implement `__call__` and `train_one_epoch`.  
9. **Register Training Pipeline**: Add the pipeline to `train_factory.py`.  
10. **Update Configuration**: Create a YAML file specifying the dataset, model, loss, and training pipeline.  
11. **Run Training**: Execute the training script with the configuration file or use `make`.  
12. **Automate with Makefile**: Add a target in the `Makefile` for the custom training pipeline.  

## 1. **Adding a New Dataset**

### Step 1: **Prepare Your Dataset**
1. Ensure your dataset is in a compatible format (e.g., CSV, JSON, or image files).
2. Place your dataset in the `data/` directory or provide a path in your configuration.

### Step 2: **Create a Dataset Loader**
1. Navigate to the `src/data_loaders/` directory.
2. Create a new Python file, e.g., `custom_dataset_loader.py`.
3. Implement a class that loads and preprocesses your dataset:
    ```python
    from dataloaders.standard_image_dataset import StandardImageDataset

    class CustomDataset(StandardImageDataset):
        def __init__(self, root_dir, transform=None, class_mapping=None):
            super().__init__(root_dir, transform, class_mapping)
    ```

---

## 2. **Adding a New Model**

### Step 1: **Create a Model File**
1. Navigate to the `src/models/` directory.
2. Create a new Python file, e.g., `custom_model.py`.

### Step 2: **Define Your Model**
1. Implement your model class:
    ```python
    import torch.nn as nn

    class CustomModel(nn.Module):
         def __init__(self):
              super(CustomModel, self).__init__()
              # Define your model architecture
              self.layer = nn.Linear(10, 1)
         
         def forward(self, x):
              return self.layer(x)
    ```

---

## 3. **Adding a New Loss Function**

### Step 1: **Create a Loss Function File**
1. Navigate to the `src/losses/` directory.
2. Create a new Python file, e.g., `custom_loss.py`.

### Step 2: **Implement Your Loss Function**
1. Define your custom loss function:
    ```python
    import torch
    import torch.nn as nn

    class CustomLoss(nn.Module):
         def __init__(self):
              super(CustomLoss, self).__init__()
         
         def forward(self, predictions, targets):
              # Add your custom loss computation logic
              return torch.mean((predictions - targets) ** 2)
    ```

---

## 4. **Define Your Training Logic**

### Step 1: **Create a New Class**
1. Create a new class that extends `BaseTrainer`:
    ```python
    class CustomTrain(BaseTrainer):
        def __init__(self, config: dict):
            self.config = config
            super().__init__()
    ```

### Step 2: **Implement Required Methods**
The `BaseTrainer` class requires the following abstract methods to be implemented:

- `__call__`: Defines the main training loop.
- `train_one_epoch`: Implements the logic for training the model for one epoch.

    ```python
    from tqdm import tqdm
    import torch

    class CustomTrain(BaseTrainer):
        def __init__(self, config: dict):
            self.config = config
            super().__init__()

        def train_one_epoch(self, model, loss_fn, optimizer, train_loader, device, epoch):
            model.train()
            running_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=running_loss / (batch_idx + 1))

            return running_loss / len(train_loader)

        def __call__(self, model, loss_fn, optimizer, train_loader, test_loader, config, logger, metric_logger):
            device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            for epoch in range(config['training']['epochs']):
                epoch_loss = self.train_one_epoch(model, loss_fn, optimizer, train_loader, device, epoch)
                logger.info(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    ```

---

## 5. **Using a Custom Dataset**

### Step 1: **Import Your Dataset**
1. Import your custom dataset in the appropriate file:
    ```python
    from dataloaders.custom_dataset import CustomDataset
    ```

### Step 2: **Use Your Dataset**
1. Use your custom dataset in your data loading logic:
    ```python
    if dataset_name == "custom_dataset":
        dataset_class = CustomDataset
    ```

---

## 6. **Using a Custom Model**

### Step 1: **Import Your Model**
1. Import your custom model in the appropriate file:
    ```python
    from models.custom_model import CustomModel
    ```

### Step 2: **Use Your Model**
1. Use your custom model in your model loading logic:
    ```python
    elif model_name == "custom_model":
        model = CustomModel()
    ```

---

## 7. **Configuration Example**

Here is an example configuration for using the custom dataset, model, and loss function:

```yaml
data:
  dataset_type: "custom_dataset"
  train_dir: "data/train"
  test_dir: "data/test"
  batch_size: 32
  num_workers: 4

model:
  name: "custom_model"
  num_classes: 10

loss:
  name: "custom_loss"

training:
  pipeline: "custom_train"
  epochs: 10
```

Pass the configuration file to the training script:

```bash
python src/pipelines/training.py --config configs/custom_train_config.yaml
```

Alternatively, you can use the following command:

```bash
make train-custom
```

---

## 8. **Using Testing Pipelines**

Testing pipelines allow you to evaluate your model using various metrics. The framework provides a default testing pipeline, but you can also create custom pipelines or metrics.

### Step 1: **Configure the Testing Pipeline**
1. Define the testing configuration in your YAML file:
    ```yaml
    testing:
      pipeline: "default"
      list_of_metrics:
        - type: "accuracy"
          top_k: 5
        - type: "precision@k"
          k: 10
    ```

2. The `list_of_metrics` section specifies the metrics to be used during testing. Each metric requires a `type` and may include additional parameters.

### Step 2: **Run the Testing Pipeline**
1. Use the following command to run the testing pipeline:
    ```bash
    python src/pipelines/testing.py --config configs/custom_test_config.yaml
    ```

2. Alternatively, add a target to the `Makefile` for automation:
    ```makefile
    test-custom:
        python src/pipelines/testing.py --config configs/custom_test_config.yaml
    ```

3. Run the `make` command:
    ```bash
    make test-custom
    ```

---

## 9. **Creating a New Metric**

You can create custom metrics to evaluate your model. Follow these steps:

### Step 1: **Create a Metric File**
1. Navigate to the `src/metrics/` directory.
2. Create a new Python file, e.g., `custom_metric.py`.

### Step 2: **Implement the Metric Class**
1. Define a class that inherits from `MetricBase`:
    ```python
    from metrics.metric_base import MetricBase

    class CustomMetric(MetricBase):
        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2

        def __call__(self, model, train_loader, test_loader, embeddings, config, logger):
            # Implement your metric logic here
            result = {"custom_metric": 0.95}  # Example result
            return result
    ```

2. The `__call__` method must compute the metric and return a dictionary with the results.

### Step 3: **Register the Metric**
1. Add your metric to the `metric_factory.py` file:
    ```python
    metric_modules = {
        'accuracy': 'metrics.accuracy.Accuracy',
        'f1_score': 'metrics.f1_score.F1Score',
        "multilabel_accuracy": "metrics.accuracy.MultilabelAccuracy",
        'map@k': 'metrics.map_at_k.MapAtK',
        'precision@k': 'metrics.precision_at_k.PrecisionAtK',
        'recall@k': 'metrics.recall_at_k.RecallAtK',
        'accuracy@k': 'metrics.accuracy_at_k.AccuracyAtK',
        'custom_metric': 'metrics.custom_metric.CustomMetric',  # Add this line
    }
    ```

2. Ensure the `metric_factory.py` file can dynamically import your metric.

### Step 4: **Use the Custom Metric**
1. Update your YAML configuration to include the custom metric:
    ```yaml
    testing:
      pipeline: "default"
      list_of_metrics:
        - type: "custom_metric"
          param1: 0.5
          param2: 10
    ```

2. Run the testing pipeline as described in the previous section.

---

## 10. **Creating a Custom Testing Pipeline**

If the default testing pipeline does not meet your requirements, you can create a custom pipeline.

### Step 1: **Define a Custom Testing Function**
1. Navigate to the `src/pipelines/testing_pipes/` directory.
2. Create a new Python file, e.g., `custom_test_pipeline.py`.
3. Implement your custom testing function:
    ```python
    from metrics.metric_factory import get_metrics
    from utils.embedding_utils import load_or_create_embeddings

    def custom_test_fn(model, train_loader, test_loader, config, logger, metric_logger):
        # Example: Custom testing logic
        metrics_list = get_metrics(config['testing'])
        device = config['device'] if config.get('device') else ('cuda' if torch.cuda.is_available() else 'cpu')

        embeddings = load_or_create_embeddings(
            model,
            train_loader,
            test_loader,
            config,
            logger,
            device
        )

        for metric in metrics_list:
            results = metric(model, train_loader, test_loader, embeddings, config, logger)
            logger.info(f'Results for {metric.__class__.__name__}: {results}')
            metric_logger.log_metrics(results)
    ```

### Step 2: **Register the Custom Pipeline**
1. Update the `test_pipeline_factory.py` file:
    ```python
    from pipelines.testing_pipes.custom_test_pipeline import custom_test_fn

    def get_test_function(testing_config: Dict):
        if testing_config['pipeline'] == 'default':
            return default_test_fn
        elif testing_config['pipeline'] == 'custom':
            return custom_test_fn
        else:
            raise ValueError(f"Testing pipeline {testing_config['pipeline']} is not supported")
    ```

### Step 3: **Use the Custom Pipeline**
1. Update your YAML configuration to use the custom pipeline:
    ```yaml
    testing:
      pipeline: "custom"
      list_of_metrics:
        - type: "custom_metric"
          param1: 0.5
          param2: 10
    ```

2. Run the testing pipeline as described earlier.

---

By following these steps, you can extend the framework to include custom testing pipelines and metrics. Ensure that your additions are thoroughly tested for compatibility and correctness.
# Master's Degree Research: Medical Image Retrieval

## Pending Items:

- [ ] Validate only test pipeline
- [ ] Implement inference pipeline
        - **File/Folder**: `/inference/`
        - Tasks:
                1. Model Classification
                2. Embedding extraction
- [ ] Test multilabel training. Check if inference results can include 2 classes
        - **File/Folder**: `/training/`
- [ ] Validate retrieval functions
        - **File/Folder**: `/retrieval/`
- [ ] Validate performance functions (Accuracy, F1, etc.)
        - **File/Folder**: `/evaluation/`
- [ ] Implement architecture
        - **File/Folder**: `/models/`
        - Tasks:
                1. Model
                2. Training Pipeline
- [ ] Process raw datasets and create dataloaders
        - **File/Folder**: `/data/`
- [ ] Check possibilities to simplify framework
        - **File/Folder**: `/framework/`
- [ ] Implement retrieval demo
        - **File/Folder**: `/demo/`
- [ ] Plan how to store embedding databases to save processing time
        - **File/Folder**: `/storage/`
- [ ] Improve how statistics are logged into MLflow for better comparison views
        - **File/Folder**: `/logging/`

## How to Use the Framework

### 1. Create a New Model
- **File/Folder**: `/models/`
- Implement your model by extending the base model class or creating a standalone model.
- Add the new model to the **Model Factory** to ensure it can be instantiated dynamically.

### 2. Add Model to the Model Factory
- **File/Folder**: `/models/factory.py`
- Register your model in the **Model Factory** by adding an entry for it in the factory's configuration or code.

### 3. Create a Training Pipeline
- **File/Folder**: `/training/`
- Use the base training pipeline class as an interface to create a custom training pipeline for your model.
- Alternatively, use the default training pipeline provided by the framework.

### 4. Create a Dataloader
- **File/Folder**: `/data/`
- Implement a new dataloader if your dataset requires custom preprocessing or loading logic.
- Add the new dataloader to the **Dataloader Factory** for seamless integration.
- If applicable, use predefined dataloaders for standard datasets.

### 5. Add a New Metric (if needed)
- **File/Folder**: `/metrics/`
- Extend the **Metric Base Class** to create a new metric class.
- Register the new metric in the **Metric Factory** to make it available for evaluation.

### 6. Define the Configuration
- **File/Folder**: `/config/`
- Use the configuration system to define all necessary parameters for your model, training pipeline, dataloader, and metrics.
- Ensure the configuration is consistent and well-documented for reproducibility.
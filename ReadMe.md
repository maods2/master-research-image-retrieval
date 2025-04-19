# Master Research Image Retrieval

This repository facilitates experiments and the creation of new models for image retrieval tasks. It provides a flexible framework for managing experiments, debugging, and analyzing results. The repository supports saving experiments in organized folders or using tools like MLflow, with the capability to integrate other tools as needed.

## Key Features

### 1. Experiment Management
- Experiments are saved in the `local_experiments` folder, categorized by dataset and model configurations.
- Supports logging and tracking experiments using MLflow or other tools specified in the configuration files.

### 2. Framework Overview
- The repository is structured to streamline the development and debugging of new models.
- Refer to the documentation files in the `src/archive` folder for detailed explanations of the framework and debugging processes.

### 3. Experiment Results
- Results from experiments are compiled and analyzed in the `notebooks` folder.
- Notebooks include tools for retrieval analysis, embedding visualization, and explainability pipelines.

### 4. Demo Applications
- The `demo` folder contains applications for image retrieval and embedding exploration using the FiftyOne library.
- These demos allow users to visualize embeddings, explore datasets, and retrieve similar images based on embeddings.

## Folder Structure

- **`local_experiments/`**: Organized storage for experiment results, categorized by dataset and model.
- **`notebooks/`**: Contains Jupyter notebooks for analyzing experiment results, including retrieval analysis and embedding visualization.
- **`demo/`**: Includes demo scripts for retrieval and embedding exploration using FiftyOne.
- **`configs/`**: Configuration files for various datasets and models, enabling easy customization of experiments.
- **`src/`**: Core framework for model development, training, and debugging.

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo-url.git
cd master-research-image-retrieval
```

### 2. Install Dependencies
Follow the instructions in the repository to install the required dependencies.

### 3. Run a Demo
Navigate to the `demo` folder and execute the scripts to explore image retrieval and embedding visualization.

### 4. Explore Experiment Results
Open the notebooks in the `notebooks` folder to analyze results and visualize embeddings.

## Documentation

- **Framework and Debugging**: Refer to the documentation files in the `docs` folder for detailed insights:
  - [Debugging Pipelines](docs/DebuggingPipelines.md): Explains the debugging processes for the framework.
  - [Extending the Framework](docs/ExtendingtheFramework.md): Provides guidance on how to extend the framework for custom use cases.

- **Configuration Files**: The `configs` folder contains templates and examples for setting up experiments with different datasets and models.

## Tools and Libraries

- **MLflow**: For experiment tracking and logging.
- **FiftyOne**: For dataset exploration and embedding visualization.
- **PyTorch**: For model development and training.
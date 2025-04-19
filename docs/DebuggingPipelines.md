# Debugging Training and Testing Pipelines Using `.vscode`

This guide explains how to set up and use the Visual Studio Code debugger to debug the training and testing pipelines in the current codebase. It leverages the `main.py` script and the `Makefile` as references for passing the necessary arguments.

---

## 1. Understanding the Debugging Context

The `main.py` script is the entry point for running both the training and testing pipelines. It requires two arguments:

- `--config`: Path to the configuration file.
- `--pipeline`: Specifies the pipeline to run (`train` or `test`).

The `Makefile` provides examples of how these arguments are passed when running the pipelines. For example:

```bash
make train CONFIG=configs/default_train_config.yaml
make test CONFIG=configs/default_test_config.yaml
```

---

## 2. Setting Up Debugging in `launch.json`

To debug the pipelines, you need to configure the `launch.json` file in the `.vscode` folder. Follow these steps:

### Step 1: Open or Create `launch.json`

1. Navigate to the `.vscode` folder in your project directory.
2. Open the `launch.json` file. If it doesn't exist, create it.

### Step 2: Add Debug Configurations

Add the following configurations for debugging the training and testing pipelines:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Training Pipeline",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "args": [
                "--config",
                "configs/default_train_config.yaml",
                "--pipeline",
                "train"
            ]
        },
        {
            "name": "Debug Testing Pipeline",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "args": [
                "--config",
                "configs/default_test_config.yaml",
                "--pipeline",
                "test"
            ]
        }
    ]
}
```

---

## 3. Customizing Debug Arguments

### Training Pipeline

- **Config File**: Use the appropriate training configuration file (e.g., `configs/default_train_config.yaml` or `configs/multilabel/train_config.yaml`).
- **Pipeline**: Set `--pipeline` to `train`.

Example:

```bash
python main.py --config configs/default_train_config.yaml --pipeline train
```

### Testing Pipeline

- **Config File**: Use the appropriate testing configuration file (e.g., `configs/default_test_config.yaml` or `configs/retrieval_test/vit_config.yaml`).
- **Pipeline**: Set `--pipeline` to `test`.

Example:

```bash
python main.py --config configs/default_test_config.yaml --pipeline test
```

---

## 4. Using the `Makefile` as a Reference

The `Makefile` provides predefined commands for running the pipelines. You can use these as a reference to determine the correct configuration file and pipeline type.

### Example: Training with Triplet Loss and ResNet

From the `Makefile`:

```bash
make train CONFIG=configs/triplet_loss_resnet.yaml
```

To debug this in VS Code, update the `args` in the `Debug Training Pipeline` configuration:

```json
"args": [
    "--config",
    "configs/triplet_loss_resnet.yaml",
    "--pipeline",
    "train"
]
```

---

## 5. Starting the Debugger

1. Open the **Run and Debug** panel in VS Code (`Ctrl+Shift+D` or `Cmd+Shift+D` on macOS).
2. Select the desired configuration (`Debug Training Pipeline` or `Debug Testing Pipeline`).
3. Click the green **Start Debugging** button or press `F5`.

---

## 6. Tips for Debugging

- **Breakpoints**: Set breakpoints in `main.py`, `training.py`, `testing.py`, or any other relevant file to inspect the execution flow.
- **Inspect Variables**: Use the debug console to inspect variables and evaluate expressions.
- **Modify Configurations**: If needed, modify the `args` in `launch.json` to test different configurations or pipelines.

---

By following this guide, you can effectively debug the training and testing pipelines in your codebase using Visual Studio Code.
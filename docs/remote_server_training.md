# Remote Server Training Guide



This guide provides instructions for setting up and running training scripts on a remote server. It includes steps for managing terminal sessions with `tmux`, preparing the environment, and executing training commands.

---

## 1. Clone the Repository

Access the server (Ivision Odin machine)
```
ssh matheus@10.131.20.77 
#or
ssh ivision@10.131.20.77
```
Start by cloning the repository to your local or remote machine:

```bash
git clone https://github.com/maods2/master-research-image-retrieval.git
cd master-research-image-retrieval
```

## 2. Using tmux

If you want to run the training script on a remote server, consider using tmux to manage your terminal session. This allows you to easily reconnect to your session if your connection to the remote server is interrupted.

### Install tmux

Install tmux if it is not already installed:

```bash
sudo apt-get update && sudo apt-get install tmux
```

### Start a tmux Session

Start a new tmux session with a custom name:

```bash
tmux new -s <my-session>
tmux new -s fsl_part1
tmux new -s fsl_part2
```

### Attach to an Existing Session

If you already have a tmux session running, you can attach to it using:

```bash
tmux attach -t <my-session>
tmux attach -t fsl_part1
tmux attach -t fsl_part2
```

### Other tmux Commands

- **Enter Copy Mode**: Press `Ctrl + b`, then `[`. This allows you to scroll through the terminal output and select text.
- **Navigate in Copy Mode**:
  - **Scroll Up**: Use the Up Arrow key or press `Ctrl + b` followed by `Page Up`.
  - **Scroll Down**: Use the Down Arrow key or press `Ctrl + b` followed by `Page Down`.
- **Exit Copy Mode**: Press `q` or `Enter` to return to normal mode.

## 3. Provide Executable Permission

Before running the GPU container, ensure that the script has executable permissions:

```bash
chmod +x run_gpu_container.sh
```

## 4. Run the GPU Container

Execute the script to run the GPU container:

```bash
./run_gpu_container.sh
```

## 5. Download the Dataset

Use the following command to download the dataset required for training:

```bash
make download-datasets
```


## 7. Run Training
To start training 
```bash
make hf-login (paste token)
make train-all-datasets-models-part1
make train-all-datasets-models-part2

```
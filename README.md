# Latent Space Exploration and Interpolation Project

This project is designed to explore and analyze the latent space produced by an autoencoder. The primary goals are to:

- **Analyze the latent space:** Visualize and characterize the latent representations (shape: (4, 32, 32)) using techniques such as dimensionality reduction (PCA, UMAP, t-SNE), clustering, and manifold learning.
- **Perform latent interpolation:** Compare various interpolation strategies between latent1, latent2, and latent3. These strategies include both deep learning–based models and non–deep learning methods (e.g., SLERP, Cosine, Cubic Hermite, and RBF interpolation).
- **Evaluate interpolation quality:** Decode interpolated latent vectors using a trained VAE decoder and visualize the decoded outputs to assess how the interpolation paths compare.
- **Automate training and evaluation:** Use command-line tools (with argparse) to train, evaluate, and compare multiple interpolation models, with checkpoints and early stopping.

Below is the project structure:

```ascii
Latent Space Exploration/
├── LICENSE
├── README.md
├── checkpoints/              # Stores model checkpoints
├── config.yaml               # Configuration for experiments
├── eval_all.sh               # Script to evaluate all models
├── latent_analysis.ipynb     # Notebook for visualizing and analyzing latent space
├── models/                   # Contains model implementations
│   ├── neural_network.py
│   ├── transformer.py
│   ├── unet.py
│   ├── linear.py
│   ├── diffusion.py
│   ├── flow_matching.py
│   ├── transformer_diffusion.py
│   └── __init__.py
├── requirements.txt          # Dependencies for the project
├── results/                  # Stores results and plots
├── train.py                  # Script for training interpolation models
├── train_all.sh              # Script to run multiple training sessions using tmux
├── utils.py                  # Utility functions for training and evaluation
└── vae/                      # Pre-trained VAE implementation
    ├── checkpoints/
    ├── notebooks/
    │   ├── latent.ipynb
    │   ├── vae_tests.ipynb
    │   └── vae.ipynb
    ├── vae_eval.py
    ├── vae_loss.png
    ├── vae_train.py
    └── vae_write_latents.py
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Diffusers (if using pre-trained VAE models)
- scikit-learn, umap-learn, seaborn, plotly, tqdm
- Other dependencies are listed in [/]requirements.txt[/]

### Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd <repository_folder>
```

3. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

4. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Components

### 1. latent_analysis.ipynb

This notebook contains a comprehensive suite of latent space exploration techniques. It includes cells for:
- **Dimensionality Reduction and Clustering:** PCA, UMAP, t-SNE, and k-means clustering.
- **Trajectory Visualization:** Plotting the sequence of latent1, latent2, and latent3, and comparing them with a straight-line interpolation.
- **Distance Metrics, Channel Statistics, and Correlation Matrices:** Detailed quantitative analyses.
- **Manifold Learning:** Isomap, Locally Linear Embedding (LLE).
- **Temporal Smoothness & Curvature Analysis:** Measuring the deviation of the actual latent path from a straight line.
- **Density and Distribution Analysis:** Histograms and KDEs for each channel.
- **Interactive Visualization:** Using Plotly for engaging 2D projections.
- **Comparison of Decoded Outputs:** Visual comparisons between decoded outputs along different interpolation paths.
- **Alternative Interpolation Strategies:** Non–deep learning methods (SLERP, Cosine, Cubic Hermite, RBF) with PCA visualizations.

### 2. train.py

This command-line script trains various deep learning interpolation models that predict latent2 from latent1 and latent3. Key features include:
- **Modular Model Selection:** Models are defined in the `models` folder and selected via an argument (options include `nn`, `transformer`, `unet`, `linear`, etc.).
- **Dataset Handling:** Loads latent representations (.npy files) from the `train`, `val`, and `test` directories.
- **Training and Validation Loops:** Includes checkpointing, early stopping, and metrics plotting.
- **Command-Line Interface:** Use argparse to set parameters such as model type, epochs, learning rate, batch size, and GPU index.
- **Evaluation:** The script can also evaluate a model on the test set using a separate command-line flag.

### 3. Additional Scripts

- **train_all.sh & eval_all.sh:** Shell scripts to automate training and evaluation of multiple models. These scripts can be scheduled (e.g., overnight) and managed with tools like tmux for concurrent execution across multiple GPUs.
- **utils.py:** Contains utility functions for saving/loading checkpoints, early stopping, and plotting training metrics.
- **models Folder:** Contains deep learning interpolation model implementations (e.g., NNModel, TransformerInterp, UNetInterp, DiffusionModel, FlowMatchingModel, TransformerDiffusion, LinearModel).

## Running the Project

### Training a Model

To train an interpolation model, run:
```bash
python train.py --model nn --epochs 200 --learning_rate 1e-3 --batch_size 32 --gpu_index 3 --data_dir /data/ggonsior/atd12k
```

### Evaluating a Model

To evaluate a trained model on the test set:
```bash
python train.py --mode eval --model nn --checkpoint checkpoints/nn_epoch200.pth --batch_size 32 --gpu_index 3 --data_dir /data/ggonsior/atd12k
```

### Exploring the Latent Space

Open the notebook `latent_analysis.ipynb` to run all the exploration cells interactively. This notebook allows you to visualize latent trajectories, clustering, manifold projections, and compare various interpolation strategies.

## Automation with tmux

You can automate training of multiple models overnight using tmux. For example, create a shell script (`train_all.sh`) similar to:

```bash
#!/bin/bash
models=("nn" "transformer" "unet" "linear" "diffusion" "flow" "transformer_diffusion")
for i in "${!models[@]}"; do
    model="${models[$i]}"
    gpu=$(( i % 2 ))  # Alternately assign GPU 0 and GPU 1 (or adjust as needed)
    tmux new-session -d -s "train_${model}" "python train.py --model ${model} --epochs 200 --learning_rate 1e-3 --batch_size 32 --gpu_index ${gpu} --data_dir /data/ggonsior/atd12k"
    echo "Started training for ${model} on GPU ${gpu} in tmux session train_${model}"
done
```

Make the script executable:
```bash
chmod +x train_all.sh
```

Then run:
```bash
./train_all.sh
```

You can attach to any tmux session (e.g., `tmux attach -t train_nn`) to monitor progress.

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgements

This project uses tools and libraries including PyTorch, Diffusers, scikit-learn, UMAP, and Plotly.

Happy exploring!

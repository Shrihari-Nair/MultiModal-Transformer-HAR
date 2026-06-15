# Multi-Modal Transformer for Human Activity Recognition (HAR)

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Project Overview
This repository contains the implementation of a **Multi-Modal Transformer** designed for Human Activity Recognition (HAR). The model leverages **Cross-View Fusion** to jointly learn from two complementary data modalities:
1. **3D Skeletal Kinematics**: Spatial and temporal joint coordinates captured via motion capture (MoCap) systems.
2. **Inertial Sensor Data**: Time-series acceleration signals (x, y, z) captured via wearable sensors.

By fusing these modalities, the model achieves robust and accurate classification of complex human activities, outperforming traditional unimodal baselines.

---

## ✨ Key Features
- **Cross-View Fusion Architecture**: Independently processes skeletal and acceleration streams before fusing their latent representations for a unified classification decision.
- **Spatial-Temporal Modeling**: Utilizes dedicated Spatial and Temporal Transformers to capture both intra-frame joint relationships and inter-frame motion dynamics.
- **Advanced Optimization**: Integrates **ASAM (Adaptive Sharpness-Aware Minimization)** and Cosine Annealing Learning Rate Scheduling to improve model generalization and training stability.
- **Flexible Data Pipeline**: Custom PyTorch `Dataset` class with built-in frame selection, padding, and signal magnitude computation.

---

## 📊 Dataset
The codebase is designed to work with synchronized multi-modal datasets containing:
- **Skeletal Data**: CSV files containing 3D coordinates $(x, y, z)$ for multiple body joints over time.
- **Acceleration Data**: CSV files containing tri-axial acceleration readings aligned with the skeletal frames.
- **Labels**: Categorical activity labels mapped to specific time segments.

*(Note: To use your own dataset, update the file paths and label mapping logic in `config.py` and `preprocessor.py` respectively.)*

---

## 🏗️ Model Architecture
The core model (`Action_Recognition_Transformer`) consists of three main encoder branches:
1. **Spatial Encoder**: Embeds individual joints and applies self-attention to learn structural body pose representations.
2. **Temporal Encoder**: Processes the sequence of spatially embedded frames to capture motion dynamics over time.
3. **Acceleration Encoder**: Processes the raw or feature-extracted acceleration time-series data.
4. **Fusion & Classification**: The classification (`cls`) tokens from the temporal and acceleration encoders are fused (e.g., via concatenation or cross-attention) and passed through a final MLP head for activity classification.

---

## ⚙️ Environment Setup
We recommend using `conda` to manage dependencies and ensure reproducibility.

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd MultiModal-Transformer-HAR
   ```

2. **Create and activate the conda environment**:
   ```bash
   conda env create -f env_setup/env.yml
   conda activate mmt_env
   ```

3. **Install PyTorch** (if not already included in the environment, match your CUDA version):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install additional dependencies** (if missing):
   ```bash
   pip install timm einops
   ```

---

## 🚀 Usage

### 1. Configuration
Before training, ensure the dataset paths are correctly set in `config.py`:
```python
# Example config.py structure
file_paths = {
    'mocap_train_path': 'path/to/train/mocap',
    'mocap_test_path': 'path/to/test/mocap',
    'acc_train_path': 'path/to/train/acc',
    'acc_test_path': 'path/to/test/acc',
    'labels_path': 'path/to/labels.csv'
}
```

### 2. Training
Run the main training script. The script will automatically handle data loading, model initialization, and the ASAM optimization loop.
```bash
python train.py
```
*Checkpoints and logs will be saved in the `exps/<experiment_name>/` directory.*

### 3. Evaluation / Validation
To evaluate a trained model on the test set, use the validation script:
```bash
python validator.py
```

### 4. Visualization
Training metrics (loss, accuracy) and confusion matrices can be generated using the plotter module:
```bash
python plotter/visualizer.py
```

---

## 📂 Project Structure
```text
MultiModal-Transformer-HAR/
├── env_setup/
│   └── env.yml                 # Conda environment specification
├── models/
│   ├── cross_view_fusion.py    # Main multi-modal Transformer architecture
│   ├── fusion.py               # Fusion mechanism utilities
│   ├── model_utils.py          # Transformer blocks, attention layers, etc.
│   ├── unimodal_acc.py         # Baseline acceleration-only model
│   └── unimodal_skeleton.py    # Baseline skeleton-only model
├── plotter/
│   └── visualizer.py           # Scripts for plotting metrics and confusion matrices
├── asam.py                     # ASAM (Adaptive Sharpness-Aware Minimization) optimizer
├── config.py                   # Global configuration and file paths
├── create_dataset.py           # Custom PyTorch Dataset and data augmentation logic
├── preprocessor.py             # Data parsing, ID mapping, and label generation
├── train.py                    # Main training loop
├── validator.py                # Model evaluation and testing script
└── README.md                   # This file
```

---

## 🛠️ Customization & Extensibility
- **Adding New Modalities**: Extend the `Action_Recognition_Transformer` class to include additional encoder branches (e.g., RGB video, EMG signals).
- **Hyperparameter Tuning**: Adjust `mocap_frames`, `acc_frames`, `sdepth`, `tdepth`, and `adepth` in `train.py` or `cross_view_fusion.py` to match your dataset's temporal resolution and complexity.
- **Loss Functions**: The codebase supports standard Cross-Entropy Loss and is pre-configured to easily swap in `LabelSmoothingCrossEntropy` from `timm`.

---

## 📜 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙏 Acknowledgements
- [PyTorch](https://pytorch.org/) for the deep learning framework.
- [timm](https://github.com/rwightman/pytorch-image-models) for advanced loss functions and utilities.
- [einops](https://github.com/arogozhnikov/einops) for elegant and readable tensor operations. 

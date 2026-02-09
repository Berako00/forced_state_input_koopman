# PIFEK - Physics-Informed Forced Extended Koopman

A PyTorch implementation of a Koopman operator-based autoencoder for learning and predicting the dynamics of physical systems with control inputs.

## Overview

This project implements a **Physics-Informed Neural Network** using **Koopman operator theory** to learn dynamical systems. The model uses an autoencoder architecture that lifts states and inputs into a higher-dimensional space where the system dynamics become approximately linear (via the Koopman operator).

### Supported Systems

1. **Simple 2D System**: A basic 2-state dynamical system with 1 control input
2. **Two-Link Robot Manipulator**: A 4-state system (joint angles and velocities) with 2 control inputs (torques)

## Project Structure

```
PIFEK/
├── main.py              # Main entry point for training
├── ga_main.py           # Main entry point with Genetic Algorithm optimization
├── nn_structure.py      # Neural network architecture (AUTOENCODER class)
├── training.py          # Training functions and loops
├── data_generation.py   # Data generation for both systems
├── loss_func.py         # Loss functions (L4, L6 multi-step prediction losses)
├── help_func.py         # Helper functions (self-feeding, model loading)
├── ga_optimizer.py      # Genetic Algorithm for hyperparameter optimization
├── plotting.py          # Visualization utilities
├── debug_func.py        # Debugging functions for loss components
└── requirements.txt     # Python dependencies
```

## Architecture

### Koopman Autoencoder

The model consists of:

- **State Encoder (`x_Encoder`)**: Maps the state `x` to a higher-dimensional observable space `y`
- **Input Encoder (`u_Encoder`)**: Maps the state-input pair `(x, u)` to an observable space `v`
- **State Koopman Operator (`x_Koopman_op`)**: Linear transformation for autonomous state evolution
- **Input Koopman Operator (`u_Koopman_op`)**: Linear transformation capturing the effect of inputs

The prediction model:
```
y_{k+1} = K_x · y_k + K_u · v_k
x_{k+1} = y_{k+1}[:num_meas]
```

### Loss Functions

- **L4**: One-step prediction loss in the lifted space
- **L6**: Multi-step prediction loss over the full trajectory

Total loss: `L_total = α₁·L4 + α₂·L6`

## Installation

```bash
# Clone the repository
git clone https://github.com/Berako00/PIFEK.git
cd PIFEK

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- openpyxl

## Usage

### Basic Training

```bash
python main.py
```

Configuration options in `main.py`:
- `Setup`: Choose between `'Simple'` or `'Twolink'` system
- `numICs`: Number of initial conditions for data generation
- `T_step`: Number of time steps per trajectory
- `dt`: Time step size
- `eps`: Number of training epochs
- `lr`: Learning rate
- `batch_size`: Training batch size

### Training with Genetic Algorithm Optimization

```bash
python ga_main.py
```

The GA optimizes hyperparameters including:
- `Num_x_Obsv`: Dimension of state observable space
- `Num_u_Obsv`: Dimension of input observable space
- `Num_x_Neurons`: Neurons per hidden layer (state encoder)
- `Num_u_Neurons`: Neurons per hidden layer (input encoder)
- `Num_hidden_x`: Number of hidden layers for state encoder
- `Num_hidden_u`: Number of hidden layers for input encoder
- `alpha0`, `alpha1`, `alpha2`: Loss weighting coefficients

GA parameters in `ga_main.py`:
- `generations`: Number of GA generations
- `pop_size`: Population size
- `tournament_size`: Tournament selection size
- `mutation_rate`: Probability of mutation

## Configuration Examples

### Simple System
```python
Setup = 'Simple'
Num_meas = 2        # State dimension
Num_inputs = 1      # Control input dimension
Num_x_Obsv = 3      # State observables
Num_u_Obsv = 3      # Input observables
```

### Two-Link Robot
```python
Setup = 'Twolink'
Num_meas = 4        # [q1, q2, dq1, dq2]
Num_inputs = 2      # [tau1, tau2]
Num_x_Obsv = 14     # State observables
Num_u_Obsv = 66     # Input observables
```

## Two-Link Robot Model

The two-link planar manipulator simulation includes:
- Realistic inertial parameters
- Gravity effects
- Joint friction/damping (B1, B2)
- Gear ratio effects (NG = 172)
- Motor and gearbox inertia

State variables: `[q1, q2, dq1, dq2]` (joint angles and angular velocities)
Control inputs: `[tau1, tau2]` (joint torques)

## Output

- **Model checkpoints**: Saved as `Autoencoder_model_params{i}.pth`
- **Best GA parameters**: Saved to `best_params.txt` (JSON format)
- **Training results**: Exported to `training_results.xlsx`
- **Visualization plots**: Training losses and prediction comparisons

## GPU Support

The code automatically detects and uses CUDA if available:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The Genetic Algorithm supports multi-GPU parallel evaluation of candidates.

## Key Features

- **Physics-Informed**: Encodes known structure (e.g., velocity-position integration) in the Koopman operator
- **Modular Design**: Separate state and input encoders/Koopman operators
- **Automatic Hyperparameter Tuning**: Genetic Algorithm optimization
- **Multi-Step Prediction**: Loss functions encourage long-horizon accuracy
- **Checkpointing**: Best models saved during training based on test loss

## License

This project is part of a Master's Thesis research project.

## Acknowledgments

- Aalborg University
- Koopman operator theory for dynamical systems

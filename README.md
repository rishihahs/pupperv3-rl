# Pupper V3 Reinforcement Learning

A Python package for training and deploying reinforcement learning policies for the Pupper V3 quadruped robot.

## Features

- Configurable MuJoCo-based environment for the Pupper V3 robot
- PPO training using JAX, MJX, and Brax
- Domain randomization for robust policy training
- Policy visualization and evaluation tools
- RTNeural export for deployment on real robots

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Setup

Clone this repository:

```bash
git clone https://github.com/rishihahs/pupperv3-rl.git
cd pupperv3_rl
```

Install the package:

```bash
pip install -e .
```

## Configuration

The project uses a flexible configuration system. Default configuration values are defined in `pupperv3_rl/config/defaults.py`. 

You can override the defaults by:
1. Creating a custom configuration file (see `examples/custom_config.py`)
2. Passing command-line arguments to the training scripts

## Usage

### Training a Policy

To train a new policy:

```bash
# Train with default configuration
python -m pupperv3_rl.scripts.train --output-dir output/my_policy

# Train with custom configuration
python -m pupperv3_rl.scripts.train --config path/to/config.py --output-dir output/my_policy
```

Training progress is logged to Weights & Biases if credentials are provided:

```bash
python -m pupperv3_rl.scripts.train --wandb-entity your_entity --wandb-project your_project
```

### Visualizing a Policy

To visualize a trained policy:

```bash
python -m pupperv3_rl.scripts.visualize --checkpoint output/my_policy/params_1000000.npz --output-dir visualizations
```

### Exporting a Policy for Deployment

To export a policy for deployment on a real robot:

```bash
python -m pupperv3_rl.scripts.export_policy --checkpoint output/my_policy/params_1000000.npz --output policy.json
```

## Project Structure

```
pupperv3_rl/
├── config/                 # Configuration handling
├── environment/            # Robot environment
├── export/                 # Policy export utilities
├── models/                 # Robot model handling
├── training/               # Training utilities
├── visualization/          # Visualization utilities
└── scripts/                # Command-line scripts
```

## Configuration Options

### Simulation Configuration

- `model_repo`: Repository for the robot model
- `max_contact_points`: Maximum number of contact points
- `max_geom_pairs`: Maximum number of geometry pairs
- `physics_dt`: Physics timestep

### Training Configuration

- `environment_dt`: Environment timestep
- `ppo.*`: PPO algorithm parameters
- `resample_velocity_step`: Step interval for resampling velocity commands
- `lin_vel_x_range`: Range for linear velocity in x direction
- `lin_vel_y_range`: Range for linear velocity in y direction
- `ang_vel_yaw_range`: Range for angular velocity
- Additional domain randomization parameters

### Policy Configuration

- `use_imu`: Whether to use IMU in policy
- `observation_history`: Number of stacked observations
- `action_scale`: Action scaling
- `hidden_layer_sizes`: Policy network architecture
- `activation`: Activation function

### Reward Configuration

Multiple reward components with configurable weights:
- `tracking_lin_vel`: Linear velocity tracking
- `tracking_ang_vel`: Angular velocity tracking
- `tracking_orientation`: Orientation tracking
- Multiple regularization terms

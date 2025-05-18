"""Setup script for Pupper V3 RL package."""

from setuptools import setup, find_packages

setup(
    name="pupperv3_rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mujoco>=3.2.7",
        "mujoco-mjx>=3.2.7",
        "brax>=0.12.1",
        "flax>=0.10.2",
        "orbax>=0.1.9",
        "jax>=0.5.0",
        "jaxlib>=0.5.0",
        "matplotlib",
        "mediapy",
        "numpy",
        "ml_collections",
        "etils",
        "wandb",
        "plotly",
    ],
    entry_points={
        "console_scripts": [
            "pupperv3_train=pupperv3_rl.scripts.train:main",
            "pupperv3_visualize=pupperv3_rl.scripts.visualize:main",
            "pupperv3_export=pupperv3_rl.scripts.export_policy:main",
        ],
    },
    python_requires=">=3.8",
)

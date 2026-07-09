# UAV Obstacle Avoidance (mini project)

### Note
Outdated code: 
+ src/train/phase4/batch_evaluate_fast.py
+ src/train/phase4/benchmark_tcr_fast.py
+ src/train/phase4/oda_utils.py
+ src/train/phase5/inference_phase5.py

## Setup

**1. Create a Python Virtual Environment**
It is highly recommended to use Python 3.12 and `conda`:
```bash
conda create -n uav_env python=3.12 -y
conda activate uav_env
```

**2. Install Core Dependencies**
Install JAX, PyTorch, and other necessary libraries:
```bash
pip install pybullet jax jaxlib numpy scipy opencv-python tqdm onnxruntime torch torchvision torchaudio
```

**3. Install Gym PyBullet Drones**
This project relies on `gym-pybullet-drones` for physical drone simulation:
```bash
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones
pip install -e .
cd ..
```

**4. Download the ODA Dataset (For training)**

Clone the original repo:
```bash
git clone https://github.com/tudelft/ODA_Dataset
```

Link to download full dataset:
https://doi.org/10.4121/14214236.v1

## Run Simulation

* 1 Simulation
```bash
python pybullet_sim/run_sim_jax.py
```

* Benchmark 500 Simulations
```bash
python pybullet_sim/batch_benchmark_jax.py
```

* Benchmark Hardware
```bash
python benchmark_pipeline.py
```

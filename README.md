# Physics-Informed Diffusion Models in Spectral Space (PISD)

This repository contains the **official implementation** of the paper ***Physics-Informed Diffusion Models in Spectral Space***.

We introduce physics-informed diffusion models operating **directly in spectral space** for solving forward, inverse, and data assimilation problems governed by PDEs under partial observations. 
This codebase builds upon and extends
[**DiffusionPDE: Generative PDE-Solving Under Partial Observation**](https://arxiv.org/abs/2406.17763).
In particular, we:

* preprocess all data in **spectral space**;
* introduce a **new diffusion architecture** tailored to Fourier coefficients;
* redesign the **inference and conditioning pipeline** to support physics-informed constraints in spectral space.

---

## Repository Structure

```
PISD/
├── configs/                     # YAML configuration files for inference
├── dnnlib/                      # EDM utilities 
├── mean_and_std/                # Dataset statistics (spectral domain)
├── scripts/                     # PDE-specific generation scripts
│   ├── generate_helmholtz_f.py
│   ├── generate_ns_nonbounded_f.py
│   └── generate_poisson_f.py
├── torch_utils/                 # PyTorch utilities
├── trained_models/              # Pre-trained models used in the paper
├── training/                    # Training utilities and loss definitions
├── generate_pde.py              # Main inference entry point
├── merge_data.py                # Spectral preprocessing (Poisson, Helmholtz)
├── merge_data_with_time.py      # Spectral preprocessing (Navier–Stokes)
├── main.py                      # Training launcher
├── train.py                     # Training
├── requirements.txt             # Python dependencies
└── LICENSE.txt
```

---

## Requirements

Install the required Python dependencies with:

```bash
pip install -r requirements.txt
```

The code was developed and tested with:

```text
Python 3.11.5
```

Using this version (or a compatible one) is recommended to avoid compatibility issues.

---

## Data Preparation

### Dataset Download

Download the datasets provided by **DiffusionPDE**:

* Paper: [https://arxiv.org/abs/2406.17763](https://arxiv.org/abs/2406.17763)
* Code & data: [https://github.com/jhhuangchloe/DiffusionPDE](https://github.com/jhhuangchloe/DiffusionPDE)

Place the unzipped train datasets in a local `data/training` directory and the test dataset in `data/test`.

### Spectral Preprocessing

Convert the datasets to spectral space using:

```bash
# Poisson and Helmholtz
python merge_data.py

# Navier–Stokes (time-dependent)
python merge_data_with_time.py
```

Each script creates a directory named:

```text
<problem>-merged-spectral/<strategy_name>/
```

which is used as input for training the diffusion models.

---

## Training

Training is performed using **distributed data parallelism** via `torchrun`.

Examples:

```bash
# Poisson
torchrun --standalone --nproc_per_node=4 train.py --outdir=pretrained-poisson --data=data/poisson-merged-spectral/hyperbolic_44/ --cond=0 --arch=ddpmpp --batch=16 --batch-gpu=4 --tick=10 --snap=50 --dump=100 --duration=5 --ema=0.05

# Helmholtz
torchrun --standalone --nproc_per_node=4 train.py --outdir=pretrained-helmholtz --data=data/helmholtz-merged-spectral/hyperbolic_44/ --cond=0 --arch=ddpmpp --batch=16 --batch-gpu=4 --tick=10 --snap=50 --dump=100 --duration=5 --ema=0.05

# Navier–Stokes (non-bounded)
torchrun --standalone --nproc_per_node=4 train.py --outdir=pretrained-ns-nonbounded --data=data/ns-nonbounded-merged-spectral/ --cond=0 --arch=ddpmpp --batch=16 --batch-gpu=4 --tick=10 --snap=50 --dump=100 --duration=5 --ema=0.05
```

---

## Inference / Generation

To generate solutions from **sparse observations**, run:

```bash
# Poisson
python generate_pde.py --config configs/poisson_f.yaml 

# Helmholtz
python generate_pde.py  --config configs/helmholtz_f.yaml 

# Navier–Stokes
python generate_pde.py --config configs/ns-nonbounded_f.yaml 
```

The weights `zeta_*` balance observation fidelity and PDE consistency.
We refer to the **appendix of the paper** for recommended values depending on the number of observations and the PDE type.

---

## Pretrained Models

All pretrained checkpoints used in the paper are available in:

```text
trained_models/
```

These can be directly used for inference without retraining.

---

## License

This repository is released under the **MIT License** (see `LICENSE.txt`).

The training framework is derived from:

* **DiffusionPDE** — CC BY-NC-SA 4.0
* **EDM** (NVLabs) — CC BY-NC-SA 4.0

See the respective repositories for full license details:

* [https://github.com/jhhuangchloe/DiffusionPDE](https://github.com/jhhuangchloe/DiffusionPDE)
* [https://github.com/NVlabs/edm](https://github.com/NVlabs/edm)



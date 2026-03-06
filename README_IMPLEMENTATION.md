# Quantum Neural Networks for Simulating Many-Body Quantum Systems

Implementation of the paper *"Quantum neural networks to simulate many-body quantum systems"* (arXiv:1805.05462) by Gardas, Rams, and Dziarmaga.

This hybrid classical-quantum algorithm uses:
- **RBM ansatz** for the wave function
- **D-Wave quantum sampler** (or classical sampling) for training
- **Variational Monte Carlo** with stochastic reconfiguration

## Features

✓ Fully connected and D-Wave topology RBM architectures  
✓ Classical samplers (Metropolis-Hastings, Simulated Annealing)  
✓ D-Wave quantum sampler interface  
✓ Transverse field Ising models (1D and 2D)  
✓ Automatic gradient computation (equation 15)  
✓ Training convergence tracking  

## Installation

```bash
pip install numpy scipy matplotlib
# Optional: D-Wave SDK
pip install dwave-system
```

## Quick Start

### 1. Train on 1D Ising with fully connected RBM (Classical sampling)

```bash
python main.py --model 1d --size 8 --h 0.5 --rbm full --sampler classical
```

### 2. Train with D-Wave topology RBM (Sparse connectivity)

```bash
python main.py --model 1d --size 8 --h 0.5 --rbm dwave --sampler classical
```

### 3. Use quantum sampler (D-Wave simulator)

```bash
python main.py --model 1d --size 8 --h 0.5 --rbm full --sampler dwave
```

### 4. Train on 2D Ising model

```bash
python main.py --model 2d --size 16 --h 1.0 --rbm full --sampler classical --iterations 200
```

## Key Components

### Core Modules

- **`ising.py`**: Ising Hamiltonian definitions
  - `TransverseFieldIsing1D`: 1D chain with exact solutions
  - `TransverseFieldIsing2D`: 2D lattice with reference solutions

- **`model.py`**: RBM implementations
  - `RBM`: Abstract base class
  - `FullyConnectedRBM`: All visible-hidden connections
  - `DWaveTopologyRBM`: Sparse, hardware-matched connectivity

- **`sampler.py`**: Sampling backends
  - `ClassicalSampler`: Metropolis-Hastings and Simulated Annealing
  - `DWaveSampler`: Quantum annealer interface
  - `AdaptiveTemperatureSampler`: Temperature-adaptive sampling

- **`encoder.py`**: Variational training
  - `Trainer`: VMC with stochastic reconfiguration
  - `ExperimentRunner`: Batch experiment management

- **`utils/utils.py`**: Helper functions
  - Majority voting for chain breaks
  - Entropy and temperature estimation
  - Training visualization

## Architecture Comparison

Run both architectures on the same system:

```python
from ising import TransverseFieldIsing1D
from model import FullyConnectedRBM, DWaveTopologyRBM
from sampler import ClassicalSampler
from encoder import ExperimentRunner

ising = TransverseFieldIsing1D(size=8, h=0.5)
sampler = ClassicalSampler()
runner = ExperimentRunner()

# Compare architectures
results = runner.compare_architectures(ising, sampler, 
    config={'n_iterations': 50, 'n_samples': 1000})
```

**Expected**: D-Wave topology RBM achieves comparable accuracy with fewer parameters and better noise properties.

## Algorithm Overview

### Wave Function Ansatz

$$\Psi(\bfm v) = \sqrt{\Phi(\bfm v)} = e^{-\bfm a \cdot \bfm v / 2} \prod_j \sqrt{2\cosh(b_j + \bfm W_j \cdot \bfm v)}$$

### Gradient Computation (Eq. 15)

$$\frac{1}{\Psi}\frac{\partial \Psi}{\partial p} = \begin{cases} 
-v_i/2, & p = a_i \\
\tanh(\theta_j)/2, & p = b_j \\
v_i\tanh(\theta_j)/2, & p = W_{ij}
\end{cases}$$

where $\theta_j = b_j + \bfm W_j \cdot \bfm v$.

### Stochastic Reconfiguration Update

$$\bfm w_{k+1} = \bfm w_k - \gamma (\bfm{S}^{-1} \bfm{F})$$

where:
- $\bfm{S}_{ij} = \langle\langle D_i^* D_j \rangle\rangle_\rho - \langle\langle D_i^* \rangle\rangle \langle\langle D_j \rangle\rangle$
- $\bfm{F}_j = \langle\langle E D_j^* \rangle\rangle_\rho - \langle\langle E \rangle\rangle \langle\langle D_j^* \rangle\rangle$

## Configuration

Customize via command line or config dict:

```bash
python main.py \
    --model 1d \
    --size 12 \
    --h 1.0 \
    --rbm full \
    --sampler classical \
    --sampling-method metropolis \
    --iterations 150 \
    --n-samples 2000 \
    --learning-rate 0.15 \
    --seed 123
```

All results saved to `results/` directory as JSON.

## Debugging & Performance

### Check gradient computation:

```python
from model import FullyConnectedRBM
rbm = FullyConnectedRBM(8, 8)
v = (2*np.random.randint(0,2,8) - 1).astype(float)
grad = rbm.gradient_log_psi(v)
# Check shapes: a:(8,), b:(8,), W:(8,8)
```

### Profile sampling speed:

```python
import time
sampler = ClassicalSampler()
t0 = time.time()
samples = sampler.sample(rbm, 1000)
print(f"Sampling 1000 configs: {time.time()-t0:.2f}s")
```

## References

- Gardas et al., *Quantum neural networks to simulate many-body quantum systems*, arXiv:1805.05462
- Carleo & Troyer, *Solving the quantum many-body problem with artificial neural networks*, Science 355.6325 (2017)
- Lanting et al., *Entanglement in a 20-Qubit Superconducting Quantum Computer*, arXiv:2006.14115

## License

Educational/Research use

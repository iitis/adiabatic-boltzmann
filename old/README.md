# Homework: Implement Gardas et al. Paper from Scratch

## Objective

Implement the complete variational quantum neural network algorithm from **"Quantum neural networks to simulate many-body quantum systems"** (arXiv:1805.05462).

You will build:
1. **RBM ansatz** with gradient computation (Eq. 15)
2. **Classical sampling** algorithms (Metropolis, simulated annealing)
3. **Stochastic reconfiguration** training (Eq. 10-13)
4. **Ising models** to learn

## Difficulty Assessment

⭐⭐⭐⭐⭐ **Very Demanding**

This homework covers the full algorithm. It requires:
- Understanding the RBM wave function and its gradients
- Implementing numerical algorithms (matrix operations, linear solvers)
- Debugging scientific code (debugging is harder with numerical computation)
- Knowledge of statistical mechanics (Boltzmann distributions, sampling)

**Estimated time**: 6-12 hours (depending on background)

---

## Files to Implement

### 1. `ising_skeleton.py` - Quantum Spin Models

**Concepts**: Hamiltonian, local energy, exact solutions

**Tasks**:

#### Task 1.1: `TransverseFieldIsing1D.local_energy(v, psi_ratio_fn)`
- Implement the local energy formula for 1D chain
- Use psi_ratio_fn to compute off-diagonal terms
- Remember: E_loc = diagonal + off-diagonal

**Starting point**: 
- Diagonal part: sum over neighbors, multiply v_i * v_{i+1}
- Off-diagonal part: loop over spins, use psi_ratio_fn(v, i)

**Testing**: 
- For h=0, should give -∑_i v_i * v_{i+1}
- For h>>1, should give small energy (weak coupling limit)

#### Task 1.2: `TransverseFieldIsing1D.exact_ground_energy()`
- Integrate dispersion relation numerically
- Use scipy.integrate.quad
- Check dimensions (should scale with system size)

**Hint**: Look up the 1D Ising ground state formula. For h∈(0,2), use the correct dispersion.

#### Task 1.3: `TransverseFieldIsing2D.local_energy(v, psi_ratio_fn)`
- Generalize 1D to 2D (same algorithm, 4 neighbors instead of 2)

#### Task 1.4: `TransverseFieldIsing2D.get_neighbors(idx)`
- Convert 1D index ↔ 2D coordinates
- Handle periodic boundary conditions

---

### 2. `model_skeleton.py` - RBM Ansatz

**Concepts**: Restricted Boltzmann Machine, gradient computation, numerical stability

**Tasks**:

#### Task 2.1: `RBM.log_psi(v)`
- Compute log(Ψ(v)) from the ansatz
- Split into: linear term (a·v) + nonlinear term (cosh terms)
- **Critical**: Use np.logcosh() for numerical stability

**Ansatz**: Ψ(v) = e^(-a·v/2) ∏_j [2·cosh(b_j + W_j·v)]^(1/2)

**Implementation**:
```
log_psi = -0.5 * a·v + 0.5 * ∑_j logcosh(b_j + W_j·v)
```

**Testing**:
- All outputs should be real (no NaN or inf)
- log_psi should be continuous as weights change

#### Task 2.2: `RBM.psi_ratio(v, flip_idx)`
- Compute Ψ(v_flip) / Ψ(v) efficiently
- **Challenge**: Don't compute full log_psi twice
- Think: what changes when one spin flips?

**Efficiency**: This is called thousands of times during training, so optimization matters.

**Testing**:
- Ratio should always be positive (it's |Ψ|^2)
- Compare to exp(log_psi(v_flip) - log_psi(v)) as sanity check

#### Task 2.3: `RBM.gradient_log_psi(v)` - Equation 15
- Implement three cases for derivatives w.r.t. a_i, b_j, W_{ij}

**Equation 15** (from paper):
```
∂log(Ψ)/∂a_i = ?  
∂log(Ψ)/∂b_j = ?  
∂log(Ψ)/∂W_{ij} = ?  
```

**Hints**:
- Use chain rule on log form
- Key insight: d/dx[log(cosh(x))] = tanh(x)
- Precompute θ_j = b_j + W_j·v
- Check dimensions: grad_a is (n_visible,), grad_b is (n_hidden,), grad_W is (n_visible, n_hidden)

**Testing**:
- Numerical gradient check: compare analytical gradient to finite differences
- This is critical! A wrong gradient will break training.

#### Task 2.4: `DWaveTopologyRBM.get_connectivity_mask()`
- Design sparse connectivity pattern (~2 connections per hidden unit)
- Example: hidden unit j connects to visible units j, (j+1)%n_visible

#### Task 2.5: `DWaveTopologyRBM.gradient_log_psi(v)`
- Apply connectivity mask to gradients
- Ensures W stays sparse during training

---

### 3. `sampler_skeleton.py` - Sampling Algorithms

**Concepts**: Metropolis-Hastings, Boltzmann distribution, temperature schedules

**Tasks**:

#### Task 3.1: `ClassicalSampler._metropolis_hastings(rbm, n_samples, config)`

**Algorithm**:
```
1. Initialize v = random ±1 configuration
2. Equilibrate (n_sweeps iterations):
   For each sweep:
     For each spin i:
       Propose flip: v' = v with v_i → -v_i
       Ratio = |Ψ(v')/Ψ(v)|^2 = psi_ratio(v, i)^2
       Accept if random() < min(1, ratio)
       Update v if accepted
3. Collect samples (every n_between sweeps)
```

**Hyperparameters**:
- n_sweeps: 100-1000 (more = better equilibration)
- n_between: 1-10 (spacing between samples)

**Testing**:
- Check that acceptance rate is ~0.5 (indicates good mixing)
- Verify samples are uncorrelated (plot autocorrelation)
- Check energy distribution (should have variance)

#### Task 3.2: `ClassicalSampler._simulated_annealing(rbm, n_samples, config)`

**Algorithm** (temperature-dependent Metropolis):
```
1. Create temperature schedule: T(step) = T_init * (T_final/T_init)^(step/n_steps)
2. For each step:
   Flip one random spin
   Ratio = psi_ratio(v, flip_idx)^2
   Acceptance = min(1, ratio^(1/T))
   Accept if random() < acceptance
   Collect sample periodically
```

**Why works**: 
- High T → easy exploration (noisy)
- Low T → greedy exploitation (steep descent)

**Hyperparameters**:
- T_initial: 1-10 (exploration strength)
- T_final: 0.01-0.1 (final greediness)
- n_steps: 1000-10000

**Testing**:
- At T→∞, samples should be random
- At T→0, samples should cluster around minima
- Energy should decrease over annealing schedule

#### Task 3.3: `DWaveSampler` (Optional)
- For homework, mock it with classical SA
- Real D-Wave integration requires embedding (advanced)

---

### 4. `encoder_skeleton.py` - Variational Training

**Concepts**: Stochastic reconfiguration, natural gradient, matrix methods

**Tasks**:

#### Task 4.1: `Trainer.train()`

**Main VMC loop** (Equations 10-13 from paper):
```
For each iteration:
  1. Sample M configurations: v_1, ..., v_M from RBM
  2. For each sample:
     - Compute local energy: E_loc(v_i) = ising.local_energy(v_i, psi_ratio_fn)
     - Compute gradients: D_i = ∇log(Ψ(v_i))
  3. Assemble matrices S and F
  4. Solve: S·x = F for update direction
  5. Update weights: w ← w - γ·x
  6. Track energy, error metrics
```

**Key insight**: This is **natural gradient descent**, not standard gradient descent!

**Testing**:
- Energy should decrease (or plateau)
- Early iterations should show large changes
- Later iterations should converge

#### Task 4.2: `Trainer._compute_sr_matrices(gradients_list, local_energies)`

**Stochastic Reconfiguration matrices** (Eq. 10-13):

```
S_ij = ⟨⟨D_i* D_j⟩⟩_ρ - ⟨⟨D_i*⟩⟩ ⟨⟨D_j⟩⟩

F_j = ⟨⟨E·D_j⟩⟩_ρ - ⟨⟨E⟩⟩ ⟨⟨D_j⟩⟩
```

**In code**:
```
D = matrix of gradients (M × n_params)
E = array of energies (M,)

mean_D = mean(D, axis=0)  # Average gradient
mean_E = mean(E)           # Average energy

S = (1/M) * D.T @ D - outer(mean_D, mean_D) + λ*I
F = (1/M) * E.T @ D - mean_E * mean_D
```

**Why this works**:
- S is a metric tensor (Fisher information matrix)
- Preconditioning by S^{-1} gives natural gradient
- Faster convergence than steepest descent

**Testing**:
- S should be symmetric (S ≈ S.T)
- S should be positive definite (eigenvalues > λ)
- F should be a real vector

---

## Testing Strategy

### Unit Tests

Test each component independently:

```bash
# Test RBM gradients
python -m pytest homework/test_model.py

# Test sampling
python -m pytest homework/test_sampler.py

# Test Ising models
python -m pytest homework/test_ising.py

# Test training
python -m pytest homework/test_encoder.py
```

### Integration Test

Run end-to-end training:

```bash
python homework/main_skeleton.py --model 1d --size 8 --h 0.5 --iterations 20
```

Expected output:
- Energy decreases over iterations
- Converges toward exact ground state

### Validation

Compare to reference implementation:

```python
from src.model import FullyConnectedRBM as ReferenceRBM
from homework.model_skeleton import FullyConnectedRBM as YourRBM

# Check gradients match
v = np.array([1, -1, 1, -1])
your_grad = YourRBM(...).gradient_log_psi(v)
ref_grad = ReferenceRBM(...).gradient_log_psi(v)
assert np.allclose(your_grad['a'], ref_grad['a'])
```

---

## Debugging Guide

### Common Issues

**1. Gradients are all NaN**
- Check: are you computing log(cosh) correctly?
- Try: start with small weights (easier to debug)
- Use: print intermediate values

**2. Training doesn't converge**
- Check: is sampling working? (print sample statistics)
- Check: is S matrix singular? (check eigenvalues)
- Try: reduce learning rate
- Try: increase regularization λ

**3. Acceptance rate is 0% or 100%**
- Check: is psi_ratio computed correctly?
- Try: vary field strength h
- Look at ratio distribution (histogram)

**4. Results don't match reference**
- Numerical gradient check on each function
- Compare intermediate matrices (S, F)
- Check dimensions of all arrays

### Numerical Stability Tips

- Use `np.logcosh()` instead of `log(cosh(x))`
- Use `np.log(ratio)` instead of computing `ratio`
- Add small regularization to S: `S += 1e-5*I`
- Clamp acceptance probabilities: `min(1.0, ratio)`

---

## Success Criteria

### Minimum (to pass):
✅ RBM computes log_psi without NaN  
✅ Gradients match numerical finite differences  
✅ Metropolis sampler produces valid configs (±1 values)  
✅ SR training reduces energy (≥10% improvement in 50 iterations)  

### Good:
✅ Energy converges toward exact ground state (within 5%)  
✅ 1D system: energy matches analytical solution  
✅ Sparse RBM uses fewer parameters than dense  

### Excellent:
✅ 2D system training works  
✅ Simulated annealing outperforms Metropolis  
✅ Stochastic convergence (convergence despite sampling noise)  
✅ Nice visualizations of training history  

---

## References

**Paper**:
- Gardas, Rams, Dziarmaga (2018): arXiv:1805.05462

**Key equations**:
- Eq. 6-7: RBM ansatz
- Eq. 15: Gradients (you must implement!)
- Eq. 10-13: Stochastic reconfiguration

**Background**:
- Carleo & Troyer (2017): Solving quantum many-body with NNs
- Liang et al. (2016): Variational neural ansätze
- Melchert et al. (2013): Efficient sampling in VMC

**Numerical methods**:
- [NumPy guide](https://numpy.org/doc/stable/)
- [SciPy linear algebra](https://docs.scipy.org/doc/scipy/reference/linalg.html)
- [Numerical recipes in Python](https://numerical.recipes/)

---

## Tips for Success

1. **Start small**: Use size=4 or 8 systems while debugging
2. **Verify each component**: Don't move to training until sampling works
3. **Print intermediate values**: Add debug output during first run
4. **Compare to reference**: Keep the original implementation nearby
5. **Commit your progress**: Make git commits after each task
6. **Ask questions**: Reference the paper equations, not intuition

Good luck! This is the real algorithm used in quantum machine learning research. 🚀

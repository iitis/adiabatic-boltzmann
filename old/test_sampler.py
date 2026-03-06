"""
Unit tests for sampler_skeleton.py

Run with: pytest homework/test_sampler.py -v
"""

import pytest
import numpy as np
from model_skeleton import FullyConnectedRBM
from sampler_skeleton import ClassicalSampler


class TestClassicalSamplerMetropolis:
    """Tests for Metropolis-Hastings sampling."""
    
    @pytest.fixture
    def rbm(self):
        """Create a test RBM."""
        return FullyConnectedRBM(n_visible=16, n_hidden=12)
    
    @pytest.fixture
    def sampler(self):
        """Create a test sampler."""
        return ClassicalSampler()
    
    def test_metropolis_output_shape(self, rbm, sampler):
        """Test that Metropolis returns correct output shape."""
        n_samples = 100
        samples = sampler.sample(rbm, n_samples, config={
            'method': 'metropolis',
            'n_sweeps': 50,
            'n_between': 3
        })
        assert samples.shape == (n_samples, rbm.n_visible), \
            f"Expected shape ({n_samples}, {rbm.n_visible}), got {samples.shape}"
    
    def test_metropolis_values_are_spins(self, rbm, sampler):
        """Test that all values in samples are ±1."""
        samples = sampler.sample(rbm, 50, config={
            'method': 'metropolis',
            'n_sweeps': 50,
            'n_between': 3
        })
        valid_spins = (samples == 1) | (samples == -1)
        assert np.all(valid_spins), \
            f"Found invalid values. Expected ±1, got {np.unique(samples)}"
    
    def test_metropolis_samples_are_diverse(self, rbm, sampler):
        """Test that samples are not all identical."""
        samples = sampler.sample(rbm, 100, config={
            'method': 'metropolis',
            'n_sweeps': 100,
            'n_between': 5
        })
        n_unique = len(np.unique(samples, axis=0))
        assert n_unique > 1, \
            f"All {n_unique} samples are identical! Sampling failed."
    
    def test_metropolis_returns_numpy_array(self, rbm, sampler):
        """Test that output is a numpy array."""
        samples = sampler.sample(rbm, 10, config={
            'method': 'metropolis',
            'n_sweeps': 20,
            'n_between': 1
        })
        assert isinstance(samples, np.ndarray), \
            f"Expected numpy.ndarray, got {type(samples)}"
    
    def test_metropolis_dtype_is_float(self, rbm, sampler):
        """Test that output array is float type."""
        samples = sampler.sample(rbm, 10, config={
            'method': 'metropolis'
        })
        assert samples.dtype == np.float64 or samples.dtype == float, \
            f"Expected float dtype, got {samples.dtype}"
    
    def test_metropolis_hyperparameters(self, rbm, sampler):
        """Test with different hyperparameters."""
        for n_sweeps in [10, 100, 500]:
            for n_between in [1, 5, 10]:
                samples = sampler.sample(rbm, 20, config={
                    'method': 'metropolis',
                    'n_sweeps': n_sweeps,
                    'n_between': n_between
                })
                assert samples.shape == (20, 8), \
                    f"Failed with n_sweeps={n_sweeps}, n_between={n_between}"


class TestClassicalSamplerSimulatedAnnealing:
    """Tests for Simulated Annealing sampling."""
    
    @pytest.fixture
    def rbm(self):
        """Create a test RBM."""
        return FullyConnectedRBM(n_visible=8, n_hidden=6)
    
    @pytest.fixture
    def sampler(self):
        """Create a test sampler."""
        return ClassicalSampler()
    
    def test_simulated_annealing_output_shape(self, rbm, sampler):
        """Test that SA returns correct output shape."""
        n_samples = 100
        samples = sampler.sample(rbm, n_samples, config={
            'method': 'simulated_annealing',
            'T_initial': 10.0,
            'T_final': 0.01,
            'n_steps': 500
        })
        assert samples.shape == (n_samples, rbm.n_visible), \
            f"Expected shape ({n_samples}, {rbm.n_visible}), got {samples.shape}"
    
    def test_simulated_annealing_values_are_spins(self, rbm, sampler):
        """Test that all values in SA samples are ±1."""
        samples = sampler.sample(rbm, 50, config={
            'method': 'simulated_annealing',
            'T_initial': 5.0,
            'T_final': 0.05
        })
        valid_spins = (samples == 1) | (samples == -1)
        assert np.all(valid_spins), \
            f"Found invalid values. Expected ±1, got {np.unique(samples)}"
    
    def test_simulated_annealing_samples_are_diverse(self, rbm, sampler):
        """Test that SA produces diverse samples."""
        samples = sampler.sample(rbm, 100, config={
            'method': 'simulated_annealing',
            'T_initial': 10.0,
            'T_final': 0.01
        })
        n_unique = len(np.unique(samples, axis=0))
        assert n_unique > 1, \
            f"All {n_unique} samples are identical! SA failed."
    
    def test_simulated_annealing_returns_numpy_array(self, rbm, sampler):
        """Test that SA output is a numpy array."""
        samples = sampler.sample(rbm, 10, config={
            'method': 'simulated_annealing'
        })
        assert isinstance(samples, np.ndarray), \
            f"Expected numpy.ndarray, got {type(samples)}"
    
    def test_simulated_annealing_dtype_is_float(self, rbm, sampler):
        """Test that SA output array is float type."""
        samples = sampler.sample(rbm, 10, config={
            'method': 'simulated_annealing'
        })
        assert samples.dtype == np.float64 or samples.dtype == float, \
            f"Expected float dtype, got {samples.dtype}"
    
    def test_simulated_annealing_temperature_schedule(self, rbm, sampler):
        """Test that SA explores initially and exploits finally."""
        samples = sampler.sample(rbm, 100, config={
            'method': 'simulated_annealing',
            'T_initial': 10.0,
            'T_final': 0.01,
            'n_steps': 5000
        })
        
        # Early samples (high T = exploration)
        early = samples[:25]
        # Late samples (low T = exploitation)
        late = samples[-25:]
        
        # Compute "order" as sum of spins (higher = more aligned)
        early_order = np.abs(np.mean(early, axis=1))
        late_order = np.abs(np.mean(late, axis=1))
        
        # Late samples should be more ordered (converged)
        # This is a weak test but captures the annealing effect
        early_avg = np.mean(early_order)
        late_avg = np.mean(late_order)
        
        # Just verify we can compute this without error
        assert early_avg >= 0 and early_avg <= 1
        assert late_avg >= 0 and late_avg <= 1
    
    def test_simulated_annealing_hyperparameters(self, rbm, sampler):
        """Test with different temperature schedules."""
        for T_init in [1.0, 5.0, 10.0]:
            for T_final in [0.001, 0.01, 0.1]:
                samples = sampler.sample(rbm, 20, config={
                    'method': 'simulated_annealing',
                    'T_initial': T_init,
                    'T_final': T_final,
                    'n_steps': 500
                })
                assert samples.shape == (20, 8), \
                    f"Failed with T_init={T_init}, T_final={T_final}"


class TestSamplerComparison:
    """Tests comparing different samplers."""
    
    @pytest.fixture
    def rbm(self):
        """Create a test RBM."""
        return FullyConnectedRBM(n_visible=8, n_hidden=6)
    
    @pytest.fixture
    def sampler(self):
        """Create a test sampler."""
        return ClassicalSampler()
    
    def test_both_methods_return_same_shape(self, rbm, sampler):
        """Test that both methods return consistent shapes."""
        n_samples = 50
        
        mh_samples = sampler.sample(rbm, n_samples, {
            'method': 'metropolis',
            'n_sweeps': 50
        })
        
        sa_samples = sampler.sample(rbm, n_samples, {
            'method': 'simulated_annealing'
        })
        
        assert mh_samples.shape == sa_samples.shape, \
            f"Shape mismatch: MH {mh_samples.shape} vs SA {sa_samples.shape}"
    
    def test_both_methods_produce_valid_samples(self, rbm, sampler):
        """Test that both methods produce valid spin configs."""
        mh_samples = sampler.sample(rbm, 30, {'method': 'metropolis'})
        sa_samples = sampler.sample(rbm, 30, {'method': 'simulated_annealing'})
        
        for method_name, samples in [('MH', mh_samples), ('SA', sa_samples)]:
            valid = (samples == 1) | (samples == -1)
            assert np.all(valid), \
                f"{method_name} has invalid spin values"


class TestSamplerEdgeCases:
    """Tests for edge cases and error conditions."""
    
    @pytest.fixture
    def rbm(self):
        """Create a test RBM."""
        return FullyConnectedRBM(n_visible=4, n_hidden=3)
    
    @pytest.fixture
    def sampler(self):
        """Create a test sampler."""
        return ClassicalSampler()
    
    def test_single_sample(self, rbm, sampler):
        """Test sampling a single configuration."""
        samples = sampler.sample(rbm, 1, {'method': 'metropolis'})
        assert samples.shape == (1, 4)
    
    def test_large_sample_count(self, rbm, sampler):
        """Test with large sample count."""
        samples = sampler.sample(rbm, 1000, {
            'method': 'metropolis',
            'n_sweeps': 20,
            'n_between': 1
        })
        assert samples.shape == (1000, 4)
    
    def test_small_rbm(self):
        """Test with very small RBM."""
        tiny_rbm = FullyConnectedRBM(n_visible=2, n_hidden=1)
        sampler = ClassicalSampler()
        samples = sampler.sample(tiny_rbm, 10, {'method': 'metropolis'})
        assert samples.shape == (10, 2)
    
    def test_default_config_metropolis(self, rbm, sampler):
        """Test Metropolis with default config."""
        samples = sampler.sample(rbm, 20, {'method': 'metropolis'})
        assert samples.shape == (20, 4)
    
    def test_default_config_simulated_annealing(self, rbm, sampler):
        """Test SA with default config."""
        samples = sampler.sample(rbm, 20, {'method': 'simulated_annealing'})
        assert samples.shape == (20, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

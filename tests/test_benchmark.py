"""
Unit tests for benchmark infrastructure.

Tests for instance generation, configuration loading, and results handling.
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ConfigLoader, setup_logging
from instance_generator import InstanceGenerator


class TestConfigLoader:
    """Test configuration loading."""
    
    def test_load_valid_config(self, tmp_path):
        """Test loading a valid configuration file."""
        config_file = tmp_path / "config.json"
        config_data = {
            "benchmark": {"n_iterations": 50},
            "instances": {"n_instances_per_config": 3},
            "paths": {"data_dir": str(tmp_path / "data")},
            "logging": {"level": "INFO"}
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        loader = ConfigLoader(str(config_file))
        
        assert loader.config == config_data
        assert loader.get_benchmark_config()['n_iterations'] == 50
        assert loader.get_instance_config()['n_instances_per_config'] == 3
    
    def test_config_file_not_found(self):
        """Test error on missing config file."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader("nonexistent_config.json")
    
    def test_invalid_json(self, tmp_path):
        """Test error on invalid JSON."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{ invalid json }")
        
        with pytest.raises(json.JSONDecodeError):
            ConfigLoader(str(config_file))
    
    def test_validate_config(self, tmp_path):
        """Test configuration validation."""
        config_file = tmp_path / "config.json"
        config_data = {
            "benchmark": {},
            "instances": {},
            "paths": {},
            "logging": {}
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        loader = ConfigLoader(str(config_file))
        assert loader.validate() is True
    
    def test_validate_missing_keys(self, tmp_path):
        """Test validation fails with missing keys."""
        config_file = tmp_path / "config.json"
        config_data = {"benchmark": {}}
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        loader = ConfigLoader(str(config_file))
        assert loader.validate() is False


class TestInstanceGenerator:
    """Test instance generation."""
    
    def test_generate_instances(self, tmp_path):
        """Test generating instances."""
        gen = InstanceGenerator(data_dir=str(tmp_path))
        
        instances = gen.generate_instances(
            system_sizes=[4, 6],
            h_values=[0.5, 1.0],
            n_per_config=2
        )
        
        assert len(instances) == 8  # 2 sizes * 2 h * 2 configs
        assert all(inst['system_size'] in [4, 6] for inst in instances)
        assert all(inst['transverse_field'] in [0.5, 1.0] for inst in instances)
    
    def test_instance_ids_unique(self, tmp_path):
        """Test that instance IDs are unique."""
        gen = InstanceGenerator(data_dir=str(tmp_path))
        
        instances = gen.generate_instances(
            system_sizes=[4, 6],
            h_values=[0.5, 1.0],
            n_per_config=2
        )
        
        ids = [inst['instance_id'] for inst in instances]
        assert len(ids) == len(set(ids))
    
    def test_instance_deterministic_seed(self, tmp_path):
        """Test that seeds are deterministic."""
        gen = InstanceGenerator(data_dir=str(tmp_path / "gen1"))
        instances1 = gen.generate_instances(
            system_sizes=[4],
            h_values=[0.5],
            n_per_config=1
        )
        
        gen2 = InstanceGenerator(data_dir=str(tmp_path / "gen2"))
        instances2 = gen2.generate_instances(
            system_sizes=[4],
            h_values=[0.5],
            n_per_config=1
        )
        
        # Same seed should give same initial state
        assert instances1[0]['seed'] == instances2[0]['seed']
        assert instances1[0]['initial_state'] == instances2[0]['initial_state']
    
    def test_load_instance(self, tmp_path):
        """Test loading a single instance."""
        gen = InstanceGenerator(data_dir=str(tmp_path))
        
        instances = gen.generate_instances(
            system_sizes=[4],
            h_values=[0.5],
            n_per_config=1
        )
        
        loaded = gen.load_instance(instances[0]['instance_id'])
        assert loaded is not None
        assert loaded['system_size'] == 4
        assert loaded['transverse_field'] == 0.5
    
    def test_load_all_instances(self, tmp_path):
        """Test loading all instances."""
        gen = InstanceGenerator(data_dir=str(tmp_path))
        
        instances = gen.generate_instances(
            system_sizes=[4, 6],
            h_values=[0.5],
            n_per_config=1
        )
        
        loaded = gen.load_all_instances()
        assert len(loaded) == 2
    
    def test_instance_file_created(self, tmp_path):
        """Test that instance files are created."""
        gen = InstanceGenerator(data_dir=str(tmp_path))
        
        instances = gen.generate_instances(
            system_sizes=[4],
            h_values=[0.5],
            n_per_config=1
        )
        
        # Check JSON file exists
        files = list(tmp_path.glob("instance_*.json"))
        assert len(files) >= 1
    
    def test_instance_index_created(self, tmp_path):
        """Test that instance index is created."""
        gen = InstanceGenerator(data_dir=str(tmp_path))
        
        instances = gen.generate_instances(
            system_sizes=[4],
            h_values=[0.5],
            n_per_config=1
        )
        
        # Check index file exists
        index_file = tmp_path / "instance_index.json"
        assert index_file.exists()
        
        with open(index_file) as f:
            index = json.load(f)
        
        assert index['total_instances'] == 1
        assert len(index['instances']) == 1


class TestInstanceData:
    """Test instance data structure."""
    
    def test_instance_has_required_fields(self, tmp_path):
        """Test that instances have all required fields."""
        gen = InstanceGenerator(data_dir=str(tmp_path))
        
        instances = gen.generate_instances(
            system_sizes=[4],
            h_values=[0.5],
            n_per_config=1
        )
        
        required_fields = [
            'instance_id', 'system_size', 'transverse_field',
            'run_id', 'seed', 'initial_state'
        ]
        
        for field in required_fields:
            assert field in instances[0]
    
    def test_initial_state_is_binary(self, tmp_path):
        """Test that initial states are binary."""
        gen = InstanceGenerator(data_dir=str(tmp_path))
        
        instances = gen.generate_instances(
            system_sizes=[4, 6, 8],
            h_values=[0.5],
            n_per_config=1
        )
        
        for inst in instances:
            state = inst['initial_state']
            assert all(s in [0, 1] for s in state)
            assert len(state) == inst['system_size']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

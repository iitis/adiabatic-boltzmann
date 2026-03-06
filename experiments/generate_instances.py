"""
Generate numbered Ising instances for consistent benchmarking.

This script creates reference Ising problem instances that will be used
in all benchmark runs. Each instance is numbered and indexed, so we always
know which problems the RBM was trained on.

Generated instances are stored in data/ with an index.json for metadata.
"""

import json
import numpy as np
from pathlib import Path


class IsingInstanceGenerator:
    """Generate random Ising instances with consistent seeding."""
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.instances = {}
        self.index = {}
    
    def generate(self, n_spins: int, h: float, n_instances: int = 3) -> list:
        """
        Generate n_instances random Ising instances.
        
        Args:
            n_spins: Number of spins in the instance
            h: Transverse field strength parameter
            n_instances: How many instances to generate
        
        Returns:
            List of (instance_config_dict, instance_id)
        """
        instances = []
        instance_counter = self._count_existing_instances()
        
        for i in range(n_instances):
            # Unique seed per instance
            seed = self.base_seed + instance_counter + i
            np.random.seed(seed)
            
            # Generate instance config
            instance_id = instance_counter + i
            instance = {
                'id': instance_id,
                'n_spins': n_spins,
                'h': h,
                'seed': seed,
                'random_state': np.random.RandomState(seed).get_state(),
            }
            
            self.instances[instance_id] = instance
            instances.append((instance, instance_id))
        
        return instances
    
    def _count_existing_instances(self) -> int:
        """Count how many instances already exist in the index."""
        return len(self.instances)
    
    def save_instances(self, output_dir: str = "data/"):
        """Save all instances to individual JSON files and create index."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        index = {}
        
        for instance_id, instance_data in self.instances.items():
            # Format: instance_N_h_ID.json (e.g., instance_4_0.50_000.json)
            n = instance_data['n_spins']
            h = instance_data['h']
            
            filename = f"instance_{n}_{h:.2f}_{instance_id:03d}.json"
            filepath = output_path / filename
            
            # Save instance
            with open(filepath, 'w') as f:
                json.dump({
                    'id': instance_data['id'],
                    'n_spins': instance_data['n_spins'],
                    'h': instance_data['h'],
                    'seed': instance_data['seed'],
                }, f, indent=2)
            
            # Add to index
            index[instance_id] = {
                'filename': filename,
                'n_spins': n,
                'h': h,
                'seed': instance_data['seed'],
            }
        
        # Save index
        index_path = output_path / "instance_index.json"
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        print(f"Saved {len(self.instances)} instances to {output_path}")
        return index


def create_standard_instances(output_dir: str = "data/"):
    """
    Create standard benchmark instances.
    
    Parameters:
    - Model sizes: N=4, 6, 8, 10
    - Transverse fields: h=0.5, 1.0, 2.0
    - Instances per combination: 3
    
    Total: 4 * 3 * 3 = 36 instances
    """
    generator = IsingInstanceGenerator(base_seed=42)
    
    sizes = [4, 6, 8, 10]
    h_values = [0.50, 1.00, 2.00]
    n_instances_per_config = 3
    
    print("Generating standard benchmark instances...")
    for size in sizes:
        for h in h_values:
            print(f"  N={size:2d}, h={h:.2f}: ", end="", flush=True)
            instances = generator.generate(size, h, n_instances_per_config)
            print(f"Generated {len(instances)} instances (IDs: {[iid for _, iid in instances]})")
    
    # Save all
    index = generator.save_instances(output_dir)
    return index


if __name__ == "__main__":
    import sys
    
    output_dir = "data/" if len(sys.argv) < 2 else sys.argv[1]
    index = create_standard_instances(output_dir)
    
    print(f"\nInstance index saved with {len(index)} total instances")
    print("\nSample entries:")
    for i, (iid, meta) in enumerate(list(index.items())[:3]):
        print(f"  {iid}: {meta['filename']}")

#!/usr/bin/env python3
"""
slurm_generate_test_config.py
Generate a minimal test configuration for HPC environment testing
"""

import yaml
import os
from pathlib import Path

def create_test_config():
    """Create a minimal test configuration"""
    
    # Minimal test configuration
    config = {
        'paths': {
            'item_scores_file': 'test_item_scores.csv',
            'item_pair_scores_file': 'test_item_pair_scores.csv', 
            'position_scores_file': 'test_position_scores.csv',
            'position_pair_scores_file': 'test_position_pair_scores.csv'
        },
        'optimization': {
            'items_to_assign': ['a', 'b', 'c', 'd'],
            'positions_to_assign': ['1', '2', '3', '4'],
            'items_assigned': None,
            'positions_assigned': None
        },
        'output': {
            'results_prefix': 'test_layout_results',
            'save_directory': 'output/layouts'
        }
    }
    
    return config

def create_test_data():
    """Create minimal test data files"""
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Test item scores
    item_scores = """item,score
a,0.8
b,0.6
c,0.7
d,0.5
"""
    
    # Test item pair scores  
    item_pair_scores = """item_pair,score
ab,0.9
ac,0.7
ad,0.6
bc,0.8
bd,0.5
cd,0.6
"""
    
    # Test position scores
    position_scores = """position,score
1,0.9
2,0.8
3,0.7
4,0.6
"""
    
    # Test position pair scores
    position_pair_scores = """position_pair,score
12,0.8
13,0.6
14,0.4
23,0.7
24,0.5
34,0.6
"""
    
    # Write test data files
    test_files = {
        'test_item_scores.csv': item_scores,
        'test_item_pair_scores.csv': item_pair_scores,
        'test_position_scores.csv': position_scores,
        'test_position_pair_scores.csv': position_pair_scores
    }
    
    for filepath, content in test_files.items():
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✅ Created {filepath}")

def main():
    """Generate test configuration and data"""
    
    print("🧪 Generating Test Configuration and Data")
    print("=" * 45)
    
    # Create directories
    os.makedirs('output/configs1', exist_ok=True)
    os.makedirs('output/layouts', exist_ok=True)
    
    # Create test data
    print("\n1. Creating test data files:")
    create_test_data()
    
    # Create test config
    print("\n2. Creating test configuration:")
    config = create_test_config()
    
    # Save main test config
    config_path = 'test_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    print(f"✅ Created {config_path}")
    
    # Save as config_1.yaml for array testing
    array_config_path = 'output/configs1/config_1.yaml'
    with open(array_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    print(f"✅ Created {array_config_path}")
    
    # Create batch file for testing
    batch_file = 'test_single.txt'
    with open(batch_file, 'w') as f:
        f.write('1\n')
    print(f"✅ Created {batch_file}")
    
    print(f"\n🏁 Test Setup Complete!")
    print(f"\nNext steps:")
    print(f"1. Test single optimization:")
    print(f"   python3 optimize_layout.py --config test_config.yaml")
    print(f"")
    print(f"2. Test SLURM job:")
    print(f"   sbatch --export=CONFIG_FILE=test_single.txt --array=0-0 slurm_array_processor.sh")
    print(f"")
    print(f"3. Check results:")
    print(f"   ls -la output/layouts/")
    
    # Display config summary
    print(f"\n📋 Test Configuration Summary:")
    print(f"   Items to assign: {config['optimization']['items_to_assign']}")
    print(f"   Positions to assign: {config['optimization']['positions_to_assign']}")
    print(f"   Total permutations: {len(config['optimization']['items_to_assign'])}! = 24")
    print(f"   Data files: 4 CSV files with synthetic scores")

if __name__ == "__main__":
    main()
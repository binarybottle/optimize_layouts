#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------
Generate configuration files to run keyboard layout optimizations in parallel 
with specific letter-to-key constraints specified in each file.

This is Step 1 in a process to optimally arrange the 24 most frequent letters 
in the 24 keys of the home block of a keyboard. 

There are two versions of this script:
  - generate_configs1.py generates an initial set of sparse keyboard layouts as config files.
  - optimize_moo.py generates optimal keyboard layouts for a given config file.
  - generate_configs2.py generates a new set of config files based on the optimal keyboard layouts.

Usage: ``python generate_configs1.py``
    
See **README_keyboards.md** for a full description.

See **README.md** for instructions to run batches of config files in parallel.

"""
import os
import yaml
import itertools

# Configuration: output directory and number of layouts per configuration
OUTPUT_DIR = '../output/configs1'
CONFIG_FILE = '../config.yaml'

# Example with 14 letters to be arranged in 14 keys
"""
Number of permutations per config file, with some of the 14 items fixed per file: 
    14 items (0 fixed):  87,178,291,200 permutations
    13 items (1 fixed):   6,227,020,800 permutations
    12 items (2 fixed):     479,001,600 permutations
    11 items (3 fixed):      39,916,800 permutations
    10 items (4 fixed):       3,628,800 permutations
      - 4 constrained items in 4 constrained positions: 4! = 24 permutations
      - 6 remaining items in 6 remaining positions: 6! = 720 permutations
      - Total: 24 × 720 = 17,280 permutations

Configuration File Generation
Fixed Items (etao):
  - 'e' fixed to 2 positions: J or K
  - 't', 'a', 'o' fixed to 3 positions from remaining 9 tier1 positions
  - Combinations: C(9,3) = 84
  - Permutations: 3! = 6
  - Total config files: 2 × 84 × 6 = 1,008 configuration files

Per-Configuration Search Space (with 10 items in 12 available positions)
Constrained items ("insr"):
  - 4 items in 4 constrained positions: 4! = 24 arrangements
Free items ("hldcum"):
  - 6 items arranged in any 6 of the remaining 8 positions
  - Choose 6 positions from 8: C(8,6) = 28 combinations
  - Arrange 6 items in chosen positions: 6! = 720 permutations
  - Subtotal: 28 × 720 = 20,160 arrangements
  - Per configuration total: 24 × 20,160 = 483,840 permutations
Total Search Space
  - 1,008 config files × 483,840 permutations = 487,711,680 total permutations

"""

top_items = "etaoinsrhldcu"  # Most frequent letters (in English) for first configs (etaoinsrhldcumfpgwybvkxj)
ntop_items = len(top_items)

# Define position tiers
tier1_positions = ["F","J","D","K","E","I","S","L","V","M"]  # Top 10
tier2_positions = ["R","U","A",";","W","O"]  # Remaining 6
top_positions = tier1_positions + tier2_positions
positions_for_item_1 = ["J","K"]  # Constrain top item ('e') to these positions

# Fixed and constrained items/positions
nfixed_items = 5  # Number of fixed items
nconstrained_items = 0  # Number of constrained items
items_assigned  = top_items[:nfixed_items]           # First letters to assign
items_to_assign = top_items[nfixed_items:ntop_items] # Remaining of top letters to assign
items_to_constrain = top_items[nfixed_items:nfixed_items + nconstrained_items]
n_assigned = len(items_assigned)
n_assign = len(items_to_assign)

# Keys to assign by the optimization process
all_N_keys = "".join(top_positions)

# Base configuration from the original config file
with open(CONFIG_FILE, 'r') as f:
    base_config = yaml.safe_load(f)

def generate_constraint_sets():
    configs = []
    
    for pos1 in positions_for_item_1:
        # Only choose from tier1 positions for fixing t,a,o
        remaining_tier1 = [pos for pos in tier1_positions if pos != pos1]
        
        # Generate combinations only from tier1 for the 3 remaining fixed items
        for comboN in itertools.combinations(remaining_tier1, n_assigned - 1):
            for permN in itertools.permutations(comboN):
                pos2, pos3, pos4, pos5 = permN
                
                positions = {
                    items_assigned[0]: pos1,  # e
                    items_assigned[1]: pos2,  # t  
                    items_assigned[2]: pos3,  # a
                    items_assigned[3]: pos4,  # o
                    items_assigned[4]: pos5   # i
                }
                
                positions_assigned = ''.join([positions[letter] for letter in items_assigned])
                
                # Create positions_to_assign from ALL remaining positions (tier1 + tier2)
                used_positions = set(positions_assigned)
                positions_to_assign = ''.join([pos for pos in top_positions if pos not in used_positions])
                
                # Available positions for constraints
                positions_to_constrain = positions_to_assign #''.join(positions_to_assign[:nconstrained_items])
                
                configs.append({
                    'items_to_assign': items_to_assign,
                    'positions_to_assign': positions_to_assign,
                    'items_assigned': items_assigned,
                    'positions_assigned': positions_assigned,
                    'items_to_constrain': items_to_constrain,
                    'positions_to_constrain': positions_to_constrain,
                })
    
    return configs

def create_config_files(configs):
    """Create individual config file for each configuration."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Print the total number of configs we're working with
    print(f"Creating {len(configs)} configuration files...")
    
    for i, config_params in enumerate(configs, 1):
        # Create a copy of the base config
        config = yaml.safe_load(yaml.dump(base_config))  # Deep copy
        
        # Update optimization parameters
        for param, value in config_params.items():
            config['optimization'][param] = value
        
        # Write the configuration to a YAML file
        config_filename = f"{OUTPUT_DIR}/config_{i}.yaml"
        with open(config_filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Print progress feedback for every 100 files
        if i % 100 == 0:
            print(f"  Created {i}/{len(configs)} configuration files...")
        
if __name__ == "__main__":
    print("Generating keyboard layout configurations...")
    configs = generate_constraint_sets()
    print(f"Found {len(configs)} valid configurations based on the constraints.")
    create_config_files(configs)
    print(f"All configuration files have been generated in the '{OUTPUT_DIR}' directory.")
    
    # Print details about the first few configs
    num_examples = min(3, len(configs))
    print(f"\nShowing details for first {num_examples} configurations:")
    for i in range(num_examples):
        config = configs[i]
        print(f"\nConfig {i+1}:")
        print(f"  items_assigned: {config['items_assigned']}")
        print(f"  positions_assigned: {config['positions_assigned']}")
        print(f"  positions_to_assign: {config['positions_to_assign']}")
        print(f"  items_to_constrain: {config['items_to_constrain']}")
        print(f"  positions_to_constrain: {config['positions_to_constrain']}")
        
        # Map letters to positions for clarity
        letter_positions = {config['items_assigned'][j]: config['positions_assigned'][j] 
                          for j in range(len(config['items_assigned']))}
        letter_positions_to_constrain = {config['items_to_constrain'][j]: config['positions_to_constrain'][j] 
                          for j in range(len(config['items_to_constrain']))}
        print("  Letter mappings to fix:")
        for letter, pos in letter_positions.items():
            print(f"    {letter} -> {pos}")
        print("  Letter mappings to constrain:")
        for letter, pos in letter_positions_to_constrain.items():
            if letter and pos:
                print(f"    {letter} -> {pos}")

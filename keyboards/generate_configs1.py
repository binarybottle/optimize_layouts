#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------
Generate configuration files to run keyboard layout optimizations 
in parallel with specific letter-to-key constraints specified in each file.

This is Step 1 in a process to optimally arrange the 24 most frequent letters 
in the 24 keys of the home block of a keyboard. 

There are two versions of this script:
  - generate_configs1.py generates an initial set of sparse keyboard layouts as config files.
  - optimize_layout.py generates optimal keyboard layouts for a given config file.
  - generate_configs2.py generates a new set of config files based on the optimal keyboard layouts.

Usage: ``python generate_keyboard_configs1.py``
    
See **README_keyboards.md** for a full description.

See **README.md** for instructions to run batches of config files in parallel.

"""
import os
import yaml
import itertools

# Configuration: output directory and number of layouts per configuration
OUTPUT_DIR = '../output/configs1'
CONFIG_FILE = '../config.yaml'
nlayouts = 1000

# Items in configurations (16 letters to be assigned to keys)
# etaoinsrhldcumfpgwybvkxjqz
items_assigned  = "etaoi"       # 5 most frequent letters (etaoin in English)
items_to_assign = "nsrhldcumfp" # next 11 most frequent letters (srhldcumfp in English)
n_assign_first_round = 1
n_assigned = len(items_assigned)
n_assign = len(items_to_assign)

# Position constraints for items_assigned
# FDRSEVAWCQXZ JKULIM;O,P./ 
positions_for_item_1 = ["F","D"] # 2 most comfortable qwerty keys
constrain_n_keys = 16
if constrain_n_keys == 16:
    positions_for_items_2thruN = ["F","D","E","S","V","R","W","A",
                                  "J","K","I","L","M","U","O",";"] # 16 most comfortable qwerty keys
elif constrain_n_keys == 18:
    positions_for_items_2thruN = ["F","D","E","S","V","R","W","A","C",
                                  "J","K","I","L","M","U","O",";",","] # 18 most comfortable qwerty keys
# Keys to assign by the optimization process
all_N_keys = "".join(positions_for_items_2thruN)

# Base configuration from the original config file
with open(CONFIG_FILE, 'r') as f:
    base_config = yaml.safe_load(f)

def generate_constraint_sets():
    """Generate all valid configurations based on the constraints."""

    # Generate all valid configurations
    configs = []
    
    # Loop through all possible position combinations
    for pos1 in positions_for_item_1:
        remaining_positions = [pos for pos in positions_for_items_2thruN if pos not in [pos1]]

        # Generate all combinations of n_assigned positions
        for comboN in itertools.combinations(remaining_positions, n_assigned - n_assign_first_round):

            # Generate all permutations of these combinations
            for permN in itertools.permutations(comboN):
                pos2, pos3, pos4, pos5 = permN
                
                # Create final position assignments
                positions = {items_assigned[0]: pos1, 
                             items_assigned[1]: pos2, 
                             items_assigned[2]: pos3, 
                             items_assigned[3]: pos4, 
                             items_assigned[4]: pos5
                            }
                
                # Create the positions_assigned string (must match the order of items_assigned)
                positions_assigned = ''.join([positions[letter] for letter in items_assigned])
                
                # Create positions_to_assign (keys not used in positions_assigned)
                used_positions = set(positions_assigned)
                positions_to_assign = ''.join([pos for pos in all_N_keys if pos not in used_positions])
                
                # Add to configs if valid and positions_to_assign has the correct number of positions
                if len(positions_to_assign) == n_assign:
                    configs.append({
                        'items_to_assign': items_to_assign,
                        'positions_to_assign': positions_to_assign,
                        'items_assigned': items_assigned,
                        'positions_assigned': positions_assigned,
                        'items_to_constrain': "",  # No constraints for these configurations
                        'positions_to_constrain': ""
                    })

    return configs

def create_config_files(configs, nlayouts=10):
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
        
        # Set nlayouts
        config['optimization']['nlayouts'] = nlayouts
        
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
    create_config_files(configs, nlayouts)
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
        
        # Map letters to positions for clarity
        letter_positions = {config['items_assigned'][j]: config['positions_assigned'][j] 
                          for j in range(len(config['items_assigned']))}
        print("  Letter mappings:")
        for letter, pos in letter_positions.items():
            print(f"    {letter} -> {pos}")

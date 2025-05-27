# optimize_layouts/optimize_layout.py
"""
Memory-efficient item-to-position layout optimization using branch and bound search.

This script uses a branch and bound algorithm to find optimal positions 
for items and item pairs by jointly considering two scores: 
item/item_pair scores and position/position_pair scores.

```bash
#------------------------------
# Single-objective optimization
#------------------------------
# The default is single-objective optimization with a config.yaml file
python optimize_layout.py

# Provide scoring details:
python optimize_layout.py --config config.yaml --verbose

# Run with validation and performance tests:
python optimize_layout.py --config config.yaml --validate

#------------------------------
# Multi-objective optimization
#------------------------------
# Finds and saves all Pareto-optimal solutions (displays 3)
python optimize_layout.py --config config.yaml --moo

# Finds and saves the first 10 Pareto-optimal solutions, with a time limit imposed
python optimize_layout.py --config config.yaml --moo --max-solutions 10 --time-limit 60

# Finds and saves all Pareto-optimal solutions with validation tests [and scoring details]
python optimize_layout.py --config config.yaml --moo --validate [--verbose]
```

See README for more details.

"""
import sys
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit, config
import gc

import os
from math import perm
import time
from datetime import datetime, timedelta
import csv
from typing import List, Dict, Tuple
import argparse

from optimization_engine import (
    create_optimization_system,
    run_validation,
    calculate_score_for_new_items,
    calculate_upper_bound,
    performance_monitor,
    optimize_memory_usage,
    clear_caches,
    LayoutScorer,
    UpperBoundCalculator,
    ParetoFront,
    MultiObjectiveOptimizer
)

# Temporarily disable JIT for debugging
from numba import config
config.DISABLE_JIT = False

scoring_mode = 'combined' #'item_only' #'pair_only', 'combined'

# Save memory by using a temporary filesystem for the Numba cache:
# 1. First try to use the job scheduler's temporary directory
# 2. Then try to use /dev/shm (RAM-based filesystem)
# 3. Fall back to /tmp which is usually available
# Check if TMPDIR is set (common in SLURM and other schedulers)
tmpdir = os.environ.get('TMPDIR')
if tmpdir and os.path.exists(tmpdir) and os.access(tmpdir, os.W_OK):
    numba_cache = os.path.join(tmpdir, 'numba_cache')
    os.makedirs(numba_cache, exist_ok=True)
    config.CACHE_DIR = numba_cache
# Or try /dev/shm if available
elif os.path.exists('/dev/shm') and os.access('/dev/shm', os.W_OK):
    numba_cache = '/dev/shm/numba_cache'
    os.makedirs(numba_cache, exist_ok=True)
    config.CACHE_DIR = numba_cache
# Fall back to /tmp
elif os.path.exists('/tmp') and os.access('/tmp', os.W_OK):
    numba_cache = '/tmp/numba_cache'
    os.makedirs(numba_cache, exist_ok=True)
    config.CACHE_DIR = numba_cache
config.THREADING_LAYER = 'safe'  # Less threading overhead

#-----------------------------------------------------------------------------
# Loading, validating, and saving functions
#-----------------------------------------------------------------------------
def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from yaml file and normalize item case and numeric types."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Store the config path for use in file naming
    config['_config_path'] = config_path

    # Create necessary output directories
    output_dirs = [config['paths']['output']['layout_results_folder']]
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)

    # Normalize strings to lowercase with consistent access pattern
    optimization = config['optimization']
    for position in ['items_to_assign', 'positions_to_assign', 
                     'items_to_constrain', 'positions_to_constrain',
                     'items_assigned', 'positions_assigned']:
        # Use get() with empty string default for all positions
        optimization[position] = optimization.get(position, '').lower()
    
    # Validate constraints
    validate_config(config)

    items_to_assign = config['optimization']['items_to_assign']
    positions_to_assign = config['optimization']['positions_to_assign']
    items_to_constrain = config['optimization'].get('items_to_constrain', '')
    positions_to_constrain = config['optimization'].get('positions_to_constrain', '')
    items_assigned = config['optimization'].get('items_assigned', '')
    positions_assigned = config['optimization'].get('positions_assigned', '')

    # Convert to lowercase/uppercase for consistency
    items_to_assign = items_to_assign.lower()
    positions_to_assign = positions_to_assign.upper()
    items_to_constrain = items_to_constrain.lower()
    positions_to_constrain = positions_to_constrain.upper()
    items_assigned = items_assigned.lower()
    positions_assigned = positions_assigned.upper()

    print("\nConfiguration:")
    print(f"{len(items_to_assign)} items to assign: {items_to_assign}")
    print(f"{len(positions_to_assign)} available positions: {positions_to_assign}")
    print(f"{len(items_to_constrain)} items to constrain: {items_to_constrain}")
    print(f"{len(positions_to_constrain)} constraining positions: {positions_to_constrain}")
    print(f"{len(items_assigned)} items already assigned: {items_assigned}")
    print(f"{len(positions_assigned)} filled positions: {positions_assigned}")

    return config

def validate_config(config):
    """
    Validate optimization inputs from config, safely handling None values.
    """
    # Safely get and convert values
    items_to_assign = config['optimization'].get('items_to_assign', '')
    items_to_assign = set(items_to_assign.lower() if items_to_assign else '')

    positions_to_assign = config['optimization'].get('positions_to_assign', '')
    positions_to_assign = set(positions_to_assign.upper() if positions_to_assign else '')

    items_to_constrain = config['optimization'].get('items_to_constrain', '')
    items_to_constrain = set(items_to_constrain.lower() if items_to_constrain else '')

    positions_to_constrain = config['optimization'].get('positions_to_constrain', '')
    positions_to_constrain = set(positions_to_constrain.upper() if positions_to_constrain else '')

    items_assigned = config['optimization'].get('items_assigned', '')
    items_assigned = set(items_assigned.lower() if items_assigned else '')

    positions_assigned = config['optimization'].get('positions_assigned', '')
    positions_assigned = set(positions_assigned.upper() if positions_assigned else '')

    # Check for duplicates
    if len(items_to_assign) != len(config['optimization']['items_to_assign']):
        raise ValueError(f"Duplicate items in items_to_assign: {config['optimization']['items_to_assign']}")
    if len(positions_to_assign) != len(config['optimization']['positions_to_assign']):
        raise ValueError(f"Duplicate positions in positions_to_assign: {config['optimization']['positions_to_assign']}")
    if len(items_assigned) != len(config['optimization']['items_assigned']):
        raise ValueError(f"Duplicate items in items_assigned: {config['optimization']['items_assigned']}")
    if len(positions_assigned) != len(config['optimization']['positions_assigned']):
        raise ValueError(f"Duplicate positions in positions_assigned: {config['optimization']['positions_assigned']}")
    
    # Check that assigned items and positions have matching lengths
    if len(items_assigned) != len(positions_assigned):
        raise ValueError(
            f"Mismatched number of assigned items ({len(items_assigned)}) "
            f"and assigned positions ({len(positions_assigned)})"
        )

    # Check no overlap between assigned and to_assign
    overlap = items_assigned.intersection(items_to_assign)
    if overlap:
        raise ValueError(f"items_to_assign contains assigned items: {overlap}")
    overlap = positions_assigned.intersection(positions_to_assign)
    if overlap:
        raise ValueError(f"positions_to_assign contains assigned positions: {overlap}")

    # Check that we have enough positions
    if len(items_to_assign) > len(positions_to_assign):
        raise ValueError(
            f"More items to assign ({len(items_to_assign)}) "
            f"than available positions ({len(positions_to_assign)})"
        )

    # Check constraints are subsets
    if not items_to_constrain.issubset(items_to_assign):
        invalid = items_to_constrain - items_to_assign
        raise ValueError(f"items_to_constrain contains items not in items_to_assign: {invalid}")
    if not positions_to_constrain.issubset(positions_to_assign):
        invalid = positions_to_constrain - positions_to_assign
        raise ValueError(f"positions_to_constrain contains positions not in positions_to_assign: {invalid}")

    # Check if we have enough constraint positions for constraint items
    if len(items_to_constrain) > len(positions_to_constrain):
        raise ValueError(
            f"Not enough constraint positions ({len(positions_to_constrain)}) "
            f"for constraint items ({len(items_to_constrain)})"
        )

def load_scores(config: dict):
    """Load scores."""
    norm_item_scores = {}
    norm_item_pair_scores = {}
    norm_position_scores = {}
    norm_position_pair_scores = {}

    #-------------------------------------------------------------------------
    # Load position scores (always needed since used for both components)
    #-------------------------------------------------------------------------
    #print("\nLoading position scores...")
    position_df = pd.read_csv(config['paths']['input']['position_scores_file'], 
                              dtype={'item_pair': str})
    norm_scores = position_df['score'].values
   
    for idx, row in position_df.iterrows():
        norm_position_scores[row['position'].lower()] = np.float32(norm_scores[idx])

    #-------------------------------------------------------------------------
    # Load item scores
    #-------------------------------------------------------------------------
    #print("\nLoading item scores...")
    item_df = pd.read_csv(config['paths']['input']['item_scores_file'], 
                            dtype={'item_pair': str})
    norm_scores = item_df['score'].values
        
    for idx, row in item_df.iterrows():
        norm_item_scores[row['item'].lower()] = np.float32(norm_scores[idx])
    

    #-------------------------------------------------------------------------
    # Load pair scores
    #-------------------------------------------------------------------------
    # Load item pair scores
    #print("\nLoading item pair scores...")
    item_pair_df = pd.read_csv(config['paths']['input']['item_pair_scores_file'], 
                                dtype={'item_pair': str})
    
    # Get valid items from the item scores file
    item_df = pd.read_csv(config['paths']['input']['item_scores_file'], 
                            dtype={'item': str})
    valid_items = set(item_df['item'].str.lower())
    
    norm_scores = item_pair_df['score'].values
    
    invalid_item_pairs = []
    for idx, row in item_pair_df.iterrows():
        item_pair = row['item_pair']
        if not isinstance(item_pair, str):
            print(f"Warning: non-string item_pair at index {idx}: {item_pair} of type {type(item_pair)}")
            continue
        if len(item_pair) != 2:
            print(f"Warning: item_pair at index {idx} must be exactly 2 characters: '{item_pair}'")
            continue
        chars = tuple(item_pair.lower())
        # Verify both characters exist in item scores file
        if not all(c in valid_items for c in chars):
            invalid_item_pairs.append((idx, item_pair))
            continue
        norm_item_pair_scores[chars] = np.float32(norm_scores[idx])
    
    if invalid_item_pairs:
        print("\nWarning: Found item pairs with items not in item scores file:")
        for idx, pair in invalid_item_pairs:
            print(f"  Row {idx}: '{pair}' contains undefined items")

    # Load position pair scores
    #print("\nLoading position pair scores...")
    position_pair_df = pd.read_csv(config['paths']['input']['position_pair_scores_file'], 
                                    dtype={'position_pair': str})
    
    # Get valid positions from the position scores file
    position_df = pd.read_csv(config['paths']['input']['position_scores_file'], 
                                dtype={'position': str})
    valid_positions = set(position_df['position'].str.lower())
    
    norm_scores = position_pair_df['score'].values
    
    invalid_position_pairs = []
    for idx, row in position_pair_df.iterrows():
        position_pair = row['position_pair']
        if not isinstance(position_pair, str):
            print(f"Warning: non-string position_pair at index {idx}: {position_pair} of type {type(position_pair)}")
            continue
        if len(position_pair) != 2:
            print(f"Warning: position_pair at index {idx} must be exactly 2 characters: '{position_pair}'")
            continue
        chars = tuple(c.lower() for c in position_pair)
        # Verify both positions exist in position scores file
        if not all(c in valid_positions for c in chars):
            invalid_position_pairs.append((idx, row['position_pair']))
            continue
        norm_position_pair_scores[chars] = np.float32(norm_scores[idx])
        
    if invalid_position_pairs:
        print("\nWarning: Found position pairs with positions not in position scores file:")
        for idx, pair in invalid_position_pairs:
            print(f"  Row {idx}: '{pair}' contains undefined positions")

    return norm_item_scores, norm_item_pair_scores, norm_position_scores, norm_position_pair_scores

def prepare_arrays(
    items_to_assign, positions_to_assign,
    norm_item_scores, norm_item_pair_scores,
    norm_position_scores, norm_position_pair_scores,
    missing_item_pair_norm_score=1.0, missing_position_pair_norm_score=1.0,
    items_assigned=None, positions_assigned=None):
    """
    Prepare arrays for optimization, including all assigned items.
    """
    # Convert to lists if not already
    items_to_assign = list(items_to_assign)
    positions_to_assign = list(positions_to_assign)
    items_assigned = list(items_assigned or [])
    positions_assigned = list(positions_assigned or [])
    
    # Get dimensions
    n_items_to_assign = len(items_to_assign)
    n_positions_to_assign = len(positions_to_assign)
    n_items_assigned = len(items_assigned)
    
    # Create arrays just for the items being optimized
    # (We'll handle interactions with assigned items separately)
    
    # Create position score matrix
    position_score_matrix = np.zeros((n_positions_to_assign, n_positions_to_assign), dtype=np.float32)
    for i, k1 in enumerate(positions_to_assign):
        for j, k2 in enumerate(positions_to_assign):
            if i == j:
                position_score_matrix[i, j] = norm_position_scores.get(k1.lower(),
                                                                       missing_position_pair_norm_score)
            else:
                position_score_matrix[i, j] = norm_position_pair_scores.get((k1.lower(), k2.lower()),
                                                                            missing_position_pair_norm_score)
    
    # Create item score array
    item_scores = np.array([
        norm_item_scores.get(l.lower(), 0.0) for l in items_to_assign
    ], dtype=np.float32)
    
    # Create item_pair score matrix
    item_pair_score_matrix = np.zeros((n_items_to_assign, n_items_to_assign), dtype=np.float32)
    for i, l1 in enumerate(items_to_assign):
        for j, l2 in enumerate(items_to_assign):
            item_pair_score_matrix[i, j] = norm_item_pair_scores.get((l1.lower(), l2.lower()),
                                                                     missing_item_pair_norm_score)

    # Verify all scores are normalized [0,1]
    arrays_to_check = [
        (item_scores, "Item scores"),
        (item_pair_score_matrix, "Item pair scores"),
        (position_score_matrix, "Position scores")
    ]
    
    for arr, name in arrays_to_check:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
        if np.any(arr < 0) or np.any(arr > 1):
            raise ValueError(f"{name} must be normalized to [0,1] range")
    
    # Build cross-interaction matrices for assigned items
    if items_assigned and positions_assigned:
        # Create cross-item matrix (items_to_assign <-> items_assigned)
        cross_item_pair_matrix = np.zeros((n_items_to_assign, n_items_assigned), dtype=np.float32)
        for i, l1 in enumerate(items_to_assign):
            for j, l2 in enumerate(items_assigned):
                cross_item_pair_matrix[i, j] = norm_item_pair_scores.get((l1.lower(), l2.lower()),
                                                                        missing_item_pair_norm_score)

        # Create reverse cross-item matrix (items_assigned <-> items_to_assign)
        reverse_cross_item_pair_matrix = np.zeros((n_items_assigned, n_items_to_assign), dtype=np.float32)
        for j, l2 in enumerate(items_assigned):
            for i, l1 in enumerate(items_to_assign):
                reverse_cross_item_pair_matrix[j, i] = norm_item_pair_scores.get((l2.lower(), l1.lower()),
                                                                            missing_item_pair_norm_score)

        # Create cross-position matrix (positions_to_assign <-> positions_assigned)
        cross_position_pair_matrix = np.zeros((n_positions_to_assign, len(positions_assigned)), dtype=np.float32)
        for i, p1 in enumerate(positions_to_assign):
            for j, p2 in enumerate(positions_assigned):
                cross_position_pair_matrix[i, j] = norm_position_pair_scores.get((p1.lower(), p2.lower()),
                                                                               missing_position_pair_norm_score)
        # Create reverse cross-position matrix (positions_assigned <-> positions_to_assign)
        reverse_cross_position_pair_matrix = np.zeros((len(positions_assigned), n_positions_to_assign), dtype=np.float32)
        for j, p2 in enumerate(positions_assigned):
            for i, p1 in enumerate(positions_to_assign):
                reverse_cross_position_pair_matrix[j, i] = norm_position_pair_scores.get((p2.lower(), p1.lower()),
                                                                                    missing_position_pair_norm_score)

        # Return matrices including cross-interaction matrices in both directions
        return (item_scores, item_pair_score_matrix, position_score_matrix, 
                cross_item_pair_matrix, cross_position_pair_matrix,
                reverse_cross_item_pair_matrix, reverse_cross_position_pair_matrix)
  
    else:
        # No assigned items, just return the regular matrices
        return item_scores, item_pair_score_matrix, position_score_matrix
       
@jit(nopython=True, fastmath=True)
def validate_mapping(mapping: np.ndarray, constrained_item_indices: np.ndarray, constrained_positions: np.ndarray) -> bool:
    """JIT-compiled validation that mapping follows all constraints."""
    for i in range(len(constrained_item_indices)):
        idx = constrained_item_indices[i]
        if mapping[idx] >= 0:
            # Check if mapping[idx] is in constrained_positions
            found = False
            for j in range(len(constrained_positions)):
                if mapping[idx] == constrained_positions[j]:
                    found = True
                    break
            if not found:
                return False
    return True

def save_results_to_csv(results: List[Tuple[float, Dict[str, str], Dict[str, dict]]],
                       config: dict,
                       output_path: str = "layout_results.csv") -> None:
    """
    Save layout results to a CSV file with proper escaping of special characters.
    Include both pre-assigned and newly optimized items.
    """
    # Get config ID from the config file path
    config_path = os.path.basename(config.get('_config_path', 'config.yaml'))
    config_id = config_path.replace('config_', '').replace('.yaml', '')
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Include config_id in the filename
    output_path = os.path.join(config['paths']['output']['layout_results_folder'],
                               f"layout_results_{config_id}_{timestamp}.csv")
    
    def escape_special_chars(text: str) -> str:
        """Helper function to escape special characters and ensure proper CSV formatting."""
        # Replace certain special characters with their character names to prevent CSV splitting
        replacements = {
            ';': '[semicolon]',
            ',': '[comma]',
            '.': '[period]',
            '/': '[slash]'
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text
    
    # Get pre-assigned items and positions
    items_assigned = config['optimization'].get('items_assigned', '')
    positions_assigned = config['optimization'].get('positions_assigned', '')
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)  # Quote all fields
        
        # Write header with configuration info
        opt = config['optimization']
        writer.writerow(['Items to assign', escape_special_chars(opt.get('items_to_assign', ''))])
        writer.writerow(['Available positions', escape_special_chars(opt.get('positions_to_assign', ''))])
        writer.writerow(['Items to constrain', escape_special_chars(opt.get('items_to_constrain', ''))])
        writer.writerow(['Constraint positions', escape_special_chars(opt.get('positions_to_constrain', ''))])
        writer.writerow(['Assigned items', escape_special_chars(opt.get('items_assigned', ''))])
        writer.writerow(['Assigned positions', escape_special_chars(opt.get('positions_assigned', ''))])
        writer.writerow([])  # Empty row for separation
        
        # Write results header
        writer.writerow([
            'Items',
            'Positions',
            'Optimized Items',
            'Optimized Positions',
            'Rank',
            'Total score',
            'Item score',
            'Item-pair score'
        ])

        # Write results
        for rank, (score, mapping, detailed_scores) in enumerate(results, 1):
            # Build complete mapping including pre-assigned items
            complete_mapping = dict(zip(items_assigned, positions_assigned.upper()))  # Ensure positions are uppercase
            complete_mapping.update({k: v.upper() for k, v in mapping.items()})  # Ensure positions are uppercase
            
            # Get the items and positions strings
            optimized_items = escape_special_chars("".join(mapping.keys()))
            optimized_positions = escape_special_chars("".join(v.upper() for v in mapping.values()))  # Uppercase
            all_items = escape_special_chars("".join(complete_mapping.keys()))
            all_positions = escape_special_chars("".join(complete_mapping.values()))  # Already uppercase from above
            
            # Get scores
            first_entry = next(iter(detailed_scores.values()))
            item_pair_score = first_entry['item_pair_score']
            item_score = first_entry['item_score']
            
            writer.writerow([
                all_items,                 # All items (pre-assigned + optimized)
                all_positions,             # All positions (pre-assigned + optimized)
                optimized_items,           # Just the optimized items
                optimized_positions,       # Just the optimized positions
                rank,
                f"{score:.9f}",
                f"{item_score:.9f}",
                f"{item_pair_score:.9f}"
            ])
    
    print(f"\nResults saved to: {output_path}")

#-----------------------------------------------------------------------------
# Visualizing functions
#-----------------------------------------------------------------------------
def visualize_keyboard_layout(mapping: Dict[str, str] = None, title: str = "Layout", config: dict = None, items_to_display: str = None, positions_to_display: str = None) -> None:
    """
    Print a visual representation of a keyboard layout showing assigned items.
    """
    # Templates
    KEYBOARD_TEMPLATE = """╭───────────────────────────────────────────────╮
│ Layout: {title:<34}    │
├─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┤
│ {q:^3} │ {w:^3} │ {e:^3} │ {r:^3} ║ {u:^3} │ {i:^3} │ {o:^3} │ {p:^3} │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│ {a:^3} │ {s:^3} │ {d:^3} │ {f:^3} ║ {j:^3} │ {k:^3} │ {l:^3} │ {sc:^3} │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│ {z:^3} │ {x:^3} │ {c:^3} │ {v:^3} ║ {m:^3} │ {cm:^3} │ {dt:^3} │ {sl:^3} │
╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯"""
    
    QWERTY_KEYBOARD_TEMPLATE = """╭───────────────────────────────────────────────╮
│ Layout: {title:<34}    │
├─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┤
│  Q  │  W  │  E  │  R  ║  U  │  I  │  O  │  P  │
│ {q:^3} │ {w:^3} │ {e:^3} │ {r:^3} ║ {u:^3} │ {i:^3} │ {o:^3} │ {p:^3} │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│  A  │  S  │  D  │  F  ║  J  │  K  │  L  │  ;  │
│ {a:^3} │ {s:^3} │ {d:^3} │ {f:^3} ║ {j:^3} │ {k:^3} │ {l:^3} │ {sc:^3} │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│  Z  │  X  │  C  │  V  ║  M  │  ,  │  .  │  /  │
│ {z:^3} │ {x:^3} │ {c:^3} │ {v:^3} ║ {m:^3} │ {cm:^3} │ {dt:^3} │ {sl:^3} │
╰─────┴─────┴─────┴─────╨─────┴─────┴─────╨─────╯"""
    
    # Position mapping for special characters
    position_mapping = {
        ';': 'sc',  # semicolon
        ',': 'cm',  # comma
        '.': 'dt',  # dot/period
        '/': 'sl'   # forward slash
    }
    
    # Ensure we have a config
    if config is None:
        raise ValueError("Configuration must be provided")
    
    # Create layout characters dictionary with empty spaces
    layout_chars = {
        'title': title,
        'q': ' ', 'w': ' ', 'e': ' ', 'r': ' ',
        'u': ' ', 'i': ' ', 'o': ' ', 'p': ' ',
        'a': ' ', 's': ' ', 'd': ' ', 'f': ' ',
        'j': ' ', 'k': ' ', 'l': ' ', 'sc': ' ',
        'z': ' ', 'x': ' ', 'c': ' ', 'v': ' ',
        'm': ' ', 'cm': ' ', 'dt': ' ', 'sl': ' '
    }
    
    # If there's no mapping provided, show the layout with positions to be filled
    if not mapping:
        # Mark positions to be filled from positions_to_assign
        positions_to_mark = config['optimization'].get('positions_to_assign', '').lower()
        for position in positions_to_mark:
            converted_position = position_mapping.get(position, position)
            layout_chars[converted_position] = '-'
        
        # Fill in items from items_to_display and positions_to_display
        if items_to_display and positions_to_display:
            for item, position in zip(items_to_display, positions_to_display):
                converted_position = position_mapping.get(position.lower(), position.lower())
                layout_chars[converted_position] = item.lower()
    else:
        # We have a mapping, so fill in all keys from the mapping
        # (this will include both pre-assigned and newly optimized keys)
        for item, position in mapping.items():
            converted_position = position_mapping.get(position.lower(), position.lower())
            
            # Just display the uppercase letter (no doubling up of letters)
            layout_chars[converted_position] = item.upper()
    
    # Use the standard keyboard template
    template = KEYBOARD_TEMPLATE
    print(template.format(**layout_chars))

def calculate_total_perms(
    n_items: int,
    n_positions: int,
    items_to_constrain: set,
    positions_to_constrain: set,
    items_assigned: set,
    positions_assigned: set
) -> dict:
    """
    Calculate exact number of permutations for two-phase search.
    
    Phase 1: Arrange constrained items within constrained positions.
    Phase 2: For each Phase 1 solution, arrange remaining items in remaining positions.
    """
    # Phase 1: Arrange constrained items in constrained positions
    n_constrained_items = len(items_to_constrain)
    n_constrained_positions = len(positions_to_constrain)

    if n_constrained_items == 0 or n_constrained_positions == 0:
        # No constraints - everything happens in Phase 2
        total_perms_phase1 = 1
        remaining_items = n_items
        remaining_positions = n_positions
    else:
        # Calculate Phase 1 permutations
        total_perms_phase1 = perm(n_constrained_positions, n_constrained_items)
        # After Phase 1, we have used n_constrained_items positions
        remaining_items = n_items - n_constrained_items
        remaining_positions = n_positions - n_constrained_items
    
    # Phase 2: For each Phase 1 solution, arrange remaining items
    perms_per_phase1 = perm(remaining_positions, remaining_items)
    total_perms_phase2 = perms_per_phase1 * total_perms_phase1
    
    return {
        'total_perms': total_perms_phase2,  # Total perms is just Phase 2 total
        'phase1_arrangements': total_perms_phase1,
        'phase2_arrangements': total_perms_phase2,
        'details': {
            'remaining_positions': remaining_positions,
            'remaining_items': remaining_items,
            'constrained_items': n_constrained_items,
            'constrained_positions': n_constrained_positions,
            'arrangements_per_phase1': perms_per_phase1
        }
    }

def print_top_results(results: List[Tuple[float, Dict[str, str], Dict[str, dict]]],
                     config: dict,
                     n: int = None,
                     missing_item_pair_norm_score: float = 1.0,
                     missing_position_pair_norm_score: float = 1.0,
                     print_keyboard: bool = True,
                     verbose: bool = False) -> None:
    """
    Print the top N results with their scores and mappings.
    
    Args:
        results: List of (score, mapping, detailed_scores) tuples
        config: Configuration dictionary
        n: Number of layouts to display (defaults to config['optimization']['nlayouts'])
        print_keyboard: Whether to print the keyboard layout
        verbose: Whether to print detailed scoring information
    """
    if n is None:
        n = config['optimization'].get('nlayouts', 5)
    
    # Get pre-assigned items and positions
    items_assigned = config['optimization'].get('items_assigned', '')
    positions_assigned = config['optimization'].get('positions_assigned', '').upper()
    
    # Load scores if verbose mode enabled
    norm_item_scores = None
    norm_item_pair_scores = None
    norm_position_scores = None
    norm_position_pair_scores = None
    
    if verbose:
        norm_item_scores, norm_item_pair_scores, norm_position_scores, norm_position_pair_scores = (
            load_scores(config)
        )
    
    if len(results) > 1:
        print(f"\nTop {n} scoring layouts:")
    else:
        print(f"\nTop-scoring layout:")
    
    for i, (score, mapping, detailed_scores) in enumerate(results[:n], 1):
        print(f"\n#{i}: Score: {score:.9f}")
        
        # Build complete mapping including pre-assigned items
        complete_mapping = dict(zip(items_assigned, positions_assigned))
        
        # Update with optimized mapping, ensuring positions are uppercase
        capitalized_mapping = {k: v.upper() for k, v in mapping.items()}
        complete_mapping.update(capitalized_mapping)
        
        # Get the items and positions strings
        pre_items = items_assigned
        pre_positions = positions_assigned
        opt_items = ''.join(mapping.keys())
        opt_positions = ''.join(v.upper() for v in mapping.values())
        all_items = ''.join(complete_mapping.keys())
        all_positions = ''.join(complete_mapping.values())
        
        # Print in the requested format
        if pre_items:
            print(f"  items: {pre_items} + {opt_items} = {all_items}")
            print(f"  positions: {pre_positions} + {opt_positions} = {all_positions}")
        else:
            # No pre-assigned items
            print(f"  items: {all_items}")
            print(f"  positions: {all_positions}")
        
        # Display detailed score breakdown if verbose mode enabled
        if verbose:
            print("\nDetailed Scoring Information:")
            
            # Get component scores from detailed_scores
            component_scores = detailed_scores.get('total', {})
            item_score = component_scores.get('item_score', 0.0)
            #item_pair_score = component_scores.get('item_pair_score', 0.0)
            
            # Display individual item scores
            print("\nIndividual Item Scores:")
            print("  Item | Position | Item Score | Position Score | Combined")
            print("  " + "-" * 55)
            
            # Calculate individual item scores
            all_items_list = list(all_items)
            all_positions_list = list(all_positions)
            
            # Create a list of item details for sorting
            item_details = []
            for idx, (item, pos) in enumerate(zip(all_items_list, all_positions_list)):
                item_score = norm_item_scores.get(item.lower(), 0.0)
                pos_score = norm_position_scores.get(pos.lower(), 0.0)
                combined = item_score * pos_score
                item_details.append((item, pos, item_score, pos_score, combined))
            
            # Sort by item score (highest first)
            item_details.sort(key=lambda x: x[2], reverse=True)
            
            # Print each item's scores
            for item, pos, item_score, pos_score, combined in item_details:
                print(f"  {item}    | {pos}        | {item_score:10.6f} | {pos_score:12.6f} | {combined:.6f}")
            
            # Calculate and display pair scores
            print("\nTop Item Pair Contributions:")
            pair_scores = []
            
            for i in range(len(all_items_list)):
                for j in range(len(all_items_list)):
                    if i == j: continue
                    
                    item_i = all_items_list[i]
                    item_j = all_items_list[j]
                    pos_i = all_positions_list[i]
                    pos_j = all_positions_list[j]
                    
                    # Forward direction (item_i->item_j)
                    fwd_item_score = norm_item_pair_scores.get((item_i.lower(), item_j.lower()), missing_item_pair_norm_score)
                    fwd_pos_score = norm_position_pair_scores.get((pos_i.lower(), pos_j.lower()), missing_position_pair_norm_score)
                    
                    # Backward direction (item_j->item_i)
                    bck_item_score = norm_item_pair_scores.get((item_j.lower(), item_i.lower()), missing_item_pair_norm_score)
                    bck_pos_score = norm_position_pair_scores.get((pos_j.lower(), pos_i.lower()), missing_position_pair_norm_score)
                    
                    # Average of both directions
                    avg_score = (fwd_item_score * fwd_pos_score + bck_item_score * bck_pos_score) / 2
                    
                    pair_scores.append((
                        (item_i, item_j),
                        (pos_i, pos_j),
                        avg_score,
                        fwd_item_score,
                        fwd_pos_score,
                        bck_item_score,
                        bck_pos_score
                    ))
            
            # Sort by average score (highest first)
            pair_scores.sort(key=lambda x: x[2], reverse=True)
            
            # Print top pairs
            num_to_show = min(10, len(pair_scores))
            print(f"  Items | Positions | Avg Score | Fwd Item | Fwd Pos | Bck Item | Bck Pos")
            print("  " + "-" * 75)
            
            for i in range(num_to_show):
                (item_i, item_j), (pos_i, pos_j), avg, fwd_item, fwd_pos, bck_item, bck_pos = pair_scores[i]
                print(f"  {item_i}{item_j:3} | {pos_i}{pos_j:7} | {avg:9.6f} | {fwd_item:8.6f} | {fwd_pos:7.6f} | {bck_item:8.6f} | {bck_pos:7.6f}")
        
        # Display keyboard layout
        if print_keyboard:
            visualize_keyboard_layout(
                mapping=complete_mapping,
                title=f"Layout #{i}",
                config=config
            )

#-----------------------------------------------------------------------------
# Upper bound functions
#-----------------------------------------------------------------------------
def call_calculate_score_for_new_items(mapping, position_score_matrix, item_scores, 
                                item_pair_score_matrix, cross_item_pair_matrix=None,
                                cross_position_pair_matrix=None,
                                reverse_cross_item_pair_matrix=None,
                                reverse_cross_position_pair_matrix=None):
    """Scoring function with caching and consistency."""
    return calculate_score_for_new_items(
        mapping, position_score_matrix, item_scores, item_pair_score_matrix,
        cross_item_pair_matrix, cross_position_pair_matrix,
        reverse_cross_item_pair_matrix, reverse_cross_position_pair_matrix
    )
    
def call_calculate_upper_bound(mapping, used, position_score_matrix, item_scores,
                        item_pair_score_matrix, depth=None,
                        cross_item_pair_matrix=None, cross_position_pair_matrix=None,
                        reverse_cross_item_pair_matrix=None, reverse_cross_position_pair_matrix=None,
                        items_assigned=None, positions_assigned=None):
    """Upper bound calculation with tight estimates."""
    return calculate_upper_bound(
        mapping, used, position_score_matrix, item_scores, item_pair_score_matrix,
        cross_item_pair_matrix, cross_position_pair_matrix,
        reverse_cross_item_pair_matrix, reverse_cross_position_pair_matrix,
        items_assigned, positions_assigned, depth
    )
        
#-----------------------------------------------------------------------------
# Branch-and-bound functions
#-----------------------------------------------------------------------------
@jit(nopython=True)
def get_next_item(
    mapping: np.ndarray,
    constrained_items: np.ndarray = None  # Change to numpy array
) -> int:
    """
    Get next item to place, prioritizing constrained items if provided.
    """
    # Handle no constraints case
    if constrained_items is None or len(constrained_items) == 0:
        # Find first unplaced item
        for item in range(len(mapping)):
            if mapping[item] < 0:
                return item
        return -1
        
    # Handle constrained items
    for item in constrained_items:
        if mapping[item] < 0:
            return item
            
    # If all constrained items placed, find next unplaced item
    for item in range(len(mapping)):
        if mapping[item] < 0:
            return item
            
    return -1

def prepare_complete_arrays(
    items, positions,
    norm_item_scores, norm_item_pair_scores,
    norm_position_scores, norm_position_pair_scores,
    missing_item_pair_norm_score=1.0, missing_position_pair_norm_score=1.0):
    """
    Prepare arrays for scoring the complete layout.
    """
    n_items = len(items)  # Total number of items
    n_positions = len(positions)  # Total number of positions
    
    # Create position score matrix for ALL positions
    position_score_matrix = np.zeros((n_positions, n_positions), dtype=np.float32)
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions):
            if i == j:
                position_score_matrix[i, j] = norm_position_scores.get(pos1.lower(), 0.0)
            else:
                position_score_matrix[i, j] = norm_position_pair_scores.get((pos1.lower(), pos2.lower()), 
                                                                        missing_position_pair_norm_score)
    
    # Create item score array for ALL items
    item_scores = np.array([
        norm_item_scores.get(item.lower(), 0.0) for item in items
    ], dtype=np.float32)
    
    # Create item_pair score matrix for ALL items
    item_pair_score_matrix = np.zeros((n_items, n_items), dtype=np.float32)
    for i, item1 in enumerate(items):
        for j, item2 in enumerate(items):
            if i != j:  # Only needed for i != j pairs
                item_pair_score_matrix[i, j] = norm_item_pair_scores.get((item1.lower(), item2.lower()),
                                                                       missing_item_pair_norm_score)
    
    # Create direct mapping for evaluating a complete layout
    mapping = np.arange(n_items, dtype=np.int16)
    
    return mapping, item_scores, item_pair_score_matrix, position_score_matrix

def complete_and_score_full_layout(
    partial_mapping: np.ndarray,
    partial_used: np.ndarray,
    items_to_assign: str,
    positions_to_assign: str,
    items_assigned: str,
    positions_assigned: str,
    norm_item_scores: dict,
    norm_item_pair_scores: dict,
    norm_position_scores: dict,
    norm_position_pair_scores: dict
) -> Tuple[np.ndarray, float, float]:
    """
    Complete a partial mapping greedily and score the full 23-24 key layout.
    Returns (completed_mapping, total_score, cross_interaction_percentage).
    """
    # Complete phase 1 greedily
    mapping = partial_mapping.copy()
    used = partial_used.copy()
    
    for i in range(len(mapping)):
        if mapping[i] < 0:
            available = [p for p in range(len(used)) if not used[p]]
            if available:
                mapping[i] = available[0]
                used[mapping[i]] = True
    
    # Now create the full layout including phase 2 items
    # For phase 2, just assign remaining items to remaining positions in order
    phase2_items = [item for item in items_assigned]
    phase2_positions = [pos for i, pos in enumerate(positions_assigned)]
    
    # Build complete item and position lists
    all_items = list(items_to_assign) + phase2_items
    all_positions = list(positions_to_assign) + phase2_positions
    
    # Create complete mapping array
    complete_mapping = np.zeros(len(all_items), dtype=np.int16)
    
    # Fill in phase 1 mappings
    for i, pos_idx in enumerate(mapping):
        complete_mapping[i] = pos_idx
    
    # Fill in phase 2 mappings (they go to positions after phase 1)
    phase2_start_idx = len(items_to_assign)
    for i, item in enumerate(phase2_items):
        complete_mapping[phase2_start_idx + i] = len(positions_to_assign) + i
    
    # Prepare arrays for scoring
    arrays = prepare_complete_arrays(
        all_items, all_positions,
        norm_item_scores, norm_item_pair_scores,
        norm_position_scores, norm_position_pair_scores
    )
    
    final_mapping, item_scores_full, item_pair_score_matrix_full, position_score_matrix_full = arrays
    
    # Calculate scores including internal new item scores and potentially cross-interactions
    # if the matrices are passed
    total_score, item_score, pair_score = call_calculate_score_for_new_items(
        complete_mapping,
        position_score_matrix_full,
        item_scores_full,
        item_pair_score_matrix_full,
        # Here's the critical part - we're passing None for cross-interaction matrices
        None, None, None, None  # This means cross-interactions won't be included
    )
    
    # Now separately calculate the cross-interactions
    cross_score = 0.0
    cross_count = 0
    phase1_indices = set(range(len(items_to_assign)))
    phase2_indices = set(range(len(items_to_assign), len(all_items)))
    
    for i in phase1_indices:
        for j in phase2_indices:
            pos_i = complete_mapping[i]
            pos_j = complete_mapping[j]
            # Both directions
            cross_score += (item_pair_score_matrix_full[i, j] * position_score_matrix_full[pos_i, pos_j])
            cross_score += (item_pair_score_matrix_full[j, i] * position_score_matrix_full[pos_j, pos_i])
            cross_count += 2
    
    # Normalize cross_score by count
    normalized_cross_score = cross_score / cross_count if cross_count > 0 else 0.0
    
    # Calculate percentages properly
    # pair_score is already normalized internal pair interactions
    # We need to combine normalized values with the right weights
    if cross_count > 0 or pair_score > 0:
        # Get the raw counts
        n_items = len(items_to_assign)
        internal_pair_count = n_items * (n_items - 1)  # Both directions
        
        # Weighted average of normalized scores
        total_pair_score = ((pair_score * internal_pair_count) + 
                           (normalized_cross_score * cross_count)) / (internal_pair_count + cross_count)
                           
        # Percentage calculation
    if internal_pair_count + cross_count > 0:
        cross_percentage = (normalized_cross_score * cross_count) / ((total_pair_score * (internal_pair_count + cross_count))) * 100
    else:
        cross_percentage = 0.0
    
    # For total score including cross interactions
    if scoring_mode == 'item_only':
        final_total_score = item_score
    elif scoring_mode == 'pair_only':
        final_total_score = total_pair_score
    else:  # combined mode
        final_total_score = item_score * total_pair_score
    
    return complete_mapping, final_total_score, cross_percentage

def branch_and_bound_optimal_nsolutions(
    arrays: tuple,
    config: dict,
    n_solutions: int = 5,
    norm_item_scores: Dict = None,
    norm_position_scores: Dict = None,
    norm_item_pair_scores: Dict = None,
    norm_position_pair_scores: Dict = None,
    missing_item_pair_norm_score: float = 1.0,
    missing_position_pair_norm_score: float = 1.0,
    validate_bounds: bool = False,
    scorer: LayoutScorer = None, 
    bound_calculator: UpperBoundCalculator = None
) -> List[Tuple[float, Dict[str, str], Dict]]:
    """
    Branch and bound implementation using depth-first search. 
    
    Uses DFS instead of best-first search because:
    1. DFS requires only O(depth) memory vs O(width^depth) for best-first
    2. Simpler implementation without heap management complexity
    3. With a mathematically sound upper bound for pruning, 
       the search order doesn't affect optimality
    
    Search is conducted in two phases:
    Phase 1:
      - Finds all valid arrangements of constrained items ('e', 't') 
        in constrained positions (F, D, J, K).
      - Each arrangement marks positions as used/assigned.

    Phase 2: For each Phase 1 solution, arrange remaining items
      - For each valid arrangement from Phase 1
        - Uses ONLY the positions that weren't assigned during Phase 1.
          For example, if a Phase 1 solution put 'e' in F and 't' in J, 
          then Phase 2 would use remaining positions (not F or J)
          to arrange the remaining items.

    Args:
        arrays: Tuple of (item_scores, item_pair_score_matrix, position_score_matrix)
        config: Configuration dictionary with optimization parameters
        n_solutions: Number of top solutions to maintain

    Returns:
        Tuple containing:
        - List of (score, mapping, detailed_scores) tuples
        - Total number of permutations processed
    """
    debug_print = False
    
    # Get items and positions from config
    items_to_assign = config['optimization']['items_to_assign']
    positions_to_assign = config['optimization']['positions_to_assign']
    items_to_constrain = config['optimization'].get('items_to_constrain', '')
    positions_to_constrain = config['optimization'].get('positions_to_constrain', '')
    items_assigned = config['optimization'].get('items_assigned', '')
    positions_assigned = config['optimization'].get('positions_assigned', '')
    
    # Initialize dimensions and arrays
    n_items_to_assign = len(items_to_assign)
    n_positions_to_assign = len(positions_to_assign)
    permutations_completed = 0

    # Unpack arrays based on whether cross-interaction matrices are included
    if len(arrays) > 3:
        item_scores, item_pair_score_matrix, position_score_matrix, cross_item_pair_matrix, cross_position_pair_matrix, reverse_cross_item_pair_matrix, reverse_cross_position_pair_matrix = arrays
    else:
        item_scores, item_pair_score_matrix, position_score_matrix = arrays
        cross_item_pair_matrix = None
        cross_position_pair_matrix = None
        reverse_cross_item_pair_matrix = None
        reverse_cross_position_pair_matrix = None
        
    # Initialize search structures
    solutions = []  # Will store (score, scores, mapping) tuples
    worst_top_n_score = np.float32(-np.inf)
    seen_mappings = set()  # Track unique mappings

    # Initialize mapping and used positions
    initial_mapping = np.full(n_items_to_assign, -1, dtype=np.int16)
    initial_used = np.zeros(n_positions_to_assign, dtype=np.bool_)

    # Track statistics
    processed_nodes = 0
    pruned_count = 0
    #pruned_by_bound = 0
    explored_count = 0
    last_progress_update = 0
    progress_update_interval = 1000000  # Print progress every X permutations

    # Memory optimization
    optimize_memory_usage()

    # Handle pre-assigned items
    if items_assigned:
        item_to_idx = {item: idx for idx, item in enumerate(items_to_assign)}
        position_to_pos = {position: pos for pos, position in enumerate(positions_to_assign)}
        
        for item, position in zip(items_assigned, positions_assigned):
            if item in item_to_idx and position in position_to_pos:
                idx = item_to_idx[item]
                pos = position_to_pos[position]
                initial_mapping[idx] = pos
                initial_used[pos] = True
                if debug_print:
                    print(f"Pre-assigned: {item} -> {position} (position {pos})")
        
    # Calculate phase perms
    if items_to_constrain:
        # Set up constraint tracking
        constrained_items = set(items_to_constrain.lower())
        constrained_positions = set(i for i, position in enumerate(positions_to_assign) 
                                if position.upper() in positions_to_constrain.upper())
        constrained_item_indices = set(i for i, item in enumerate(items_to_assign) 
                                    if item in constrained_items)
        n_constrained = len(constrained_items)

        total_perms_phase1 = perm(len(constrained_positions), len(constrained_items))
    else:
        constrained_items = None
        constrained_positions = None
        constrained_item_indices = None
        n_constrained = 0
        total_perms_phase1 = 0

    n_phase2_remaining_items = n_items_to_assign - n_constrained
    n_phase2_available_positions = n_positions_to_assign - n_constrained
    n_perms_per_phase1_solution = perm(n_phase2_available_positions, n_phase2_remaining_items)
    total_perms_phase2 = n_perms_per_phase1_solution * total_perms_phase1

    if n_constrained:
        print(f"\nPhase 1 (constrained items): {total_perms_phase1:,} permutations")
        print(f"Phase 2 (remaining items): {total_perms_phase2:,} permutations")
    
    #-------------------------------------------------------------------------
    # Phase 1: Find all valid arrangements of constrained items
    #-------------------------------------------------------------------------
    def phase1_dfs(mapping: np.ndarray, used: np.ndarray, depth: int, pbar: tqdm) -> List[Tuple[np.ndarray, np.ndarray]]:
        """DFS for Phase 1 (constrained items)."""
        solutions = []
        
        # Found a valid phase 1 arrangement
        if all(mapping[i] >= 0 for i in constrained_item_indices):
            solutions.append((mapping.copy(), used.copy()))
            pbar.update(1)
            return solutions
            
        # Try next constrained item
        current_item_idx = -1
        for i in constrained_item_indices:
            if mapping[i] < 0:
                current_item_idx = i
                break
                
        # Try each constrained position
        for pos in constrained_positions:
            if not used[pos]:
                # Store original values for backtracking
                old_value = mapping[current_item_idx]
                
                # Make changes in-place
                mapping[current_item_idx] = pos
                used[pos] = True
                
                # Recursive call with the modified arrays
                sub_solutions = phase1_dfs(mapping, used, depth + 1, pbar)
                solutions.extend(sub_solutions)
                
                # Backtrack: restore original state
                mapping[current_item_idx] = old_value
                used[pos] = False

        return solutions

    if n_constrained:
        phase1_solutions = []
        with tqdm(total=total_perms_phase1, desc="Phase 1", unit='perms') as pbar:
            phase1_solutions = phase1_dfs(initial_mapping, initial_used, 0, pbar)      
        print(f"\nFound {len(phase1_solutions)} valid phase 1 arrangements")
    
    #-------------------------------------------------------------------------
    # Phase 2: For each Phase 1 solution, arrange remaining items
    #-------------------------------------------------------------------------
    def phase2_dfs(initial_mapping, initial_used, initial_depth, pbar, scorer=None):
        """Iterative version of DFS for Phase 2 using an explicit stack."""
        # Stack entries: (mapping, used, depth, path_str)
        stack = [(initial_mapping.copy(), initial_used.copy(), initial_depth, "")]
        
        # Use nonlocal variables from outer function
        nonlocal solutions, worst_top_n_score, processed_nodes, pruned_count, explored_count
        nonlocal last_progress_update, permutations_completed

        # Initialize the counter for negative bounds
        negative_bound_count = 0

        while stack:
            # Get current state from stack
            mapping, used, depth, path_str = stack.pop()
            
            # Count this node as processed
            processed_nodes += 1
            
            # Do garbage collection periodically
            if processed_nodes % 100000 == 0:
                gc.collect()
                gc.collect() # Force a second collection pass
            
            # Memory management
            if processed_nodes % 50000 == 0:
                optimize_memory_usage()
                if processed_nodes % 200000 == 0:
                    # Clear caches periodically to prevent memory bloat
                    if scorer is not None:
                        clear_caches(scorer)
                        #print(f"Cleared caches at {processed_nodes:,} nodes")
                    else:
                        gc.collect()
                        #print(f"Cleared memory at {processed_nodes:,} nodes")
            
            # Update progress periodically
            if processed_nodes - last_progress_update >= progress_update_interval:
                pbar.update(processed_nodes - last_progress_update)
                last_progress_update = processed_nodes
                if debug_print:
                    print(f"\nProgress: {processed_nodes:,} permutations processed")
                    print(f"Pruned: {pruned_count:,}, Explored: {explored_count:,}")
                    print(f"Pruning ratio: {pruned_count/(pruned_count+explored_count)*100:.2f}%")
                    if solutions:
                        print(f"Current best score: {solutions[-1][0]:.9f}")
            
            # Process complete solutions
            if depth == n_items_to_assign:
                permutations_completed += 1
                if n_constrained:
                    if not validate_mapping(mapping, constrained_item_indices, constrained_positions):
                        continue  # Skip to next stack item
                
                # Calculate score using the SAME arrays as the upper bound calculation
                total_score, item_component, item_pair_component = call_calculate_score_for_new_items(
                    mapping, 
                    position_score_matrix, 
                    item_scores, 
                    item_pair_score_matrix,
                    cross_item_pair_matrix, 
                    cross_position_pair_matrix,
                    reverse_cross_item_pair_matrix, 
                    reverse_cross_position_pair_matrix
                )
                
                # Check if solution qualifies
                worst_minus_epsilon = worst_top_n_score - np.abs(worst_top_n_score) * np.finfo(np.float32).eps
                margin = total_score - worst_minus_epsilon
                # Create a unique key for this mapping
                mapping_items = tuple(sorted((items_to_assign[i], positions_to_assign[mapping[i]]) 
                                            for i in range(len(mapping))))

                # Only process if this is a new unique solution
                if mapping_items not in seen_mappings:
                    seen_mappings.add(mapping_items)

                    # Check if solution qualifies for top-N
                    if len(solutions) < n_solutions or total_score > worst_top_n_score:
                        solution = (
                            total_score, item_component, item_pair_component, mapping.tolist()
                        )
                        solutions.append(solution)
                        solutions.sort(key=lambda x: x[0])  # Sort ascending (worst first)
                        
                        # Keep only top N solutions
                        if len(solutions) > n_solutions:
                            solutions.pop(0)  # Remove worst

                        # Update threshold
                        if solutions:
                            worst_top_n_score = solutions[0][0]

                # Skip to next stack item
                continue
            
            # Find next unassigned item
            current_item_idx = get_next_item(mapping)
            if current_item_idx == -1:
                continue  # Skip to next stack item
            
            # Update path for debugging
            if debug_print:
                new_path = path_str + items_to_assign[current_item_idx]
            
            # Get valid positions
            if n_constrained and current_item_idx in constrained_item_indices:
                valid_positions = [pos for pos in constrained_positions if not used[pos]]
            else:
                valid_positions = [pos for pos in range(n_positions_to_assign) if not used[pos]]
            
            if debug_print:
                print(f"\nPlacing item {items_to_assign[current_item_idx]} (idx {current_item_idx})")
                print("Current mapping: ", mapping)
                print(f"Valid positions: {valid_positions} ({[positions_to_assign[p] for p in valid_positions]})")
            
            # Try positions in reverse order (so first one is popped first)
            for pos in reversed(valid_positions):
                if debug_print:
                    print(f"  Trying {items_to_assign[current_item_idx]} in position {positions_to_assign[pos]}")
                
                # Temporarily modify the mapping and used arrays in-place (just for the pruning decision)
                mapping[current_item_idx] = pos
                used[pos] = True
                
                # Pruning decision using the SAME arrays
                should_prune = False
                if len(solutions) > 0:
                    upper_bound = call_calculate_upper_bound(
                        mapping, used,
                        position_score_matrix,  # Use the same arrays as score calculation
                        item_scores, 
                        item_pair_score_matrix,
                        cross_item_pair_matrix=cross_item_pair_matrix,
                        cross_position_pair_matrix=cross_position_pair_matrix,
                        reverse_cross_item_pair_matrix=reverse_cross_item_pair_matrix,
                        reverse_cross_position_pair_matrix=reverse_cross_position_pair_matrix,
                        items_assigned=items_assigned,
                        positions_assigned=positions_assigned
                    )

                    # Only prune if upper bound is definitely worse than our worst known solution
                    if len(solutions) > 0:
                        epsilon = 0.0001  # Small value that shouldn't affect optimality in practice
                        margin = upper_bound - worst_top_n_score
                        if margin < epsilon:
                            should_prune = True

                # Restore the original state before making a copy
                mapping[current_item_idx] = -1  # Restore to unassigned
                used[pos] = False
                
                if should_prune:
                    pruned_count += 1
                else:
                    explored_count += 1
                    # Only make copies when needed (after pruning decision)
                    new_mapping = mapping.copy()
                    new_mapping[current_item_idx] = pos
                    new_used = used.copy()
                    new_used[pos] = True
                    
                    # Add to stack
                    stack.append((
                        new_mapping,
                        new_used,
                        depth + 1,
                        new_path if debug_print else ""
                    ))

        if validate_bounds and negative_bound_count > 0:
            print(f"\n⚠️ Warning: {negative_bound_count} invalid bounds detected during optimization!")
            print("This may mean the optimization did not find the true optimal solution.")
            
    current_phase1_solution_index = 0

    # Estimate the total number of nodes in the search tree (an approximation)
    estimated_nodes = total_perms_phase2 * 1.5  # Add some buffer for internal nodes

    with tqdm(total=estimated_nodes, desc="Phase 2", unit=' nodes') as pbar:
        if n_constrained:
            for phase1_mapping, phase1_used in phase1_solutions:
                print(f"\nProcessing Phase 1 solution {current_phase1_solution_index + 1}/{len(phase1_solutions)}")
                
                # Calculate initial depth based on assigned items
                initial_depth = sum(1 for i in range(n_items_to_assign) if phase1_mapping[i] >= 0)
                
                # Start DFS from this phase 1 solution
                phase2_dfs(phase1_mapping, phase1_used, initial_depth, pbar, scorer)
                
                current_phase1_solution_index += 1
        else:
            # Calculate initial depth based on assigned items
            initial_depth = sum(1 for i in range(n_items_to_assign))
            
            # Start DFS from this phase 1 solution
            phase2_dfs(initial_mapping, initial_used, 0, pbar, scorer)

    # Convert final solutions to return format
    return_solutions = []
    for score, item_score, item_pair_score, mapping_list in reversed(solutions):
        mapping = np.array(mapping_list, dtype=np.int16)
        item_mapping = dict(zip(items_to_assign, [positions_to_assign[i] for i in mapping]))
        
        # Build complete mapping including pre-assigned items
        complete_mapping = dict(zip(items_assigned, positions_assigned))
        complete_mapping.update(item_mapping)
        all_items = list(complete_mapping.keys())
        all_positions = list(complete_mapping.values())
        
        # Use the optimization score directly - don't recalculate
        # (score, item_score, item_pair_score already have correct values from optimization)
        
        return_solutions.append((
            score,
            item_mapping,
            {'total': {
                'total_score': score,
                'item_pair_score': item_pair_score,
                'item_score': item_score
            }}
        ))

    # Ensure progress bar is fully updated at the end
    if processed_nodes > last_progress_update:
        pbar.update(processed_nodes - last_progress_update)
    
    # Print final statistics
    print(f"\nFinal statistics:")
    print(f"Permutations completed: {permutations_completed:,}")
    print(f"Nodes processed: {processed_nodes:,}")
    print(f"Branches pruned: {pruned_count:,}")
    print(f"Branches explored: {explored_count:,}")
    if pruned_count + explored_count > 0:
        print(f"Pruning ratio: {pruned_count/(pruned_count+explored_count)*100:.2f}%")

    diagnose_solutions = False
    if diagnose_solutions:
        print(f"\n=== SOLUTION COLLECTION DIAGNOSTICS ===")
        print(f"Requested solutions: {n_solutions}")
        print(f"Total solutions found: {len(solutions)}")
        print(f"Unique mappings seen: {len(seen_mappings)}")

        if len(solutions) > 1:
            best_score = max(sol[0] for sol in solutions)
            worst_score = min(sol[0] for sol in solutions)
            print(f"Best score: {best_score:.9f}")
            print(f"Worst score: {worst_score:.9f}")
            print(f"Score range: {best_score - worst_score:.9f}")
            
            # Check for score duplicates
            scores = [sol[0] for sol in solutions]
            unique_scores = len(set(round(s, 9) for s in scores))
            print(f"Unique scores: {unique_scores}")
            
            if unique_scores == 1:
                print("🚨 BUG CONFIRMED: All solutions have identical scores!")
            elif unique_scores < len(solutions) / 2:
                print("⚠️  WARNING: Many solutions have duplicate scores")
        else:
            print("Only one solution found total")

        # Show first few and last few solutions
        print("\nFirst 3 solutions:")
        for i, (score, _, _, mapping) in enumerate(solutions[-3:]):  # Last 3 = best 3 after sorting
            layout = {items_to_assign[j]: positions_to_assign[mapping[j]] for j in range(len(mapping))}
            layout_str = ''.join(f"{k}:{v}" for k, v in sorted(layout.items()))
            print(f"  #{i+1}: {score:.9f} - {layout_str}")

        if len(solutions) > 6:
            print("\nLast 3 solutions:")
            for i, (score, _, _, mapping) in enumerate(solutions[:3]):  # First 3 = worst 3 after sorting
                layout = {items_to_assign[j]: positions_to_assign[mapping[j]] for j in range(len(mapping))}
                layout_str = ''.join(f"{k}:{v}" for k, v in sorted(layout.items()))
                print(f"  #{len(solutions)-2+i}: {score:.9f} - {layout_str}")

        print(f"\nDEBUG - Scoring layout: {mapping}")

        # Check what scoring methods actually exist
        print(f"Available LayoutScorer methods: {[m for m in dir(scorer) if not m.startswith('_')]}")

        # Use the correct method name - should be score_layout with return_components
        try:
            # Try with return_components to get breakdown
            total_score, item_component, pair_component = scorer.score_layout(mapping, return_components=True)
            print(f"  Total score: {total_score}")
            print(f"  Item component: {item_component}")
            print(f"  Pair component: {pair_component}")
            
            # Check if cross-interactions exist
            if hasattr(scorer, 'has_cross_interactions') and scorer.has_cross_interactions:
                cross_score, cross_count = scorer._calculate_cross_interactions(mapping)
                print(f"  Cross component: {cross_score} (count: {cross_count})")
            
        except Exception as e:
            print(f"  Error with return_components: {e}")
            # Fall back to basic scoring
            total_score = scorer.score_layout(mapping)
            print(f"  Total score (basic): {total_score}")

        # Also check the input arrays to see if they're all zeros or constants
        print(f"  Item scores array sample: {scorer.item_scores_array[:5] if len(scorer.item_scores_array) > 5 else scorer.item_scores_array}")
        print(f"  Position matrix shape: {scorer.position_matrix.shape}")
        print(f"  Position matrix sample: {scorer.position_matrix[:3] if len(scorer.position_matrix) > 3 else scorer.position_matrix}")

        # Check item pair matrix
        if hasattr(scorer, 'item_pair_matrix'):
            print(f"  Item pair matrix shape: {scorer.item_pair_matrix.shape}")
            print(f"  Item pair matrix sum: {scorer.item_pair_matrix.sum()}")
            print(f"  Item pair matrix max: {scorer.item_pair_matrix.max()}")
            print(f"  Item pair matrix min: {scorer.item_pair_matrix.min()}")

        # Manual calculation check
        manual_score = 0.0
        item_count = 0
        for i, pos in enumerate(mapping):
            if pos >= 0:
                item_val = scorer.item_scores_array[i] if i < len(scorer.item_scores_array) else 0
                if scorer.position_matrix.ndim == 1:
                    pos_val = scorer.position_matrix[pos] if pos < len(scorer.position_matrix) else 0
                else:
                    pos_val = scorer.position_matrix[pos, pos] if pos < scorer.position_matrix.shape[0] else 0
                contrib = item_val * pos_val
                manual_score += contrib
                item_count += 1
                if i < 3:  # Only show first 3 for brevity
                    print(f"    Item {i} at pos {pos}: {item_val} * {pos_val} = {contrib}")

        if item_count > 0:
            manual_score = manual_score / item_count
            print(f"  Manual item score calculation: {manual_score}")

        print("---")

    return return_solutions, processed_nodes, permutations_completed

#-----------------------------------------------------------------------------
# Debugging functions
#-----------------------------------------------------------------------------
def debug_upper_bound(
    mapping: np.ndarray,
    used: np.ndarray,
    phase1_items: int,
    phase1_positions: int,
    phase2_items: int,
    phase2_positions: int,
    item_scores_full: np.ndarray,
    item_pair_score_matrix_full: np.ndarray,
    position_score_matrix_full: np.ndarray,
    items_to_assign: str,
    positions_to_assign: str,
    items_assigned: str,
    positions_assigned: str,
    norm_item_scores: dict,
    norm_item_pair_scores: dict,
    norm_position_scores: dict,
    norm_position_pair_scores: dict,
    verbose: bool = True
):
    """
    Debug the full layout upper bound calculation by comparing it with a completed solution.
    This helps identify issues where the upper bound is incorrectly lower than
    an achievable score, which would lead to incorrect pruning during optimization.
    """
    # Calculate the full layout upper bound
    full_upper_bound = call_calculate_upper_bound(
        mapping, used,
        position_score_matrix_full,
        item_scores_full,
        item_pair_score_matrix_full,
        cross_item_pair_matrix=None,
        cross_position_pair_matrix=None,
        reverse_cross_item_pair_matrix=None,
        reverse_cross_position_pair_matrix=None,
        items_assigned=items_assigned,
        positions_assigned=positions_assigned
    )
    
    # Complete the solution greedily and score it
    completed_mapping, actual_score, cross_percentage = complete_and_score_full_layout(
        mapping.copy(), used.copy(),
        items_to_assign, positions_to_assign,
        items_assigned, positions_assigned,
        norm_item_scores, norm_item_pair_scores,
        norm_position_scores, norm_position_pair_scores
    )
    
    # This should NEVER be negative!
    gap = full_upper_bound - actual_score
    if gap < np.finfo(float).eps:
        if verbose:
            print(f"ERROR: Full upper bound {full_upper_bound:.6f} < actual {actual_score:.6f}")
            print(f"Gap: {gap:.6f}")
            print(f"Mapping: {mapping}")
            
            # Additional debugging information
            print("\nDetailed breakdown:")
            
            print(f"Phase 1 items: {phase1_items}, positions: {phase1_positions}")
            print(f"Phase 2 items: {phase2_items}, positions: {phase2_positions}")
            
            # Calculate metrics for the completed solution
            print(f"Cross-interaction percentage in completed solution: {cross_percentage:.1f}%")
    else:
        if verbose:
            print(f"Valid full upper bound: {full_upper_bound:.6f} >= actual {actual_score:.6f}")
            print(f"Gap: {gap:.6f}")
    
    return gap, full_upper_bound, actual_score
      
def analyze_upper_bound_quality(
    config, 
    sample_size=1000, 
    max_depth_to_sample=5, 
    buffer_values=[0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
    missing_item_pair_norm_score=1.0, 
    missing_position_pair_norm_score=1.0
):
    """ 
    Analyze the quality of upper bounds and determine the ideal buffer value.
    
    Args:
        config: Configuration dictionary
        sample_size: Number of nodes to sample at each depth
        max_depth_to_sample: Maximum depth to sample nodes
        buffer_values: List of buffer values to test
    """
    print("\n=== UPPER BOUND QUALITY ANALYSIS ===")
    
    # Load optimization data as in the normal optimization flow
    items_to_assign = config['optimization']['items_to_assign']
    positions_to_assign = config['optimization']['positions_to_assign']
    n_items_to_assign = len(items_to_assign)
    items_assigned = config['optimization'].get('items_assigned', '')
    positions_assigned = config['optimization'].get('positions_assigned', '')
    
    # Load and normalize scores
    norm_item_scores, norm_item_pair_scores, norm_position_scores, norm_position_pair_scores = load_scores(config)
    
    # Prepare arrays
    arrays = prepare_arrays(
        items_to_assign, positions_to_assign,
        norm_item_scores, norm_item_pair_scores, 
        norm_position_scores, norm_position_pair_scores,
        missing_item_pair_norm_score, missing_position_pair_norm_score,
        items_assigned, positions_assigned)
    
    # Unpack arrays based on whether cross-interaction matrices are included
    if len(arrays) > 3:
        item_scores, item_pair_score_matrix, position_score_matrix, cross_item_pair_matrix, cross_position_pair_matrix, reverse_cross_item_pair_matrix, reverse_cross_position_pair_matrix = arrays
    else:
        item_scores, item_pair_score_matrix, position_score_matrix = arrays
        cross_item_pair_matrix = None
        cross_position_pair_matrix = None
        reverse_cross_item_pair_matrix = None
        reverse_cross_position_pair_matrix = None
    
    # Data collection structures
    bound_quality_data = {
        'depth': [],
        'actual_score': [],
        'estimated_score': [],
        'gap': [],
        'would_prune': {buffer: [] for buffer in buffer_values}
    }
    
    # Sample node collection
    nodes_to_sample = []
    
    # Function to DFS and collect sample nodes
    def collect_sample_nodes(mapping, used, depth, nodes_collected):
        """Modified sampling function to ensure better coverage across depths."""
        if depth >= max_depth_to_sample:
            return
            
        # Record this node if we need more at this depth
        depth_count = sum(1 for n in nodes_collected if n[2] == depth)
        if depth_count < sample_size:
            # Deep copy to avoid reference issues
            nodes_collected.append((mapping.copy(), used.copy(), depth))
        
        # If we already have enough samples at this depth, and we're not at the max depth,
        # prioritize going deeper rather than exploring more at this level
        if depth_count >= sample_size and depth < max_depth_to_sample - 1:
            # Continue deeper only
            current_item_idx = -1
            for i in range(len(mapping)):
                if mapping[i] < 0:
                    current_item_idx = i
                    break
                    
            if current_item_idx == -1:
                return
                
            # Try just a few random positions to go deeper quickly
            valid_positions = [pos for pos in range(len(used)) if not used[pos]]
            if valid_positions:
                # Randomly select a position instead of trying all
                np.random.shuffle(valid_positions)
                pos = valid_positions[0]
                new_mapping = mapping.copy()
                new_mapping[current_item_idx] = pos
                new_used = used.copy()
                new_used[pos] = True
                
                collect_sample_nodes(new_mapping, new_used, depth + 1, nodes_collected)
            return
        
        # Normal exploration when we need more samples at this depth
        # Find unassigned item
        current_item_idx = -1
        for i in range(len(mapping)):
            if mapping[i] < 0:
                current_item_idx = i
                break
                
        if current_item_idx == -1:
            return
            
        # Try each valid position with randomization
        valid_positions = [pos for pos in range(len(used)) if not used[pos]]
        np.random.shuffle(valid_positions)  # Randomize exploration order
        
        # Dynamically adjust branching factor based on depth
        # Use more branches at shallow depths, fewer at deeper depths
        branch_factor = max(1, 5 - depth)  # 5 at depth 0, 4 at depth 1, etc.
        
        for pos in valid_positions[:min(branch_factor, len(valid_positions))]:
            new_mapping = mapping.copy()
            new_mapping[current_item_idx] = pos
            new_used = used.copy()
            new_used[pos] = True
            
            collect_sample_nodes(new_mapping, new_used, depth + 1, nodes_collected)
                    
    # Collect sample nodes
    initial_mapping = np.full(n_items_to_assign, -1, dtype=np.int16)
    initial_used = np.zeros(len(positions_to_assign), dtype=np.bool_)

    print(f"Collecting {sample_size} sample nodes at each depth up to {max_depth_to_sample}...")

    # Track progress per depth
    collected_per_depth = [0] * max_depth_to_sample
    max_attempts = 50  # Limit attempts to prevent infinite loops

    for attempt in range(max_attempts):
        # Check if we have enough samples at all depths
        if all(count >= sample_size for count in collected_per_depth):
            break
            
        # Collect more samples
        prev_size = len(nodes_to_sample)
        collect_sample_nodes(initial_mapping, initial_used, 0, nodes_to_sample)
        
        # Update counts
        collected_per_depth = [sum(1 for n in nodes_to_sample if n[2] == d) 
                            for d in range(max_depth_to_sample)]
        
        # Print progress
        if attempt % 5 == 0:
            print(f"Attempt {attempt+1}/{max_attempts}: Collected {len(nodes_to_sample)} nodes")
            for d in range(max_depth_to_sample):
                print(f"  Depth {d}: {collected_per_depth[d]}/{sample_size}")
        
        # If we didn't add any new nodes, try a different random seed
        if len(nodes_to_sample) == prev_size:
            np.random.seed(attempt + 42)

    print(f"Collected {len(nodes_to_sample)} total nodes")
    for d in range(max_depth_to_sample):
        print(f"  Depth {d}: {collected_per_depth[d]}/{sample_size}")
    
    # Analyze bound quality for each node
    print("\nAnalyzing bound quality...")
    for mapping, used, depth in nodes_to_sample:
        # Calculate actual solution score
        # Find a complete solution from this partial solution
        completed_mapping = mapping.copy()
        completed_used = used.copy()
        
        # Complete the mapping greedily for testing
        unassigned = [i for i in range(n_items_to_assign) if completed_mapping[i] < 0]
        available = [i for i in range(len(completed_used)) if not completed_used[i]]
        
        # Simple greedy completion
        for item in unassigned:
            if available:
                pos = available.pop(0)
                completed_mapping[item] = pos
                completed_used[pos] = True
        
        # Calculate actual score of the completed solution
        actual_score, _, _ = call_calculate_score_for_new_items(completed_mapping, 
                                position_score_matrix, 
                                item_scores, 
                                item_pair_score_matrix,
                                None,  # cross_item_pair_matrix
                                None,  # cross_position_pair_matrix
                                None,  # reverse_cross_item_pair_matrix
                                None   # reverse_cross_position_pair_matrix
        )
            
        # Calculate upper bound estimate
        estimate = call_calculate_upper_bound(mapping, used, position_score_matrix, 
                        item_scores, item_pair_score_matrix,
                        cross_item_pair_matrix=cross_item_pair_matrix,
                        cross_position_pair_matrix=cross_position_pair_matrix,
                        reverse_cross_item_pair_matrix=reverse_cross_item_pair_matrix,
                        reverse_cross_position_pair_matrix=reverse_cross_position_pair_matrix,
                        items_assigned=items_assigned, positions_assigned=positions_assigned)

        # Calculate gap
        gap = estimate - actual_score
        
        # Record data
        bound_quality_data['depth'].append(depth)
        bound_quality_data['actual_score'].append(actual_score)
        bound_quality_data['estimated_score'].append(estimate)
        bound_quality_data['gap'].append(gap)
        
        # Check if this would be pruned with different buffer values
        for buffer in buffer_values:
            # Assume a threshold score of 90% of the actual_score as a test
            threshold = actual_score * 0.9
            would_prune = (estimate - threshold) < buffer
            bound_quality_data['would_prune'][buffer].append(would_prune)
    
    # Analyze results by depth
    print("\nBound Quality Analysis by Depth:")
    for depth in range(max_depth_to_sample):
        indices = [i for i, d in enumerate(bound_quality_data['depth']) if d == depth]
        if not indices:
            continue
            
        gaps = [bound_quality_data['gap'][i] for i in indices]
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        max_gap = max(gaps) if gaps else 0
        
        print(f"Depth {depth}:")
        print(f"  Average gap: {avg_gap:.6f}")
        print(f"  Maximum gap: {max_gap:.6f}")
        
        # Pruning analysis for different buffers
        for buffer in buffer_values:
            pruned = sum(bound_quality_data['would_prune'][buffer][i] for i in indices)
            prune_rate = pruned / len(indices) if indices else 0
            print(f"  Buffer {buffer}: Would prune {pruned}/{len(indices)} nodes ({prune_rate*100:.2f}%)")
    
    # Overall recommendation
    print("\nBuffer Recommendation:")
    
    # For each buffer value, calculate how many nodes would be incorrectly pruned
    for buffer in buffer_values:
        incorrect_prunes = sum(
            1 for i in range(len(bound_quality_data['gap'])) 
            if bound_quality_data['would_prune'][buffer][i] and 
               bound_quality_data['actual_score'][i] > bound_quality_data['actual_score'][0] * 0.95
        )
        
        prune_rate = sum(bound_quality_data['would_prune'][buffer]) / len(bound_quality_data['would_prune'][buffer])
        
        print(f"Buffer {buffer}:")
        print(f"  Overall prune rate: {prune_rate*100:.2f}%")
        print(f"  Risk of pruning good solutions: {incorrect_prunes} nodes ({incorrect_prunes/len(bound_quality_data['gap'])*100:.2f}%)")
    
    print("\nSuggested buffer value: Choose the smallest buffer that has <1% risk of pruning good solutions")
    print("=== END ANALYSIS ===\n")

def validate_bounds_statistically(
    config, 
    n_trials=1000, 
    print_progress=True
):
    """
    Statistically validate that upper bounds are never lower than achievable scores.
    Fixed to use the same arrays as the actual optimization.
    """
    # Load data
    items_to_assign = config['optimization']['items_to_assign']
    positions_to_assign = config['optimization']['positions_to_assign']
    items_assigned = config['optimization'].get('items_assigned', '')
    positions_assigned = config['optimization'].get('positions_assigned', '')
    
    # Load scores
    norm_item_scores, norm_item_pair_scores, norm_position_scores, norm_position_pair_scores = load_scores(config)
    
    # Prepare arrays CONSISTENTLY - use the same approach as optimization
    arrays = prepare_arrays(
        items_to_assign, positions_to_assign,
        norm_item_scores, norm_item_pair_scores, 
        norm_position_scores, norm_position_pair_scores,
        1.0, 1.0,  # missing scores
        items_assigned, positions_assigned
    )
    
    # Unpack arrays
    if len(arrays) > 3:
        item_scores, item_pair_score_matrix, position_score_matrix, cross_item_pair_matrix, cross_position_pair_matrix, reverse_cross_item_pair_matrix, reverse_cross_position_pair_matrix = arrays
    else:
        item_scores, item_pair_score_matrix, position_score_matrix = arrays
        cross_item_pair_matrix = None
        cross_position_pair_matrix = None
        reverse_cross_item_pair_matrix = None
        reverse_cross_position_pair_matrix = None
    
    # Create full arrays when there are pre-assigned items
    if items_assigned:
        all_items = items_to_assign + items_assigned
        all_positions = positions_to_assign + positions_assigned
        full_arrays = prepare_arrays(
            all_items, all_positions,
            norm_item_scores, norm_item_pair_scores, 
            norm_position_scores, norm_position_pair_scores,
            1.0, 1.0,  # missing scores
        )
        item_scores_full, item_pair_score_matrix_full, position_score_matrix_full = full_arrays[:3]
    else:
        # No pre-assigned items, use optimization arrays
        item_scores_full = item_scores
        item_pair_score_matrix_full = item_pair_score_matrix
        position_score_matrix_full = position_score_matrix
    
    # Setup for trials
    n_items = len(items_to_assign)
    n_positions = len(positions_to_assign)
    
    gaps = []
    depths = []
    invalid_bounds = 0
    
    # Run trials
    iterator = tqdm(range(n_trials)) if print_progress else range(n_trials)
    for _ in iterator:
        # Generate random partial mapping
        depth = np.random.randint(0, min(n_items, 6))  # Random depth up to 5
        depths.append(depth)
        
        # Create random partial mapping at this depth
        mapping = np.full(n_items, -1, dtype=np.int16)
        used = np.zeros(n_positions, dtype=np.bool_)
        
        # Randomly place 'depth' number of items
        positions = list(range(n_positions))
        np.random.shuffle(positions)
        for i in range(depth):
            if i < n_items:  # Make sure we don't go out of bounds
                mapping[i] = positions[i]
                used[positions[i]] = True
        
        # Calculate upper bound using the SAME arrays as optimization
        upper_bound = call_calculate_upper_bound(
            mapping, used,
            position_score_matrix_full,  # Use full arrays
            item_scores_full,
            item_pair_score_matrix_full,
            cross_item_pair_matrix=cross_item_pair_matrix,
            cross_position_pair_matrix=cross_position_pair_matrix,
            reverse_cross_item_pair_matrix=reverse_cross_item_pair_matrix,
            reverse_cross_position_pair_matrix=reverse_cross_position_pair_matrix,
            items_assigned=items_assigned,
            positions_assigned=positions_assigned
        )
        
        # Complete solution greedily
        completed_mapping = mapping.copy()
        completed_used = used.copy()
        
        # Simple greedy completion
        unassigned = [i for i in range(n_items) if completed_mapping[i] < 0]
        available = [i for i in range(n_positions) if not completed_used[i]]
        
        for item in unassigned:
            if available:
                pos = available.pop(0)
                completed_mapping[item] = pos
                completed_used[pos] = True
        
        # **CRITICAL FIX**: Calculate actual score using the same method as optimization
        if items_assigned:
            # For validation with pre-assigned items, we need to score the complete layout
            complete_mapping_for_scoring = np.zeros(len(all_items), dtype=np.int16)
            
            # Fill in the optimization part
            for i in range(n_items):
                complete_mapping_for_scoring[i] = completed_mapping[i]
            
            # Fill in the pre-assigned part
            for i, item in enumerate(items_assigned):
                complete_mapping_for_scoring[n_items + i] = n_positions + i
            
            actual_score, _, _ = call_calculate_score_for_new_items(
                complete_mapping_for_scoring, 
                position_score_matrix_full, 
                item_scores_full, 
                item_pair_score_matrix_full,
                None, None, None, None  # No cross-interactions for complete layout
            )
        else:
            # No pre-assigned items, score normally
            actual_score, _, _ = call_calculate_score_for_new_items(
                completed_mapping, 
                position_score_matrix, 
                item_scores, 
                item_pair_score_matrix,
                cross_item_pair_matrix, 
                cross_position_pair_matrix,
                reverse_cross_item_pair_matrix, 
                reverse_cross_position_pair_matrix
            )
        
        # Calculate gap and check validity
        gap = upper_bound - actual_score
        gaps.append(gap)
        
        if gap < np.finfo(float).eps:
            invalid_bounds += 1
            if print_progress and invalid_bounds <= 3:  # Only print first few
                print(f"\nInvalid bound detected: {upper_bound:.6f} < {actual_score:.6f}")
                print(f"Gap: {gap:.6f}, Depth: {depth}")
    
    # Rest of the analysis code remains the same...
    min_gap = min(gaps) if gaps else 0
    max_gap = max(gaps) if gaps else 0
    avg_gap = sum(gaps) / len(gaps) if gaps else 0
    
    # Results by depth
    depth_stats = {}
    for d in sorted(set(depths)):
        d_gaps = [gaps[i] for i in range(len(gaps)) if depths[i] == d]
        if d_gaps:
            depth_stats[d] = {
                'count': len(d_gaps),
                'min_gap': min(d_gaps),
                'avg_gap': sum(d_gaps) / len(d_gaps),
                'invalid': sum(1 for g in d_gaps if g < 0)
            }
    
    # Print results
    print("\n=== UPPER BOUND VALIDATION RESULTS ===")
    if invalid_bounds == 0:
        print(f"✅ VALID: All {n_trials} bounds were valid!")
    else:
        print(f"❌ INVALID: Found {invalid_bounds}/{n_trials} invalid bounds ({invalid_bounds/n_trials*100:.2f}%)")
    
    print(f"\nGap Statistics:")
    print(f"  Minimum gap: {min_gap:.6f}")
    print(f"  Maximum gap: {max_gap:.6f}")
    print(f"  Average gap: {avg_gap:.6f}")
    
    print("\nDepth-wise Analysis:")
    for d, stats in depth_stats.items():
        print(f"  Depth {d}: {stats['count']} trials, avg gap {stats['avg_gap']:.6f}, " + 
              (f"❌ {stats['invalid']} invalid" if stats['invalid'] > 0 else "✅ valid"))
    
    if invalid_bounds > 0:
        print("\n⚠️ Your upper bound calculation is incorrect and may prune optimal solutions!")
    else:
        print("\n✅ Your upper bound calculation appears to be valid.")
        print(f"   Average gap of {avg_gap:.6f} (smaller gaps mean tighter bounds and more efficient pruning)")
    
    return {
        'valid': invalid_bounds == 0,
        'invalid_count': invalid_bounds,
        'trials': n_trials,
        'min_gap': min_gap,
        'avg_gap': avg_gap,
        'max_gap': max_gap,
        'depth_stats': depth_stats
    }

#-----------------------------------------------------------------------------
# Main function and pipeline
#-----------------------------------------------------------------------------

def score_layout_components(self, mapping: np.ndarray) -> Tuple[float, float, float]:
    """
    Return the three independent objective components:
    - Item score component (normalized)
    - Internal pair score component (normalized) 
    - Cross-interaction score component (normalized)
    """
    n_items = len(mapping)
    
    # 1. Calculate item score component
    item_score = 0.0
    placed_count = 0
    for i in range(n_items):
        pos = mapping[i]
        if pos >= 0:
            if self.position_matrix.ndim == 1:
                pos_val = self.position_matrix[pos]
            else:
                pos_val = self.position_matrix[pos, pos] if pos < self.position_matrix.shape[1] else 1.0
            
            item_score += self.item_scores_array[i] * pos_val
            placed_count += 1
    
    # Normalize item score
    if placed_count > 0:
        item_score = item_score / placed_count
    
    # 2. Calculate internal pair score component
    pair_score = 0.0
    pair_count = 0
    for i in range(n_items):
        pos_i = mapping[i]
        if pos_i < 0:
            continue
            
        for j in range(i + 1, n_items):
            pos_j = mapping[j]
            if pos_j < 0:
                continue
                
            if self.position_matrix.ndim == 1:
                pos_i_val = self.position_matrix[pos_i]
                pos_j_val = self.position_matrix[pos_j]
            else:
                pos_i_val = self.position_matrix[pos_i, pos_i] if pos_i < self.position_matrix.shape[1] else 1.0
                pos_j_val = self.position_matrix[pos_j, pos_j] if pos_j < self.position_matrix.shape[1] else 1.0
            
            # Both directions for internal pairs
            pair_score += self.item_pair_matrix[i, j] * pos_i_val * pos_j_val
            pair_score += self.item_pair_matrix[j, i] * pos_j_val * pos_i_val
            pair_count += 2
    
    # Normalize pair score
    if pair_count > 0:
        pair_score = pair_score / pair_count
    
    # 3. Calculate cross-interaction score component
    cross_score = 0.0
    if self.has_cross_interactions:
        cross_score, cross_count = self._calculate_cross_interactions(mapping)
        # cross_score is already normalized in _calculate_cross_interactions
    
    return item_score, pair_score, cross_score

class MultiObjectiveUpperBoundCalculator:
    """Calculate upper bounds for each objective component independently."""
    
    def __init__(self, scorer: LayoutScorer):
        self.scorer = scorer
        self._cache = {}
        
    def calculate_component_upper_bounds(self, partial_mapping: np.ndarray, 
                                       used_positions: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate upper bounds for each objective component:
        Returns (item_upper_bound, pair_upper_bound, cross_upper_bound)
        """
        # Cache key for performance
        cache_key = (tuple(partial_mapping), tuple(used_positions))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Find unplaced items and available positions
        unplaced_items = [i for i in range(len(partial_mapping)) if partial_mapping[i] < 0]
        available_positions = [i for i in range(len(used_positions)) if not used_positions[i]]
        
        # Calculate current scores
        current_item, current_pair, current_cross = self.scorer.score_layout_components(partial_mapping)
        
        # Calculate maximum possible additions
        max_item_addition = self._calculate_max_item_addition(unplaced_items, available_positions)
        max_pair_addition = self._calculate_max_pair_addition(unplaced_items, available_positions, partial_mapping)
        max_cross_addition = self._calculate_max_cross_addition(unplaced_items, available_positions)
        
        # Calculate upper bounds (current + maximum possible addition)
        item_upper = current_item + max_item_addition
        pair_upper = current_pair + max_pair_addition  
        cross_upper = current_cross + max_cross_addition
        
        result = (item_upper, pair_upper, cross_upper)
        self._cache[cache_key] = result
        return result
    
    def _calculate_max_item_addition(self, unplaced_items, available_positions):
        """Calculate maximum possible addition to item score."""
        if not unplaced_items or not available_positions:
            return 0.0
            
        max_addition = 0.0
        placed_count = len([i for i in range(len(self.scorer.item_scores_array)) if i not in unplaced_items])
        
        # For each unplaced item, find its best possible position
        for item in unplaced_items:
            best_item_score = 0.0
            for pos in available_positions:
                if self.scorer.position_matrix.ndim == 1:
                    pos_val = self.scorer.position_matrix[pos]
                else:
                    pos_val = self.scorer.position_matrix[pos, pos] if pos < self.scorer.position_matrix.shape[1] else 1.0
                
                item_score = self.scorer.item_scores_array[item] * pos_val
                best_item_score = max(best_item_score, item_score)
            
            max_addition += best_item_score
        
        # Normalize by total number of items that will be placed
        total_items = placed_count + len(unplaced_items)
        if total_items > 0:
            max_addition = max_addition / total_items
            
        return max_addition
    
    def _calculate_max_pair_addition(self, unplaced_items, available_positions, partial_mapping):
        """Calculate maximum possible addition to pair score."""
        if len(unplaced_items) < 2 or len(available_positions) < 2:
            return 0.0
        
        max_addition = 0.0
        
        # Calculate pairs between unplaced items
        for i, item1 in enumerate(unplaced_items):
            for item2 in unplaced_items[i+1:]:
                if len(available_positions) >= 2:
                    best_pair_score = 0.0
                    
                    # Try all position pairs
                    for pos1 in available_positions:
                        for pos2 in available_positions:
                            if pos1 != pos2:
                                if self.scorer.position_matrix.ndim == 1:
                                    pos1_val = self.scorer.position_matrix[pos1]
                                    pos2_val = self.scorer.position_matrix[pos2]
                                else:
                                    pos1_val = self.scorer.position_matrix[pos1, pos1]
                                    pos2_val = self.scorer.position_matrix[pos2, pos2]
                                
                                # Both directions
                                pair_score = (self.scorer.item_pair_matrix[item1, item2] * pos1_val * pos2_val +
                                            self.scorer.item_pair_matrix[item2, item1] * pos2_val * pos1_val)
                                best_pair_score = max(best_pair_score, pair_score)
                    
                    max_addition += best_pair_score
        
        # Calculate pairs between unplaced and placed items
        placed_items = [i for i in range(len(partial_mapping)) if partial_mapping[i] >= 0]
        for unplaced_item in unplaced_items:
            for placed_item in placed_items:
                placed_pos = partial_mapping[placed_item]
                
                best_mixed_pair_score = 0.0
                for pos in available_positions:
                    if self.scorer.position_matrix.ndim == 1:
                        pos_val = self.scorer.position_matrix[pos]
                        placed_pos_val = self.scorer.position_matrix[placed_pos]
                    else:
                        pos_val = self.scorer.position_matrix[pos, pos]
                        placed_pos_val = self.scorer.position_matrix[placed_pos, placed_pos]
                    
                    # Both directions
                    pair_score = (self.scorer.item_pair_matrix[unplaced_item, placed_item] * pos_val * placed_pos_val +
                                self.scorer.item_pair_matrix[placed_item, unplaced_item] * placed_pos_val * pos_val)
                    best_mixed_pair_score = max(best_mixed_pair_score, pair_score)
                
                max_addition += best_mixed_pair_score
        
        # Normalize by total number of pairs
        total_items = len(partial_mapping)
        total_pairs = total_items * (total_items - 1) if total_items > 1 else 1
        max_addition = max_addition / total_pairs
        
        return max_addition
    
    def _calculate_max_cross_addition(self, unplaced_items, available_positions):
        """Calculate maximum possible addition to cross-interaction score."""
        if not self.scorer.has_cross_interactions or not unplaced_items:
            return 0.0
        
        max_addition = 0.0
        n_assigned = self.scorer.cross_item_pair_matrix.shape[1]
        
        for item in unplaced_items:
            best_item_cross = 0.0
            
            for pos in available_positions:
                item_cross_total = 0.0
                
                # Sum cross-interactions with all pre-assigned items
                for assigned_idx in range(n_assigned):
                    # Forward direction
                    fwd_item = self.scorer.cross_item_pair_matrix[item, assigned_idx]
                    fwd_pos = self.scorer.cross_position_pair_matrix[pos, assigned_idx]
                    
                    # Backward direction
                    bwd_item = self.scorer.reverse_cross_item_pair_matrix[assigned_idx, item]
                    bwd_pos = self.scorer.reverse_cross_position_pair_matrix[assigned_idx, pos]
                    
                    item_cross_total += fwd_item * fwd_pos + bwd_item * bwd_pos
                
                best_item_cross = max(best_item_cross, item_cross_total)
            
            max_addition += best_item_cross
        
        # Normalize by total cross-interactions
        total_items = len([i for i in range(len(self.scorer.item_scores_array))])
        total_cross_pairs = total_items * n_assigned * 2 if n_assigned > 0 else 1
        max_addition = max_addition / total_cross_pairs
        
        return max_addition

def pareto_dominates(obj1: List[float], obj2: List[float]) -> bool:
    """
    Check if obj1 Pareto dominates obj2.
    obj1 dominates obj2 if obj1 >= obj2 in all dimensions and obj1 > obj2 in at least one.
    """
    at_least_one_better = False
    for v1, v2 in zip(obj1, obj2):
        if v1 < v2:
            return False  # obj1 is worse in this dimension
        if v1 > v2:
            at_least_one_better = True
    return at_least_one_better

def can_improve_pareto_front(upper_bounds: List[float], pareto_front: List[List[float]]) -> bool:
    """
    Check if the upper bounds could potentially improve the Pareto front.
    Returns True if there's potential for improvement, False if should prune.
    """
    if not pareto_front:
        return True  # Empty front, always can improve
    
    # Check if upper bounds are dominated by any existing solution
    for existing_obj in pareto_front:
        if pareto_dominates(existing_obj, upper_bounds):
            return False  # Upper bounds are dominated, can't improve
    
    return True  # Upper bounds are not dominated, might improve front

class FixedMultiObjectiveOptimizer:
    """Fixed multi-objective optimizer with proper Pareto-based branch and bound."""
    
    def __init__(self, scorer: LayoutScorer, 
                 items_to_assign: str, available_positions: str,
                 items_to_constrain: str = '', positions_to_constrain: str = '',
                 items_assigned: str = '', positions_assigned: str = ''):
        self.scorer = scorer
        self.items_to_assign = list(items_to_assign)
        self.available_positions = list(available_positions)
        self.items_to_constrain = items_to_constrain
        self.positions_to_constrain = positions_to_constrain
        self.items_assigned = items_assigned
        self.positions_assigned = positions_assigned
        
        # Create multi-objective upper bound calculator
        self.upper_bound_calc = MultiObjectiveUpperBoundCalculator(scorer)
        
        # Pareto front storage: List of (mapping, objectives) tuples
        self.pareto_front = []
        self.pareto_objectives = []  # Just the objective values for quick dominance checking
        
        # Constraint setup
        self.constrained_items = set(items_to_constrain.lower()) if items_to_constrain else set()
        self.constrained_positions = set(i for i, pos in enumerate(available_positions) 
                                       if pos.upper() in positions_to_constrain.upper()) if positions_to_constrain else set()
        self.constrained_item_indices = set(i for i, item in enumerate(self.items_to_assign) 
                                          if item in self.constrained_items) if self.constrained_items else set()
        
        # Statistics
        self.nodes_processed = 0
        self.nodes_pruned = 0
        self.solutions_found = 0
        
    def optimize(self, max_solutions: int = 50, time_limit: float = 60.0):
        """Run multi-objective optimization with Pareto-based pruning."""
        print(f"Running Fixed Multi-Objective Optimization...")
        print(f"Objectives: Item Score, Pair Score, Cross-Interaction Score")
        
        start_time = time.time()
        
        # Initialize search
        n_items = len(self.items_to_assign)
        n_positions = len(self.available_positions)
        
        # Create initial partial mapping
        initial_mapping = np.full(n_items, -1, dtype=int)
        initial_used = np.zeros(n_positions, dtype=bool)
        
        # Apply pre-assigned constraints if any
        if self.items_assigned:
            item_to_idx = {item: idx for idx, item in enumerate(self.items_to_assign)}
            position_to_pos = {position: pos for pos, position in enumerate(self.available_positions)}
            
            for item, position in zip(self.items_assigned, self.positions_assigned):
                if item in item_to_idx and position in position_to_pos:
                    idx = item_to_idx[item]
                    pos = position_to_pos[position]
                    initial_mapping[idx] = pos
                    initial_used[pos] = True
        
        # Start recursive search
        self._search_recursive(initial_mapping, initial_used, max_solutions, time_limit, start_time)
        
        return self.pareto_front
    
    def _search_recursive(self, partial_mapping: np.ndarray, used_positions: np.ndarray,
                         max_solutions: int, time_limit: float, start_time: float):
        """Recursive search with Pareto-based pruning."""
        
        self.nodes_processed += 1
        
        # Check termination conditions
        if time_limit and (time.time() - start_time) > time_limit:
            return
        
        if len(self.pareto_front) >= max_solutions:
            return
        
        # Multi-objective pruning check
        upper_bounds = list(self.upper_bound_calc.calculate_component_upper_bounds(partial_mapping, used_positions))
        
        if not can_improve_pareto_front(upper_bounds, self.pareto_objectives):
            self.nodes_pruned += 1
            return  # Prune this branch
        
        # Find next unassigned item
        next_item = self._get_next_item(partial_mapping)
        
        # If all items assigned, evaluate complete solution
        if next_item == -1:
            objectives = list(self.scorer.score_layout_components(partial_mapping))
            
            # Check if this solution improves the Pareto front
            is_non_dominated = True
            dominated_indices = []
            
            for i, existing_obj in enumerate(self.pareto_objectives):
                if pareto_dominates(existing_obj, objectives):
                    is_non_dominated = False
                    break
                elif pareto_dominates(objectives, existing_obj):
                    dominated_indices.append(i)
            
            if is_non_dominated:
                # Remove dominated solutions
                for i in reversed(sorted(dominated_indices)):
                    del self.pareto_front[i]
                    del self.pareto_objectives[i]
                
                # Add new solution
                self.pareto_front.append((partial_mapping.copy(), objectives))
                self.pareto_objectives.append(objectives)
                self.solutions_found += 1
            
            return
        
        # Get valid positions for this item
        valid_positions = self._get_valid_positions(next_item, used_positions)
        
        # Try assigning next item to each valid position
        for pos_idx in valid_positions:
            # Make assignment
            partial_mapping[next_item] = pos_idx
            used_positions[pos_idx] = True
            
            # Recursive call
            self._search_recursive(partial_mapping, used_positions, max_solutions, time_limit, start_time)
            
            # Backtrack
            partial_mapping[next_item] = -1
            used_positions[pos_idx] = False
    
    def _get_next_item(self, mapping: np.ndarray) -> int:
        """Get next item to assign, prioritizing constrained items."""
        # First try constrained items
        if self.constrained_item_indices:
            for item_idx in self.constrained_item_indices:
                if mapping[item_idx] == -1:
                    return item_idx
        
        # Then any unassigned item
        for i in range(len(mapping)):
            if mapping[i] == -1:
                return i
        
        return -1
    
    def _get_valid_positions(self, item_idx: int, used_positions: np.ndarray) -> List[int]:
        """Get valid positions for an item."""
        if item_idx in self.constrained_item_indices:
            # Constrained item - only constrained positions
            return [pos for pos in self.constrained_positions if not used_positions[pos]]
        else:
            # Unconstrained item - any available position
            return [pos for pos in range(len(used_positions)) if not used_positions[pos]]

def run_multi_objective_optimization(config: dict, 
                                   norm_item_scores: dict, norm_item_pair_scores: dict,
                                   norm_position_scores: dict, norm_position_pair_scores: dict,
                                   max_solutions: int = None, time_limit: float = None):
    """Run fixed multi-objective optimization and return results."""
    
    # Get parameters from config
    items_to_assign = config['optimization']['items_to_assign']
    positions_to_assign = config['optimization']['positions_to_assign']
    items_to_constrain = config['optimization'].get('items_to_constrain', '')
    positions_to_constrain = config['optimization'].get('positions_to_constrain', '')
    items_assigned = config['optimization'].get('items_assigned', '')
    positions_assigned = config['optimization'].get('positions_assigned', '')
    
    # Prepare arrays
    arrays = prepare_arrays(
        items_to_assign, positions_to_assign,
        norm_item_scores, norm_item_pair_scores, 
        norm_position_scores, norm_position_pair_scores,
        1.0, 1.0,  # missing scores
        items_assigned, positions_assigned
    )
    
    # Create scorer using the NEW optimization system
    scorer, _ = create_optimization_system(arrays, config)
    
    # Create multi-objective optimizer with FIXED implementation
    optimizer = MultiObjectiveOptimizer(
        scorer=scorer,
        items_to_assign=items_to_assign,
        available_positions=positions_to_assign,
        items_to_constrain=items_to_constrain,
        positions_to_constrain=positions_to_constrain,
        items_assigned=items_assigned,
        positions_assigned=positions_assigned
    )
    
    # Run optimization
    pareto_front = optimizer.optimize(max_solutions=max_solutions, time_limit=time_limit)
    
    return pareto_front, optimizer

def run_multi_objective_mode(config: dict, verbose: bool = False,
                           max_solutions: int = None, time_limit: float = None) -> None:
    """Run multi-objective optimization with optional limits."""
    print(f"\n" + "="*50)
    print("MULTI-OBJECTIVE OPTIMIZATION MODE")
    print("=" * 50)
    
    # Load scores and run optimization
    norm_item_scores, norm_item_pair_scores, norm_position_scores, norm_position_pair_scores = load_scores(config)
    
    pareto_front, optimizer = run_multi_objective_optimization(
        config, norm_item_scores, norm_item_pair_scores,
        norm_position_scores, norm_position_pair_scores,
        max_solutions, time_limit  # Pass through the actual values (including None)
    )
    
    # Display results
    elapsed_time = time.time() - start_time
    print(f"\n" + "="*60)
    print(f"OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Final Pareto front: {len(pareto_front)} solutions")
    print(f"Total solutions found: {optimizer.solutions_found:,}")
    print(f"Nodes processed: {optimizer.nodes_processed:,}")
    print(f"Nodes pruned: {optimizer.nodes_pruned:,}")
    
    if optimizer.nodes_processed > 0:
        prune_rate = optimizer.nodes_pruned / optimizer.nodes_processed * 100
        print(f"Pruning efficiency: {prune_rate:.1f}%")
        
        # Calculate exploration rate
        n_items = len(config['optimization']['items_to_assign'])
        n_positions = len(config['optimization']['positions_to_assign'])
        import math
        total_space = math.factorial(n_positions) // math.factorial(n_positions - n_items)
        explored_pct = (optimizer.nodes_processed / total_space) * 100
        print(f"Search space explored: {explored_pct:.6f}%")
    
    print(f"Total runtime: {elapsed_time:.2f} seconds")
    
    if len(pareto_front) == 0:
        print("No solutions found!")
        return
        
    # Display solutions with proper objective names
    objective_names = ['Item Score', 'Pair Score', 'Cross-Interaction Score']
    
    print(f"\nPareto-Optimal Solutions:")
    print("-" * 100)
    
    items_to_assign = config['optimization']['items_to_assign']
    positions_to_assign = config['optimization']['positions_to_assign']
    items_assigned = config['optimization'].get('items_assigned', '')
    positions_assigned = config['optimization'].get('positions_assigned', '')
    
    # Sort solutions by combined score for display (but keep all as Pareto-optimal)
    pareto_with_combined = []
    for layout, objectives in pareto_front:
        combined_score = sum(objectives)  # Simple sum for display sorting
        pareto_with_combined.append((combined_score, layout, objectives))
    
    pareto_with_combined.sort(key=lambda x: x[0], reverse=True)
    
    for i, (combined_score, layout, objectives) in enumerate(pareto_with_combined[:5]):   # Show top 5
        print(f"\nSolution #{i+1} (Combined Score: {combined_score:.6f}):")
        
        # Create mapping dictionary
        item_mapping = {item: positions_to_assign[pos] for item, pos in 
                       zip(items_to_assign, layout) if pos >= 0}
        
        # Build complete mapping including pre-assigned items
        complete_mapping = dict(zip(items_assigned, positions_assigned))
        complete_mapping.update(item_mapping)
        
        # Display objectives with proper names
        print(f"  Independent Objectives:")
        for obj_name, obj_value in zip(objective_names, objectives):
            print(f"    {obj_name}: {obj_value:.6f}")
        
        # Display layout
        all_items = ''.join(complete_mapping.keys())
        all_positions = ''.join(complete_mapping.values())
        print(f"  Layout: {all_items} -> {all_positions}")
        
        # Display keyboard if requested
        if config['visualization'].get('print_keyboard', True):
            visualize_keyboard_layout(
                mapping=complete_mapping,
                title=f"Pareto Solution #{i+1}",
                config=config
            )
    
    # Save results to CSV with FIXED saving
    save_multi_objective_results_to_csv(pareto_front, objective_names, config)

def save_multi_objective_results_to_csv(pareto_front: List[Tuple[np.ndarray, List[float]]], 
                                       objective_names: List[str],
                                       config: dict) -> None:
    """Save multi-objective results to CSV with proper objective separation."""
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.basename(config.get('_config_path', 'config.yaml'))
    config_id = config_path.replace('config_', '').replace('.yaml', '')
    
    output_path = os.path.join(
        config['paths']['output']['layout_results_folder'],
        f"pareto_results_{config_id}_{timestamp}.csv"
    )
    
    items_to_assign = config['optimization']['items_to_assign']
    positions_to_assign = config['optimization']['positions_to_assign']
    items_assigned = config['optimization'].get('items_assigned', '')
    positions_assigned = config['optimization'].get('positions_assigned', '')
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        
        # Header with configuration
        writer.writerow(['Multi-Objective Pareto Results'])
        writer.writerow(['Items to assign', config['optimization'].get('items_to_assign', '')])
        writer.writerow(['Available positions', config['optimization'].get('positions_to_assign', '')])
        writer.writerow(['Assigned items', config['optimization'].get('items_assigned', '')])
        writer.writerow(['Assigned positions', config['optimization'].get('positions_assigned', '')])
        writer.writerow([])
        
        # Results header - include all three objectives plus combined total
        header = ['Rank', 'Items', 'Positions'] + objective_names + ['Combined Total']
        writer.writerow(header)
        
        # Sort solutions by combined score for ranking
        pareto_with_combined = []
        for layout, objectives in pareto_front:
            combined_score = sum(objectives)
            pareto_with_combined.append((combined_score, layout, objectives))
        
        pareto_with_combined.sort(key=lambda x: x[0], reverse=True)
        
        # Write solutions
        for rank, (combined_score, layout, objectives) in enumerate(pareto_with_combined, 1):
            # Create mapping
            item_mapping = {item: positions_to_assign[pos] for item, pos in 
                           zip(items_to_assign, layout) if pos >= 0}
            complete_mapping = dict(zip(items_assigned, positions_assigned))
            complete_mapping.update(item_mapping)
            
            all_items = ''.join(complete_mapping.keys())
            all_positions = ''.join(complete_mapping.values())
            
            row = ([rank, all_items, all_positions] + 
                   [f"{obj:.9f}" for obj in objectives] + 
                   [f"{combined_score:.9f}"])
            writer.writerow(row)
    
    print(f"\nPareto results saved to: {output_path}")

def validate_moo_objectives(config: dict, sample_size: int = 100):
    """Validate that MOO objectives are properly separated and don't double-count."""
    print("\n=== MOO OBJECTIVE VALIDATION ===")
    
    # Load data
    items_to_assign = config['optimization']['items_to_assign']
    positions_to_assign = config['optimization']['positions_to_assign']
    items_assigned = config['optimization'].get('items_assigned', '')
    positions_assigned = config['optimization'].get('positions_assigned', '')
    
    norm_item_scores, norm_item_pair_scores, norm_position_scores, norm_position_pair_scores = load_scores(config)
    
    # Prepare arrays
    arrays = prepare_arrays(
        items_to_assign, positions_to_assign,
        norm_item_scores, norm_item_pair_scores, 
        norm_position_scores, norm_position_pair_scores,
        1.0, 1.0,  # missing scores
        items_assigned, positions_assigned
    )
    
    # Create scorer
    scorer, _ = create_optimization_system(arrays, config)
    
    # Test with random mappings
    n_items = len(items_to_assign)
    n_positions = len(positions_to_assign)
    
    print(f"Testing {sample_size} random mappings...")
    
    objective_sums = []
    total_scores = []
    
    for _ in range(sample_size):
        # Generate random complete mapping
        mapping = np.random.permutation(n_positions)[:n_items]
        
        # Get independent objectives using NEW method
        item_obj, pair_obj, cross_obj = scorer.score_layout_components(mapping)
        
        # Get combined score using OLD method
        total_score = scorer.score_layout(mapping)
        
        # Store for analysis
        objective_sums.append(item_obj + pair_obj + cross_obj)
        total_scores.append(total_score)
        
        # Print first few for inspection
        if len(objective_sums) <= 3:
            print(f"  Sample {len(objective_sums)}:")
            print(f"    Item objective: {item_obj:.6f}")
            print(f"    Pair objective: {pair_obj:.6f}")
            print(f"    Cross objective: {cross_obj:.6f}")
            print(f"    Sum of objectives: {objective_sums[-1]:.6f}")
            print(f"    Total score (SOO): {total_scores[-1]:.6f}")
    
    # Analysis
    obj_sum_mean = np.mean(objective_sums)
    total_score_mean = np.mean(total_scores)
    
    print(f"\nValidation Results:")
    print(f"  Mean sum of objectives: {obj_sum_mean:.6f}")
    print(f"  Mean total score (SOO): {total_score_mean:.6f}")
    print(f"  Difference: {abs(obj_sum_mean - total_score_mean):.6f}")
    
    # Check if objectives are properly separated
    if abs(obj_sum_mean - total_score_mean) < 0.001:
        print("  ✅ Objectives appear to be properly separated (sum ≈ total)")
    else:
        print("  ❌ Objectives may have double-counting issues")
    
    print("=== MOO VALIDATION COMPLETE ===\n")

def optimize_layout(config: dict, verbose: bool = False, 
                    analyze_bounds: bool = False,     
                    missing_item_pair_norm_score: float = 1.0,
                    missing_position_pair_norm_score: float = 1.0,
                    validate_bounds: bool = False) -> None:
    """
    Main optimization function. Uses branch-and-bound search.
    """
    print(f"\n" + "="*50)
    print("SINGLE-OBJECTIVE OPTIMIZATION MODE")
    print("=" * 50)

    start_time = time.time()

    # Validate configuration
    validate_config(config)

    # Get parameters from config
    items_to_assign = config['optimization']['items_to_assign']
    positions_to_assign = config['optimization']['positions_to_assign']
    items_to_constrain = config['optimization'].get('items_to_constrain', '')
    positions_to_constrain = config['optimization'].get('positions_to_constrain', '')
    items_assigned = config['optimization'].get('items_assigned', '')
    positions_assigned = config['optimization'].get('positions_assigned', '')
    n_layouts = config['optimization'].get('nlayouts', 5)
    print_keyboard = config['visualization'].get('print_keyboard', True)
    if analyze_bounds:
        analyze_upper_bound_quality(config)

    # Calculate and print search space size
    search_space = calculate_total_perms(
        n_items=len(items_to_assign),
        n_positions=len(positions_to_assign),
        items_to_constrain=set(items_to_constrain),
        positions_to_constrain=set(positions_to_constrain),
        items_assigned=set(items_assigned),
        positions_assigned=set(positions_assigned)
    )
    print("\nSearch space:")
    if search_space['phase1_arrangements'] > 1:
        print(f"Phase 1 ({search_space['details']['constrained_items']} items constrained to {search_space['details']['constrained_positions']} positions): {search_space['phase1_arrangements']:,} permutations")
        print(f"Phase 2 ({search_space['details']['remaining_items']} items to arrange in {search_space['details']['remaining_positions']} positions): {search_space['details']['arrangements_per_phase1']:,} permutations per Phase 1 solution")
        print(f"Total permutations: {search_space['total_perms']:,}")
    else:
        print("No constraints - running single-phase optimization")
        print(f"Total permutations: {search_space['total_perms']:,}")
        print(f"- Arranging {search_space['details']['remaining_items']} items in {search_space['details']['remaining_positions']} positions")
    
    # Show initial keyboard
    if print_keyboard:
        print("\n")
        visualize_keyboard_layout(
            mapping=None,
            title="Positions to optimize",
            items_to_display=items_assigned,
            positions_to_display=positions_assigned,
            config=config
        )
    
    # Load normalized scores
    norm_item_scores, norm_item_pair_scores, norm_position_scores, norm_position_pair_scores = load_scores(config)
    
    # Prepare arrays for optimization
    arrays = prepare_arrays(
        items_to_assign, positions_to_assign,
        norm_item_scores, norm_item_pair_scores, 
        norm_position_scores, norm_position_pair_scores,
        missing_item_pair_norm_score, missing_position_pair_norm_score,
        items_assigned, positions_assigned
    )
    
    # CREATE OPTIMIZATION SYSTEM
    print("\nInitializing optimization system...")
    scorer, bound_calculator = create_optimization_system(arrays, config)
    
    if n_layouts > 1:
        print(f"\nFinding top {n_layouts} solutions")
    else:
        print("\nFinding top solution")

    # Run optimization WITH scorer and bound_calculator
    results, processed_nodes, permutations_completed = branch_and_bound_optimal_nsolutions(
        arrays=arrays,
        config=config,
        n_solutions=n_layouts,
        norm_item_scores=norm_item_scores,
        norm_position_scores=norm_position_scores, 
        norm_item_pair_scores=norm_item_pair_scores,  
        norm_position_pair_scores=norm_position_pair_scores,
        missing_item_pair_norm_score=missing_item_pair_norm_score,
        missing_position_pair_norm_score=missing_position_pair_norm_score,
        validate_bounds=validate_bounds,
        scorer=scorer,
        bound_calculator=bound_calculator
    )
        
    # Use optimization scores directly - skip recalculation
    updated_results = []
    for score, mapping, detailed_scores in results:
        updated_results.append((score, mapping, detailed_scores))

    # Sort by optimization score (highest first)
    updated_results.sort(key=lambda x: x[0], reverse=True)

    # Print the updated results
    print_top_results(
        results=updated_results,
        config=config,
        n=None,
        missing_item_pair_norm_score=1.0,
        missing_position_pair_norm_score=1.0,
        print_keyboard=print_keyboard,
        verbose=verbose
    )
    
    # Save the updated results
    save_results_to_csv(updated_results, config)
    
    # Final statistics reporting
    elapsed_time = time.time() - start_time
    percent_explored = (permutations_completed / search_space['total_perms']) * 100
    print(f"{permutations_completed} of {search_space['total_perms']} permutations ({percent_explored:.1f}% of solution space explored) in {elapsed_time} seconds")

    # PERFORMANCE MONITORING
    performance_monitor.print_stats()

if __name__ == "__main__":

    analyze_bounds = False
    missing_item_pair_norm_score = 1.0
    missing_position_pair_norm_score = 1.0
    validate_bounds = False

    try:
        parser = argparse.ArgumentParser(description='Optimize keyboard layout.')
        parser.add_argument('--config', type=str, default='config.yaml',
                          help='Path to configuration file (default: config.yaml)')
        parser.add_argument('--verbose', action='store_true',
                          help='Print detailed scoring information')
        parser.add_argument('--validate', action='store_true',
                          help='Run comprehensive validation before optimization')        
        parser.add_argument('--multi-objective', '--moo', action='store_true',
                        help='Use multi-objective optimization (Pareto front)')
        parser.add_argument('--max-solutions', type=int, default=None,        # ← Unlimited by default
                        help='Maximum number of Pareto solutions to find (default: unlimited)')
        parser.add_argument('--time-limit', type=float, default=None,         # ← Unlimited by default  
                        help='Time limit in seconds for MOO (default: unlimited)')
        
        args = parser.parse_args()
        
        start_time = time.time()
        
        # Load configuration
        config = load_config(args.config)
        
        # Enhanced validation (if requested)
        if args.validate:
            print("Running validation...")

            # Generate realistic normalized data (all in [0,1] range)
            print("Using realistic normalized random data for validation...")
            import random
            random.seed(42)  # Reproducible validation
            
            # Generate realistic score ranges within [0,1]
            item_scores = {c: random.uniform(0.1, 1.0) for c in config['optimization']['items_to_assign']}
            pair_scores = {(c1, c2): random.uniform(0.0, 0.8) 
                        for c1 in config['optimization']['items_to_assign'] 
                        for c2 in config['optimization']['items_to_assign']}
            pos_scores = {c: random.uniform(0.2, 1.0) for c in config['optimization']['positions_to_assign']}
            pos_pair_scores = {(c1, c2): random.uniform(0.0, 0.6) 
                            for c1 in config['optimization']['positions_to_assign'] 
                            for c2 in config['optimization']['positions_to_assign']}

            # Create arrays with realistic data
            (item_scores_array, item_pair_matrix, position_matrix,
            cross_item_pair_matrix, cross_position_pair_matrix,
            reverse_cross_item_pair_matrix, reverse_cross_position_pair_matrix) = prepare_arrays(
                config['optimization']['items_to_assign'],
                config['optimization']['positions_to_assign'], 
                item_scores,
                pair_scores,
                pos_scores,
                pos_pair_scores,
                1.0,
                1.0,
                config['optimization']['items_assigned'],
                config['optimization']['positions_assigned']
            )

            # Create test scorer with realistic data
            test_scorer = LayoutScorer(
                item_scores_array,
                item_pair_matrix,
                position_matrix,
                cross_item_pair_matrix,
                cross_position_pair_matrix,
                reverse_cross_item_pair_matrix, 
                reverse_cross_position_pair_matrix,
                'combined'
            )

            test_bound_calculator = UpperBoundCalculator(test_scorer)

            # Run validation with realistic data
            validation_passed = run_validation(config, test_scorer, test_bound_calculator)
            if not validation_passed:
                print("⚠️ Validation failed - may need bound calculator adjustment")
                print("Continuing with optimization...\n")
        else:
            print("Skipping validation (use --validate to enable)")

        # CHOOSE OPTIMIZATION MODE
        if args.multi_objective:
            # Run multi-objective optimization
            run_multi_objective_mode(
                config, 
                verbose=args.verbose,
                max_solutions=args.max_solutions,
                time_limit=args.time_limit
            )
        else:
            # Run single-objective optimization (existing code)
            optimize_layout(config, 
                            verbose=args.verbose,
                            analyze_bounds=analyze_bounds,
                            missing_item_pair_norm_score=missing_item_pair_norm_score,
                            missing_position_pair_norm_score=missing_position_pair_norm_score,
                            validate_bounds=validate_bounds
                            )
        
        elapsed = time.time() - start_time
        print(f"Total runtime: {timedelta(seconds=int(elapsed))}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

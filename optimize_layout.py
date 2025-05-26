# optimize_layouts/optimize_layout.py
"""
Memory-efficient item-to-position layout optimization using branch and bound search.

This script uses a branch and bound algorithm to find optimal positions 
for items and item pairs by jointly considering two scores: 
item/item_pair scores and position/position_pair scores.

See README for more details.

>> python optimize_layout.py
"""
import sys
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit, config
import gc
from scipy.optimize import linear_sum_assignment

import os
from math import perm
import time
from datetime import datetime, timedelta
import csv
from typing import List, Dict, Tuple
import argparse

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
   
def validate_mapping(mapping: np.ndarray, constrained_item_indices: set, constrained_positions: set) -> bool:
    """Validate that mapping follows all constraints."""
    for idx in constrained_item_indices:
        if mapping[idx] >= 0 and mapping[idx] not in constrained_positions:
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
# Branch-and-bound functions
#-----------------------------------------------------------------------------
@jit(nopython=True, fastmath=True)
def calculate_score_for_new_items(
    mapping: np.ndarray,
    position_score_matrix: np.ndarray,
    item_scores: np.ndarray,
    item_pair_score_matrix: np.ndarray,
    cross_item_pair_matrix=None,
    cross_position_pair_matrix=None,
    reverse_cross_item_pair_matrix=None,
    reverse_cross_position_pair_matrix=None
) -> Tuple[float, float, float]:
    """
    Calculate layout score with option to return component scores.
    
    This function calculates scores for interactions within newly assigned items
    and interactions between newly assigned and pre-assigned items
    (items that were already assigned before this optimization step).
    The code does not account for interactions between pre-assigned items;
    it is assumed that these interactions don't need to be calculated during the 
    optimization because they're fixed and don't change as new assignments are made.

    Args:
        mapping: Array of position indices for each item (-1 for unplaced)
        position_score_matrix: Matrix of position and position-pair scores
        item_scores: Array of item scores
        item_pair_score_matrix: Matrix of item-pair scores
        cross_item_pair_matrix: Matrix of cross-interactions between items_to_assign and items_assigned
        cross_position_pair_matrix: Matrix of cross-interactions between positions_to_assign and positions_assigned
        reverse_cross_item_pair_matrix: Matrix for reverse item interactions (pre-assigned → newly assigned)
        reverse_cross_position_pair_matrix: Matrix for reverse position interactions
    
    Returns:
        tuple of (total_score, item_component, pair_component)
    """
    new_item_score = np.float32(0.0)
    new_pair_score = np.float32(0.0)
    n_items = len(mapping)
    new_item_count = 0
    new_pair_count = 0
    
    # Calculate item scores
    for i in range(n_items):
        pos_raw = mapping[i]
        if pos_raw >= 0:
            # Convert using standard Python int (this works better in Numba)
            pos = int(pos_raw)
            # Access the matrix using flat indexing: row * n_cols + col
            matrix_size = position_score_matrix.shape[0]
            idx = pos * matrix_size + pos
            value = position_score_matrix.flat[idx]
            new_item_score += value * item_scores[i]
            new_item_count += 1
            
    # Normalize item scores
    if new_item_count > 0:
        new_item_score /= new_item_count    
    
    # Calculate pair scores
    for i in range(n_items):
        pos_i_raw = mapping[i]
        if pos_i_raw >= 0:
            pos_i = int(pos_i_raw)
            for j in range(i + 1, n_items):
                pos_j_raw = mapping[j]
                if pos_j_raw >= 0:
                    pos_j = int(pos_j_raw)
                    
                    # Compute flat indices for the matrices
                    matrix_size = position_score_matrix.shape[0]
                    idx_fwd = pos_i * matrix_size + pos_j
                    idx_bck = pos_j * matrix_size + pos_i
                    
                    # Get values using flat indexing
                    pos_fwd = position_score_matrix.flat[idx_fwd]
                    pos_bck = position_score_matrix.flat[idx_bck]
                    
                    # Get item pair values (these should be fine since i,j are Python ints)
                    item_fwd = item_pair_score_matrix[i, j]
                    item_bck = item_pair_score_matrix[j, i]
                    
                    fwd_score = pos_fwd * item_fwd
                    bck_score = pos_bck * item_bck
                    
                    new_pair_score += (fwd_score + bck_score)
                    new_pair_count += 2
                    
    # Normalize pair scores
    if new_pair_count > 0:
        new_pair_score /= new_pair_count
        
    # Add cross-interactions
    cross_pair_score, cross_pair_count = calculate_cross_interactions(
        mapping,
        cross_item_pair_matrix,
        cross_position_pair_matrix,
        reverse_cross_item_pair_matrix,
        reverse_cross_position_pair_matrix)
    
    if cross_pair_count > 0:
        cross_pair_score /= cross_pair_count
        
    # Calculate total score
    total_pair_count = new_pair_count + cross_pair_count
    if total_pair_count > 0:
        total_pair_score = ((new_pair_score * new_pair_count) +
                           (cross_pair_score * cross_pair_count)) / total_pair_count
    else:
        total_pair_score = 0.0
        
    # Final score based on mode
    if scoring_mode == 'item_only':
        total_score = new_item_score
    elif scoring_mode == 'pair_only':
        total_score = total_pair_score
    else:  # combined mode
        if new_item_count + total_pair_count > 0:
            total_score = new_item_score * total_pair_score
        else:
            total_score = 0.0
            
    return total_score, new_item_score, total_pair_score

@jit(nopython=True, fastmath=True)
def calculate_cross_interactions(
    mapping: np.ndarray,
    cross_item_pair_matrix: np.ndarray,
    cross_position_pair_matrix: np.ndarray,
    reverse_cross_item_pair_matrix: np.ndarray,
    reverse_cross_position_pair_matrix: np.ndarray
) -> Tuple[float, int]:
    """
    Calculate cross-interactions between assigned and unassigned items.
    """
    cross_interaction_score = 0.0
    interaction_count = 0
    
    if (cross_item_pair_matrix is None or cross_position_pair_matrix is None or
        reverse_cross_item_pair_matrix is None or reverse_cross_position_pair_matrix is None):
        return 0.0, 0
        
    n_items_to_assign = len(mapping)
    n_items_assigned = cross_item_pair_matrix.shape[1]
    
    for i in range(n_items_to_assign):
        pos_i_raw = mapping[i]
        if pos_i_raw < 0:
            continue
            
        pos_i = int(pos_i_raw)
        
        for j in range(n_items_assigned):
            # Compute flat indices for position matrices
            cross_pos_matrix_cols = cross_position_pair_matrix.shape[1]
            rev_pos_matrix_cols = reverse_cross_position_pair_matrix.shape[1]
            
            # Forward direction matrices
            fwd_item_score = cross_item_pair_matrix[i, j]
            fwd_pos_idx = pos_i * cross_pos_matrix_cols + j
            fwd_pos_score = cross_position_pair_matrix.flat[fwd_pos_idx]
            
            cross_interaction_score += fwd_item_score * fwd_pos_score
            
            # Backward direction matrices
            bwd_item_score = reverse_cross_item_pair_matrix[j, i]
            bwd_pos_idx = j * rev_pos_matrix_cols + pos_i
            bwd_pos_score = reverse_cross_position_pair_matrix.flat[bwd_pos_idx]
            
            cross_interaction_score += bwd_item_score * bwd_pos_score
            
            interaction_count += 2
            
    return cross_interaction_score, interaction_count
            
@jit(nopython=True, fastmath=True)
def prepare_bound_data(
    partial_mapping, partial_used,
    phase1_items, phase1_positions,
    phase2_items, phase2_positions,
    item_scores_full, item_pair_score_matrix_full,
    position_score_matrix_full
):
    """JIT-compiled function to prepare data for bound calculation."""
    # Just identify items and positions, don't try to create lists
    # Return information that will allow the caller to create appropriate arrays
    
    # Count placed items
    placed_count = 0
    for i in range(phase1_items):
        if partial_mapping[i] >= 0:
            placed_count += 1
    
    # Count unplaced phase 1 items
    unplaced_phase1_count = phase1_items - placed_count
    
    # Count available phase 1 positions
    available_phase1_count = 0
    for i in range(phase1_positions):
        if not partial_used[i]:
            available_phase1_count += 1
    
    return placed_count, unplaced_phase1_count, available_phase1_count

@jit(nopython=True, fastmath=True)
def calculate_upper_bound(
    mapping: np.ndarray,
    used: np.ndarray,
    position_score_matrix: np.ndarray,
    item_scores: np.ndarray,
    item_pair_score_matrix: np.ndarray,
    depth: int = None,
    cross_item_pair_matrix=None,
    cross_position_pair_matrix=None,
    reverse_cross_item_pair_matrix=None,
    reverse_cross_position_pair_matrix=None,
    items_assigned=None,
    positions_assigned=None  
) -> float:
    """
    Calculate upper bound on best possible score from this node.
    Fixed to properly normalize all components.
    """
    # Get current score from placed items (already normalized)
    current_score, current_item_score, current_pair_score = calculate_score_for_new_items(
        mapping, 
        position_score_matrix, 
        item_scores, 
        item_pair_score_matrix,
        cross_item_pair_matrix, 
        cross_position_pair_matrix,
        reverse_cross_item_pair_matrix, 
        reverse_cross_position_pair_matrix)
    
    # Find unplaced items and available positions
    unplaced = np.where(mapping < 0)[0]
    available = np.where(~used)[0]
    
    # If all items are placed, return the actual score
    if len(unplaced) == 0:
        return current_score
    
    # Calculate exact counts for normalization
    n_items = len(mapping)
    n_placed = n_items - len(unplaced)
    n_unplaced = len(unplaced)
    
    # Count of already-assigned items (from previous phase)
    n_assigned = 0
    if cross_item_pair_matrix is not None:
        n_assigned = cross_item_pair_matrix.shape[1]
    
    #-------------------------------------------------------------------------
    # Item component upper bound
    #-------------------------------------------------------------------------
    # Current item score contribution (unnormalized)
    current_item_total = current_item_score * n_placed
    
    # Maximum possible item score for unplaced items
    max_unplaced_item_total = 0.0
    
    # Get scores for available positions and unplaced items
    position_values = np.zeros(len(available), dtype=np.float32)
    for i, pos in enumerate(available):
        position_values[i] = position_score_matrix[pos, pos]
    position_values = np.sort(position_values)[::-1]
    
    item_values = item_scores[unplaced]
    item_values = np.sort(item_values)[::-1]
    
    # Greedy matching for upper bound
    n_to_match = min(len(position_values), len(item_values))
    for i in range(n_to_match):
        max_unplaced_item_total += position_values[i] * item_values[i]
    
    # Normalize by total number of items
    max_item_component = (current_item_total + max_unplaced_item_total) / n_items
    
    #-------------------------------------------------------------------------
    # Pair component upper bound
    #-------------------------------------------------------------------------
    # Count current pairs (within placed items)
    n_placed_pairs = 0
    for i in range(n_items):
        if mapping[i] >= 0:
            for j in range(i + 1, n_items):
                if mapping[j] >= 0:
                    n_placed_pairs += 2  # Both directions
    
    # Current pair score contribution (unnormalized)
    current_pair_total = current_pair_score * n_placed_pairs if n_placed_pairs > 0 else 0.0
    
    # Upper bound for future pairs
    future_pair_total = 0.0
    n_future_pairs = 0
    
    # 1. Pairs between placed and unplaced items
    placed_indices = np.where(mapping >= 0)[0]
    for p_idx in placed_indices:
        p_pos = mapping[p_idx]
        for u_idx in unplaced:
            # Find best position for this unplaced item w.r.t. this placed item
            best_pair_score = 0.0
            for a_pos in available:
                fwd = item_pair_score_matrix[p_idx, u_idx] * position_score_matrix[p_pos, a_pos]
                bwd = item_pair_score_matrix[u_idx, p_idx] * position_score_matrix[a_pos, p_pos]
                pair_score = fwd + bwd
                best_pair_score = max(best_pair_score, pair_score)
            future_pair_total += best_pair_score
            n_future_pairs += 2
    
    # 2. Pairs among unplaced items
    if n_unplaced >= 2:
        # Collect all possible item pair scores
        unplaced_item_pairs = []
        for i in range(n_unplaced):
            for j in range(i + 1, n_unplaced):
                i_idx = unplaced[i]
                j_idx = unplaced[j]
                score = item_pair_score_matrix[i_idx, j_idx] + item_pair_score_matrix[j_idx, i_idx]
                unplaced_item_pairs.append(score)
        
        # Collect all possible position pair scores
        available_pos_pairs = []
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                i_pos = available[i]
                j_pos = available[j]
                score = position_score_matrix[i_pos, j_pos] + position_score_matrix[j_pos, i_pos]
                available_pos_pairs.append(score)
        
        # Sort both in descending order
        unplaced_item_pairs = np.array(unplaced_item_pairs)
        available_pos_pairs = np.array(available_pos_pairs)
        unplaced_item_pairs = np.sort(unplaced_item_pairs)[::-1]
        available_pos_pairs = np.sort(available_pos_pairs)[::-1]
        
        # Greedy matching
        n_pairs = min(len(unplaced_item_pairs), len(available_pos_pairs))
        for i in range(n_pairs):
            future_pair_total += unplaced_item_pairs[i] * available_pos_pairs[i] / 2.0
            n_future_pairs += 2
    
    # 3. Cross-interactions with already-assigned items
    if n_assigned > 0 and cross_item_pair_matrix is not None:
        # For each unplaced item, estimate best cross-interactions
        for u_idx in unplaced:
            for a_idx in range(n_assigned):
                # Find best position for this cross-interaction
                best_cross_score = 0.0
                for a_pos in available:
                    fwd = cross_item_pair_matrix[u_idx, a_idx] * cross_position_pair_matrix[a_pos, a_idx]
                    bwd = reverse_cross_item_pair_matrix[a_idx, u_idx] * reverse_cross_position_pair_matrix[a_idx, a_pos]
                    cross_score = fwd + bwd
                    best_cross_score = max(best_cross_score, cross_score)
                future_pair_total += best_cross_score
                n_future_pairs += 2
    
    # Total pairs for normalization
    total_pairs = n_placed_pairs + n_future_pairs
    
    # Normalize pair component
    max_pair_component = 0.0
    if total_pairs > 0:
        max_pair_component = (current_pair_total + future_pair_total) / total_pairs
    
    #-------------------------------------------------------------------------
    # Combine components based on scoring mode
    #-------------------------------------------------------------------------
    if scoring_mode == 'item_only':
        return max_item_component
    elif scoring_mode == 'pair_only':
        return max_pair_component
    else:  # combined
        return max_item_component * max_pair_component
    
@jit(nopython=True, fastmath=True)
def compute_placed_score(placed_item_indices, placed_position_indices, item_scores_full, item_pair_score_matrix_full, position_score_matrix_full):
    """JIT-compiled function to compute score of placed items."""
    score = 0.0
    
    # Check if there are any placed items
    n_placed = len(placed_item_indices)
    if n_placed == 0:
        return score
    
    # Individual scores
    for i in range(n_placed):
        item_idx = placed_item_indices[i]
        pos_idx = placed_position_indices[i]
        score += item_scores_full[item_idx] * position_score_matrix_full[pos_idx, pos_idx]
    
    # Pair scores
    for i in range(n_placed):
        for j in range(i + 1, n_placed):
            item_i = placed_item_indices[i]
            item_j = placed_item_indices[j]
            pos_i = placed_position_indices[i] 
            pos_j = placed_position_indices[j]
            
            score += item_pair_score_matrix_full[item_i, item_j] * position_score_matrix_full[pos_i, pos_j]
            score += item_pair_score_matrix_full[item_j, item_i] * position_score_matrix_full[pos_j, pos_i]
    
    return score

def compute_individual_assignment(all_unplaced, all_available, item_scores_full, position_score_matrix_full):
    """Compute individual assignments using Hungarian algorithm."""
    from scipy.optimize import linear_sum_assignment
    
    n_unplaced = len(all_unplaced)
    n_available = len(all_available)
    
    if n_unplaced == 0 or n_available == 0:
        return 0.0
        
    # Create cost matrix
    max_dim = max(n_unplaced, n_available)
    cost_matrix = np.zeros((max_dim, max_dim))
    
    # Fill cost matrix (negative because we're maximizing)
    for i in range(n_unplaced):
        for j in range(n_available):
            item_idx = all_unplaced[i]
            pos_idx = all_available[j]
            cost_matrix[i, j] = -(item_scores_full[item_idx] * 
                                 position_score_matrix_full[pos_idx, pos_idx])
    
    # Fill unused entries
    if n_unplaced < max_dim or n_available < max_dim:
        for i in range(n_unplaced, max_dim):
            cost_matrix[i, :] = 0  # No penalty for not using these rows
        for j in range(n_available, max_dim):
            cost_matrix[:, j] = 0  # No penalty for not using these columns
    
    # Run Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Calculate score
    score = 0.0
    for i, j in zip(row_ind, col_ind):
        if i < n_unplaced and j < n_available:  # Only count real assignments
            item_idx = all_unplaced[i]
            pos_idx = all_available[j]
            score -= cost_matrix[i, j]  # Negative because we stored negative scores
    
    return score

@jit(nopython=True, fastmath=True)
def compute_pair_scores(all_unplaced_array, all_available_array, item_pair_score_matrix_full, position_score_matrix_full):
    """
    JIT-compiled function to compute pair score bounds.
    Uses the greedy approach of matching highest item pair scores with highest position pair scores.
    
    Args:
        all_unplaced_array: NumPy array of indices of unplaced items
        all_available_array: NumPy array of indices of available positions
        item_pair_score_matrix_full: Matrix of item pair scores
        position_score_matrix_full: Matrix of position pair scores
    
    Returns:
        Upper bound on the pair score component
    """
    score = 0.0
    
    n_unplaced = len(all_unplaced_array)
    n_available = len(all_available_array)
    
    if n_unplaced <= 1 or n_available <= 1:
        return score
    
    # Calculate number of pairs
    n_item_pairs = n_unplaced * (n_unplaced - 1) // 2
    n_pos_pairs = n_available * (n_available - 1) // 2
    
    # Create arrays to hold pair scores
    item_pair_scores = np.zeros(n_item_pairs, dtype=np.float32)
    pos_pair_scores = np.zeros(n_pos_pairs, dtype=np.float32)
    
    # Extract item pair scores (both directions)
    idx = 0
    for i in range(n_unplaced):
        for j in range(i + 1, n_unplaced):
            item_i = all_unplaced_array[i]
            item_j = all_unplaced_array[j]
            # Sum of both directions
            item_pair_scores[idx] = (item_pair_score_matrix_full[item_i, item_j] + 
                                     item_pair_score_matrix_full[item_j, item_i])
            idx += 1
    
    # Extract position pair scores (both directions)
    idx = 0
    for i in range(n_available):
        for j in range(i + 1, n_available):
            pos_i = all_available_array[i]
            pos_j = all_available_array[j]
            # Sum of both directions
            pos_pair_scores[idx] = (position_score_matrix_full[pos_i, pos_j] + 
                                   position_score_matrix_full[pos_j, pos_i])
            idx += 1
    
    # Sort scores in descending order
    item_pair_scores = np.sort(item_pair_scores)[::-1]
    pos_pair_scores = np.sort(pos_pair_scores)[::-1]
    
    # Match highest item pair scores with highest position pair scores
    n_pairs = min(n_item_pairs, n_pos_pairs)
    for i in range(n_pairs):
        # Divide by 2 because we counted both directions
        score += (item_pair_scores[i] * pos_pair_scores[i]) / 2.0
    
    return score

@jit(nopython=True, fastmath=True)
def compute_cross_interactions(placed_items_idxs, placed_items_pos, all_unplaced_array, all_available_array, 
                              item_pair_score_matrix_full, position_score_matrix_full):
    """
    JIT-compiled function to compute cross-interaction bounds.
    For each placed item, find best possible interactions with unplaced items.
    
    Args:
        placed_items_idxs: NumPy array of indices of placed items
        placed_items_pos: NumPy array of positions of placed items
        all_unplaced_array: NumPy array of indices of unplaced items
        all_available_array: NumPy array of indices of available positions
        item_pair_score_matrix_full: Matrix of item pair scores
        position_score_matrix_full: Matrix of position pair scores
    
    Returns:
        Upper bound on the cross-interaction component
    """
    score = 0.0
    
    n_placed = len(placed_items_idxs)
    n_unplaced = len(all_unplaced_array)
    n_available = len(all_available_array)
    
    if n_placed == 0 or n_unplaced == 0 or n_available == 0:
        return score
    
    # For each placed item, find best possible interaction with each unplaced item
    for p in range(n_placed):
        item_idx = placed_items_idxs[p]
        pos_idx = placed_items_pos[p]
        
        for u in range(n_unplaced):
            unplaced_idx = all_unplaced_array[u]
            best_score = 0.0
            
            # Find best position for this unplaced item
            for a in range(n_available):
                avail_pos = all_available_array[a]
                
                # Forward direction (placed -> unplaced)
                fwd = (item_pair_score_matrix_full[item_idx, unplaced_idx] * 
                      position_score_matrix_full[pos_idx, avail_pos])
                
                # Backward direction (unplaced -> placed)
                bwd = (item_pair_score_matrix_full[unplaced_idx, item_idx] * 
                      position_score_matrix_full[avail_pos, pos_idx])
                
                # Total interaction score
                pair_score = fwd + bwd
                
                if pair_score > best_score:
                    best_score = pair_score
            
            # Add best possible score for this pair
            score += best_score
    
    return score

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
    total_score, item_score, pair_score = calculate_score_for_new_items(
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
    validate_bounds: bool = False
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
    pruned_by_bound = 0
    
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
    
    # Initialize mapping and used positions
    initial_mapping = np.full(n_items_to_assign, -1, dtype=np.int16)
    initial_used = np.zeros(n_positions_to_assign, dtype=np.bool_)

    # Track statistics
    processed_nodes = 0
    pruned_count = 0
    explored_count = 0
    last_progress_update = 0
    progress_update_interval = 1000000  # Print progress every X permutations

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
    def phase2_dfs(initial_mapping, initial_used, initial_depth, pbar):
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
                
                # Calculate score
                total_score, item_component, item_pair_component = calculate_score_for_new_items(
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
                if len(solutions) < n_solutions or margin > 0:
                    solution = (
                        total_score, item_component, item_pair_component, mapping.tolist()
                    )
                    solutions.append(solution)
                    solutions.sort(key=lambda x: x[0])
                    if len(solutions) > n_solutions:
                        solutions.pop(0)
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
                
                # Pruning decision
                should_prune = False
                if len(solutions) > 0:

                    # Prepare full arrays if not already done
                    if 'item_scores_full' not in locals():
                        all_items = items_to_assign + items_assigned
                        all_positions = positions_to_assign + positions_assigned
                        arrays_full = prepare_arrays(
                            all_items, all_positions,
                            norm_item_scores, norm_item_pair_scores,
                            norm_position_scores, norm_position_pair_scores,
                            missing_item_pair_norm_score, missing_position_pair_norm_score
                        )
                        item_scores_full, item_pair_score_matrix_full, position_score_matrix_full = arrays_full[:3]

                    upper_bound = calculate_upper_bound(
                        mapping, used,
                        position_score_matrix_full,
                        item_scores_full, 
                        item_pair_score_matrix_full,
                        cross_item_pair_matrix=cross_item_pair_matrix,
                        cross_position_pair_matrix=cross_position_pair_matrix,
                        reverse_cross_item_pair_matrix=reverse_cross_item_pair_matrix,
                        reverse_cross_position_pair_matrix=reverse_cross_position_pair_matrix,
                        items_assigned=items_assigned,
                        positions_assigned=positions_assigned
                    )
                    
                    # Validate the bound
                    if validate_bounds and len(solutions) > 0:
                        # Calculate parameters for the debug function
                        n_phase1_items = len(items_to_assign)
                        n_phase1_positions = len(positions_to_assign)
                        n_phase2_items = len(items_assigned) if items_assigned else 0
                        n_phase2_positions = len(positions_assigned) if positions_assigned else 0
                        
                        # Get the full arrays if not already calculated
                        if 'item_scores_full' not in locals():
                            all_items = items_to_assign + items_assigned
                            all_positions = positions_to_assign + positions_assigned
                            arrays_full = prepare_arrays(
                                all_items, all_positions,
                                norm_item_scores, norm_item_pair_scores,
                                norm_position_scores, norm_position_pair_scores,
                                missing_item_pair_norm_score, missing_position_pair_norm_score
                            )
                            item_scores_full, item_pair_score_matrix_full, position_score_matrix_full = arrays_full[:3]
                        
                        # Debug the bound
                        gap, _, _ = debug_upper_bound(
                            mapping, used,
                            n_phase1_items, n_phase1_positions,
                            n_phase2_items, n_phase2_positions,
                            item_scores_full, item_pair_score_matrix_full,
                            position_score_matrix_full,
                            items_to_assign, positions_to_assign,
                            items_assigned, positions_assigned,
                            norm_item_scores, norm_item_pair_scores,
                            norm_position_scores, norm_position_pair_scores,
                            verbose=False
                        )

                        if gap < 0:
                            print(f"\n⚠️ Invalid bound detected: {gap:.6f}")
                            negative_bound_count += 1

                    margin = upper_bound - worst_top_n_score
                    
                    epsilon = 0.0001  # Small value that shouldn't affect optimality in practice
                    if margin < epsilon:
                        should_prune = True
                
                # Restore the original state before making a copy
                mapping[current_item_idx] = -1  # Restore to unassigned
                used[pos] = False
                
                if should_prune:
                    pruned_count += 1
                    pruned_by_bound += 1
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

        print(f"Pruned by bounds: {pruned_by_bound}/{processed_nodes} nodes ({pruned_by_bound/processed_nodes*100:.2f}%)")

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
                phase2_dfs(phase1_mapping, phase1_used, initial_depth, pbar)
                
                current_phase1_solution_index += 1
        else:
            # Calculate initial depth based on assigned items
            initial_depth = sum(1 for i in range(n_items_to_assign))
            
            # Start DFS from this phase 1 solution
            phase2_dfs(initial_mapping, initial_used, 0, pbar)

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
        
        # Calculate score for the complete layout if there are pre-assigned items
        if items_assigned:
            # Prepare arrays for the complete layout
            complete_mapping_array, complete_item_scores, complete_item_pair_score_matrix, complete_position_score_matrix = (
                prepare_complete_arrays(
                    all_items, all_positions, 
                    norm_item_scores, norm_item_pair_scores,
                    norm_position_scores, norm_position_pair_scores,
                    missing_item_pair_norm_score, missing_position_pair_norm_score
                )
            )
            
            # Calculate score for the complete layout
            complete_score, complete_item_score, complete_item_pair_score = (
                calculate_score_for_new_items(
                    complete_mapping_array,
                    complete_position_score_matrix, 
                    complete_item_scores,
                    complete_item_pair_score_matrix,
                    None,  # cross_item_pair_matrix
                    None,  # cross_position_pair_matrix
                    None,  # reverse_cross_item_pair_matrix
                    None   # reverse_cross_position_pair_matrix
                )
            )
            
            # Use the complete scores
            score = complete_score
            item_score = complete_item_score
            item_pair_score = complete_item_pair_score
        
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
    full_upper_bound = calculate_upper_bound(
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
    if gap < 0:
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
        actual_score, _, _ = calculate_score_for_new_items(completed_mapping, 
                                position_score_matrix, 
                                item_scores, 
                                item_pair_score_matrix,
                                None,  # cross_item_pair_matrix
                                None,  # cross_position_pair_matrix
                                None,  # reverse_cross_item_pair_matrix
                                None   # reverse_cross_position_pair_matrix
        )
            
        # Calculate upper bound estimate
        estimate = calculate_upper_bound(mapping, used, position_score_matrix, 
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
    n_trials=10000, 
    print_progress=True
):
    """
    Statistically validate that upper bounds are never lower than achievable scores
    by testing many random partial mappings.
    
    Args:
        config: Configuration dictionary
        n_trials: Number of random trials to run
        print_progress: Whether to print progress
        
    Returns:
        dict: Statistics about bound validity and tightness
    """
    # Load data
    items_to_assign = config['optimization']['items_to_assign']
    positions_to_assign = config['optimization']['positions_to_assign']
    items_assigned = config['optimization'].get('items_assigned', '')
    positions_assigned = config['optimization'].get('positions_assigned', '')
    
    # Load scores
    norm_item_scores, norm_item_pair_scores, norm_position_scores, norm_position_pair_scores = load_scores(config)
    
    # Prepare full arrays
    all_items = items_to_assign + items_assigned
    all_positions = positions_to_assign + positions_assigned
    arrays_full = prepare_arrays(
        all_items, all_positions,
        norm_item_scores, norm_item_pair_scores,
        norm_position_scores, norm_position_pair_scores
    )
    item_scores_full, item_pair_score_matrix_full, position_score_matrix_full = arrays_full[:3]
    
    # Setup for trials
    n_items = len(items_to_assign)
    n_positions = len(positions_to_assign)
    phase1_items = len(items_to_assign)
    phase1_positions = len(positions_to_assign)
    phase2_items = len(items_assigned)
    phase2_positions = len(positions_assigned)
    
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
        
        # Calculate upper bound
        upper_bound = calculate_upper_bound(
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
        
        # Complete solution and get actual score
        completed_mapping, actual_score, _ = complete_and_score_full_layout(
            mapping.copy(), used.copy(),
            items_to_assign, positions_to_assign,
            items_assigned, positions_assigned,
            norm_item_scores, norm_item_pair_scores,
            norm_position_scores, norm_position_pair_scores
        )
        
        # Calculate gap and check validity
        gap = upper_bound - actual_score
        gaps.append(gap)
        
        if gap < 0:
            invalid_bounds += 1
            if print_progress:
                print(f"\nInvalid bound detected: {upper_bound:.6f} < {actual_score:.6f}")
                print(f"Gap: {gap:.6f}, Depth: {depth}")
    
    # Analyze results
    min_gap = min(gaps)
    max_gap = max(gaps)
    avg_gap = sum(gaps) / len(gaps)
    
    # Results by depth
    depth_stats = {}
    for d in sorted(set(depths)):
        d_gaps = [gaps[i] for i in range(len(gaps)) if depths[i] == d]
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
def optimize_layout(config: dict, verbose: bool = False, 
                    analyze_bounds: bool = False,     
                    missing_item_pair_norm_score: float = 1.0,
                    missing_position_pair_norm_score: float = 1.0) -> None:
    """
    Main optimization function. Uses branch-and-bound search.
    """
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
    if n_layouts > 1:
        print(f"\nFinding top {n_layouts} solutions")
    else:
        print("\nFinding top solution")

    if items_to_constrain:
        print(f"  - {len(items_to_constrain)} constrained items: {items_to_constrain}")
        print(f"  - {len(positions_to_constrain)} constrained positions: {positions_to_constrain}")

    # Run optimization
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
        validate_bounds=validate_bounds
    )
    
    # Before printing results, recalculate scores for the complete layout
    updated_results = []
    for score, mapping, detailed_scores in results:
        # Skip recalculation if there are no pre-assigned items
        if not items_assigned:
            updated_results.append((score, mapping, detailed_scores))
            continue
        
        # Build complete mapping including pre-assigned items
        complete_mapping_dict = dict(zip(items_assigned, positions_assigned))
        complete_mapping_dict.update(mapping)
        all_items = list(complete_mapping_dict.keys())
        all_positions = list(complete_mapping_dict.values())
        
        # Prepare arrays for the complete layout
        complete_mapping, complete_item_scores, complete_item_pair_score_matrix, complete_position_score_matrix = prepare_complete_arrays(
            all_items, all_positions, 
            norm_item_scores, norm_item_pair_scores,
            norm_position_scores, norm_position_pair_scores,
            missing_item_pair_norm_score, missing_position_pair_norm_score
        )
        
        # Calculate score for the complete layout
        complete_score, complete_item_score, complete_item_pair_score = calculate_score_for_new_items(
            complete_mapping,
            complete_position_score_matrix, 
            complete_item_scores,
            complete_item_pair_score_matrix,
            None,  # cross_item_pair_matrix
            None,  # cross_position_pair_matrix
            None,  # reverse_cross_item_pair_matrix
            None   # reverse_cross_position_pair_matrix
        )
        
        # Update the score and detailed scores
        updated_detailed_scores = {'total': {
            'total_score': complete_score,
            'item_pair_score': complete_item_pair_score,
            'item_score': complete_item_score
        }}
        
        updated_results.append((complete_score, mapping, updated_detailed_scores))
    
    # Sort the updated results
    updated_results = sorted(
        updated_results,
        key=lambda x: x[0],
        reverse=True
    )
    
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

if __name__ == "__main__":
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Optimize keyboard layout.')
        parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
        parser.add_argument('--verbose', action='store_true',
                        help='Print detailed scoring information')
        args = parser.parse_args()
        
        start_time = time.time()
        
        # Load configuration from specified path
        config = load_config(args.config)

        # Validate bounds
        validate_bounds = True  # Set to True for debugging runs
        if validate_bounds:
            print("Validating upper bounds statistically...")
            # Run statistical validation of bounds
            validate_results = validate_bounds_statistically(
                config, 
                n_trials=1000  # Fewer trials for quicker validation
            )
            if not validate_results['valid']:
                print("ERROR: Upper bound validation failed. Optimization may not find optimal solutions.")
                if input("Continue anyway? (y/n): ").lower() != 'y':
                    sys.exit(1)

        # Optimize the layout
        optimize_layout(config, 
                        verbose=args.verbose, 
                        analyze_bounds=False,     
                        missing_item_pair_norm_score=1.0,
                        missing_position_pair_norm_score=1.0)
        
        elapsed = time.time() - start_time
        print(f"Total runtime: {timedelta(seconds=int(elapsed))}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
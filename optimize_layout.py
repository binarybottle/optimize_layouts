# optimize_layouts/optimize_layout.py
"""
Memory-efficient item-to-position layout optimization using branch and bound search.

This script uses a branch and bound algorithm to find optimal positions 
for items and item pairs by jointly considering two scores: 
item/item_pair scores and position/position_pair scores.

See README for more details.

>> python optimize_layout.py
"""
import yaml
import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm
import psutil

import os
from math import perm
import time
from datetime import datetime, timedelta
import csv
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import argparse

#-----------------------------------------------------------------------------
# Loading, validating, and saving functions
#-----------------------------------------------------------------------------
def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from yaml file and normalize item case and numeric types."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create necessary output directories
    output_dirs = [config['paths']['output']['layout_results_folder']]
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)

    # Convert weights to float32
    config['optimization']['scoring']['item_weight'] = np.float32(
        config['optimization']['scoring']['item_weight'])
    config['optimization']['scoring']['item_pair_weight'] = np.float32(
        config['optimization']['scoring']['item_pair_weight'])
    
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
    item_weight = config['optimization']['scoring']['item_weight']
    item_pair_weight = config['optimization']['scoring']['item_pair_weight']

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
    print(f"Item weight: {item_weight}")
    print(f"Item-pair weight: {item_pair_weight}")

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
        
        # Create cross-position matrix (positions_to_assign <-> positions_assigned)
        cross_position_pair_matrix = np.zeros((n_positions_to_assign, len(positions_assigned)), dtype=np.float32)
        for i, p1 in enumerate(positions_to_assign):
            for j, p2 in enumerate(positions_assigned):
                cross_position_pair_matrix[i, j] = norm_position_pair_scores.get((p1.lower(), p2.lower()),
                                                                               missing_position_pair_norm_score)
        
        # Return matrices including cross-interaction matrices
        return (item_scores, item_pair_score_matrix, position_score_matrix, 
                cross_item_pair_matrix, cross_position_pair_matrix,
                items_assigned, positions_assigned)
    
    else:
        # No assigned items, just return the regular matrices
        return item_scores, item_pair_score_matrix, position_score_matrix
    
def load_scores(config: dict):
    """Load scores."""
    item_weight = config['optimization']['scoring']['item_weight']
    item_pair_weight = config['optimization']['scoring']['item_pair_weight']

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
    # Load item scores if item_weight > 0
    #-------------------------------------------------------------------------
    if item_weight > 0:
        #print("\nLoading item scores...")
        item_df = pd.read_csv(config['paths']['input']['item_scores_file'], 
                              dtype={'item_pair': str})
        norm_scores = item_df['score'].values
            
        for idx, row in item_df.iterrows():
            norm_item_scores[row['item'].lower()] = np.float32(norm_scores[idx])
    

    #-------------------------------------------------------------------------
    # Load pair scores if item_pair_weight > 0
    #-------------------------------------------------------------------------
    if item_pair_weight > 0:
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
    # Generate timestamp and set output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config['paths']['output']['layout_results_folder'],
                               f"layout_results_{timestamp}.csv")
    
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
        writer.writerow(['Item weight', opt['scoring']['item_weight']])
        writer.writerow(['Item-pair weight', opt['scoring']['item_pair_weight']])
        writer.writerow([])  # Empty row for separation
        
        # Write results header
        writer.writerow([
            'Items',
            'Positions',
            'Optimized Items',
            'Optimized Positions',
            'Rank',
            'Total score',
            'Item score (unweighted)',
            'Item-pair score (unweighted)'
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
            unweighted_item_pair_score = first_entry['unweighted_item_pair_score']
            unweighted_item_score = first_entry['unweighted_item_score']
            
            writer.writerow([
                all_items,                 # All items (pre-assigned + optimized)
                all_positions,             # All positions (pre-assigned + optimized)
                optimized_items,           # Just the optimized items
                optimized_positions,       # Just the optimized positions
                rank,
                f"{score:.12f}",
                f"{unweighted_item_score:.12f}",
                f"{unweighted_item_pair_score:.12f}"
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

def update_progress_bar(pbar, processed_perms: int, start_time: float, total_perms: int) -> None:
    """Update progress bar with accurate statistics."""
    current_time = time.time()
    elapsed = current_time - start_time   
    if elapsed > 0:
        perms_per_second = processed_perms / elapsed
        percent_explored = (processed_perms / total_perms) * 100 if total_perms > 0 else 0
        remaining_perms = total_perms - processed_perms
        eta_minutes = (remaining_perms / perms_per_second) / 60 if perms_per_second > 0 else 0
        
        pbar.set_postfix({
            'Perms/sec': f"{perms_per_second:,.0f}",
            'Explored': f"{percent_explored:.1f}%",
            'ETA': 'Complete' if processed_perms >= total_perms else f"{eta_minutes:.1f}m",
            'Memory': f"{psutil.Process().memory_info().rss/1e9:.1f}GB"
        })

def print_top_results(results: List[Tuple[float, Dict[str, str], Dict[str, dict]]],
                     config: dict,
                     n: int = None,
                     items_to_display: str = None,
                     positions_to_display: str = None,
                     print_keyboard: bool = True,
                     verbose: bool = False) -> None:
    """
    Print the top N results with their scores and mappings.
    
    Args:
        results: List of (score, mapping, detailed_scores) tuples
        config: Configuration dictionary
        n: Number of layouts to display (defaults to config['optimization']['nlayouts'])
        items_to_display: items that have already been assigned
        positions_to_display: positions that have already been assigned
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
        missing_item_pair_norm_score = config['optimization']['scoring'].get('missing_item_pair_norm_score', 1.0)
        missing_position_pair_norm_score = config['optimization']['scoring'].get('missing_position_pair_norm_score', 1.0)
    
    if len(results) > 1:
        print(f"\nTop {n} scoring layouts:")
    else:
        print(f"\nTop-scoring layout:")
    
    for i, (score, mapping, detailed_scores) in enumerate(results[:n], 1):
        print(f"\n#{i}: Score: {score:.12f}")
        
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
            unweighted_item_score = component_scores.get('unweighted_item_score', 0.0)
            unweighted_item_pair_score = component_scores.get('unweighted_item_pair_score', 0.0)
            
            # Display scoring weights and scores
            item_weight = config['optimization']['scoring']['item_weight']
            item_pair_weight = config['optimization']['scoring']['item_pair_weight']
            print(f"Scoring weights:")
            print(f"  Item weight:      {item_weight:.2f}")
            print(f"  Item-pair weight: {item_pair_weight:.2f}")
            print(f"Scores:")
            print(f"  Unweighted item score:      {unweighted_item_score:.12f}")
            print(f"  Unweighted item-pair score: {unweighted_item_pair_score:.12f}")
            print(f"  Total weighted score:       {score:.12f}")
            
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
def calculate_score(
    mapping: np.ndarray,
    position_score_matrix: np.ndarray,
    item_scores: np.ndarray,
    item_pair_score_matrix: np.ndarray,
    item_weight: float,
    item_pair_weight: float,
    cross_item_pair_matrix=None,
    cross_position_pair_matrix=None,
    items_assigned=None,
    positions_assigned=None
) -> Tuple[float, float, float]:
    """
    Calculate layout score with option to return component scores.
    Handles both single and multi-solution scoring needs.
    
    Args:
        mapping: Array of position indices for each item (-1 for unplaced)
        position_score_matrix: Matrix of position and position-pair scores
        item_scores: Array of item scores
        item_pair_score_matrix: Matrix of item-pair scores
        item_weight: Weight for item score component
        item_pair_weight: Weight for item-pair score component
        cross_item_pair_matrix: Matrix of cross-interactions between items_to_assign and items_assigned
        cross_position_pair_matrix: Matrix of cross-interactions between positions_to_assign and positions_assigned
        items_assigned: List of already assigned items
        positions_assigned: List of already assigned positions
    
    Returns:
        tuple of (total_score, item_component, pair_component)
    """
    item_component = np.float32(0.0)
    item_pair_component = np.float32(0.0)
    
    # Get number of items
    n_items = len(mapping)
    n_placed_items = 0
    
    # Only calculate item component if weight > 0
    if item_weight > 0:
        for i in range(n_items):
            pos = mapping[i]
            if pos >= 0:
                item_component += position_score_matrix[pos, pos] * item_scores[i]
                n_placed_items += 1
        
        # Normalize by number of placed items
        if n_placed_items > 0:
            item_component /= n_placed_items
    
    # Only calculate pair component if weight > 0    
    if item_pair_weight > 0:
        pair_count = 0
        
        # Internal pairs (between items being optimized)
        for i in range(n_items):
            pos_i = mapping[i]
            if pos_i >= 0:
                # Pairs between items being optimized
                for j in range(i + 1, n_items):
                    pos_j = mapping[j]
                    if pos_j >= 0:
                        # Score both directions and add to total
                        fwd_score = position_score_matrix[pos_i, pos_j] * item_pair_score_matrix[i, j]
                        bck_score = position_score_matrix[pos_j, pos_i] * item_pair_score_matrix[j, i]
                        item_pair_component += (fwd_score + bck_score)
                        pair_count += 2  # Count both directions
        
        # Normalize pair score by number of pairs
        if pair_count > 0:
            item_pair_component /= pair_count
    
    # Calculate weighted total
    total_score = item_weight * item_component + item_pair_weight * item_pair_component
    
    return total_score, item_component, item_pair_component

def calculate_cross_interactions(
    mapping: np.ndarray,
    items_to_assign: str,
    positions_to_assign: str,
    items_assigned: str,
    positions_assigned: str,
    norm_item_pair_scores: Dict[Tuple[str, str], float],
    norm_position_pair_scores: Dict[Tuple[str, str], float],
    missing_item_pair_norm_score: float = 0.0,
    missing_position_pair_norm_score: float = 0.0
) -> float:
    """Calculate interaction scores between fixed items and items being optimized."""
    if not items_assigned or not positions_assigned:
        return 0.0
        
    cross_score = 0.0
    interaction_count = 0
    
    for i, item1 in enumerate(items_to_assign):
        pos1_idx = mapping[i]
        if pos1_idx < 0:  # Skip if not assigned yet
            continue
            
        pos1 = positions_to_assign[pos1_idx]
        
        for j, item2 in enumerate(items_assigned):
            pos2 = positions_assigned[j]
            
            # Forward interaction (assigned → fixed)
            pair_key = (item1.lower(), item2.lower())
            pos_key = (pos1.lower(), pos2.lower())
            
            pair_score = norm_item_pair_scores.get(pair_key, missing_item_pair_norm_score)
            pos_score = norm_position_pair_scores.get(pos_key, missing_position_pair_norm_score)
            
            cross_score += pos_score * pair_score
            
            # Backward interaction (fixed → assigned)
            reverse_pair_key = (item2.lower(), item1.lower())
            reverse_pos_key = (pos2.lower(), pos1.lower())
            
            reverse_pair_score = norm_item_pair_scores.get(reverse_pair_key, missing_item_pair_norm_score)
            reverse_pos_score = norm_position_pair_scores.get(reverse_pos_key, missing_position_pair_norm_score)
            
            cross_score += reverse_pos_score * reverse_pair_score
            
            interaction_count += 2  # Count both directions
    
    # Normalize
    return cross_score / interaction_count if interaction_count > 0 else 0.0
            
@jit(nopython=True, fastmath=True)
def calculate_upper_bound(
    mapping: np.ndarray,
    used: np.ndarray,
    position_score_matrix: np.ndarray,
    item_scores: np.ndarray,
    item_pair_score_matrix: np.ndarray,
    item_weight: float,
    item_pair_weight: float,
    best_score: float = np.float32(-np.inf),
    single_solution: bool = False,
    depth: int = None,
    cross_item_pair_matrix=None,
    cross_position_pair_matrix=None,
    items_assigned=None,
    positions_assigned=None
) -> float:
    """
    Calculate upper bound on best possible score from this node.
    Optimized for both single and multi-solution search.
    
    Args:
        mapping: Current partial mapping of items to positions
        used: Boolean array of used positions
        position_score_matrix: Matrix of position/position-pair scores
        item_scores: Array of item scores
        item_pair_score_matrix: Matrix of item-pair scores
        item_weight: Weight for item score component
        item_pair_weight: Weight for item-pair score component
        best_score: Current best score (used for pruning in single-solution mode)
        single_solution: If True, use aggressive pruning optimized for single solution
        depth: Search depth (only used in multi-solution mode)
        cross_item_pair_matrix: Matrix of cross-interactions between items_to_assign and items_assigned
        cross_position_pair_matrix: Matrix of cross-interactions between positions_to_assign and positions_assigned
        items_assigned: List of already assigned items
        positions_assigned: List of already assigned positions
    
    Returns:
        Upper bound on best possible score from this node
    """
    debug_print = False
    
    # Get current score from placed items - pass cross-interaction matrices
    current_score, current_unweighted_item_score, current_unweighted_item_pair_score = calculate_score(
        mapping, position_score_matrix, item_scores,
        item_pair_score_matrix, item_weight, item_pair_weight,
        cross_item_pair_matrix, cross_position_pair_matrix,
        items_assigned, positions_assigned
    )
    
    if debug_print and depth is not None and depth < 2:
        print("\nUPPER BOUND CALCULATION:")
        print("    Current mapping:", mapping)
        print("    Current scores:")
        print("      Item score: ", current_unweighted_item_score)
        print("      Pair score: ", current_unweighted_item_pair_score)
    
    # Quick rejection check with floating point margin
    if single_solution:
        # (1 is the maximum possible contribution for each normalized score)
        margin = ((current_unweighted_item_score + 1) * item_weight +
                  (current_unweighted_item_pair_score + 1) * item_pair_weight) - \
                 (best_score - np.abs(best_score) * np.finfo(np.float32).eps)
        if margin <= 0:
            if debug_print and depth is not None and depth < 2:
                print("    Quick rejection: True")
            return -np.inf
    
    # Find unplaced items and available positions
    unplaced = np.where(mapping < 0)[0]
    available = np.where(~used)[0]
    
    if debug_print and depth is not None and depth < 2:
        print("    Unplaced items:", unplaced)
        print("    Available positions:", available)
    
    if len(unplaced) == 0:
        return current_score
    
    #-------------------------------------------------------------------------
    # Single-item component
    #-------------------------------------------------------------------------
    # Get position scores for remaining positions
    position_values = np.zeros(len(available))
    for i, pos in enumerate(available):
        position_values[i] = position_score_matrix[pos, pos]
    position_values.sort()
    position_values = position_values[::-1]  # Highest to lowest
    
    # Get scores for remaining items
    item_values = item_scores[unplaced]
    item_values.sort()
    item_values = item_values[::-1]  # Highest to lowest
    
    if debug_print and depth is not None and depth < 2:
        print("    Position values:", position_values)
        print("    Item values:", item_values)
    
    # Maximum possible item component
    len_values = min(len(position_values), len(item_values))
    max_item_component = np.sum(position_values[:len_values] *
                                item_values[:len_values]) / len(mapping)
    
    #-------------------------------------------------------------------------
    # Paired-item component
    #-------------------------------------------------------------------------
    # Maximum possible pair component
    max_pair_component = 0.0
    n_pairs = 0
    
    # 1. Pairs between placed and unplaced
    placed = np.where(mapping >= 0)[0]
    
    if debug_print and depth is not None and depth < 2:
        print("\n    Detail of pair calculations:")
        if len(placed) > 0:
            p_item = placed[0]  # t's index
            u_item = unplaced[0]  # h's index
            print("      T-H pair scores:")
            print("        t->h (item pair):", item_pair_score_matrix[p_item, u_item])
            print("        h->t (item pair):", item_pair_score_matrix[u_item, p_item])
            p_pos = mapping[p_item]  # t's position
            for pos in available:
                print("\n      Position scores for pos", p_pos, "->", pos, ":")
                print("        Forward:", position_score_matrix[p_pos, pos])
                print("        Backward:", position_score_matrix[pos, p_pos])
                score1 = item_pair_score_matrix[p_item, u_item] * position_score_matrix[p_pos, pos]
                score2 = item_pair_score_matrix[u_item, p_item] * position_score_matrix[pos, p_pos]
                print("        Products:")
                print("          Forward:", score1)
                print("          Backward:", score2)
                print("          Sum:", score1 + score2)
    
    for p_item in placed:
        p_pos = mapping[p_item]
        for u_item in unplaced:
            # Initialize best_sum before using it in max()
            best_sum = np.float32(-np.inf)
            for pos in available:
                score1 = item_pair_score_matrix[p_item, u_item] * position_score_matrix[p_pos, pos]
                score2 = item_pair_score_matrix[u_item, p_item] * position_score_matrix[pos, p_pos]
                current_sum = score1 + score2
                best_sum = max(best_sum, current_sum)
            
            max_pair_component += best_sum
            n_pairs += 2  # Count both directions
    
    # 2. Pairs between unplaced items
    n_unplaced_pairs = len(unplaced) * (len(unplaced) - 1)
    if n_unplaced_pairs > 0:
        # Collect and sort all pair score sums across both directions
        pair_scores = []
        for i in range(len(unplaced)):
            for j in range(i + 1, len(unplaced)):
                score = (item_pair_score_matrix[unplaced[i], unplaced[j]] +
                         item_pair_score_matrix[unplaced[j], unplaced[i]])
                pair_scores.append(score)
        pair_scores.sort()
        pair_scores.reverse()
        
        # Match with best position pairs
        pos_pair_scores = []
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                score = (position_score_matrix[available[i], available[j]] +
                         position_score_matrix[available[j], available[i]])
                pos_pair_scores.append(score)
        pos_pair_scores.sort()
        pos_pair_scores.reverse()
        
        # Take best possible matches
        n_pairs_to_match = min(len(pair_scores), len(pos_pair_scores))
        for i in range(n_pairs_to_match):
            max_pair_component += pair_scores[i] * pos_pair_scores[i]
            n_pairs += 2  # Count both directions

    # 3. Potential interactions with assigned items
    # Only include cross-interactions if all required components are present
    if (cross_item_pair_matrix is not None and 
        cross_position_pair_matrix is not None and 
        positions_assigned is not None and 
        len(positions_assigned) > 0):  # Check that positions_assigned is not empty
        
        for u_item in unplaced:
            best_cross_score = 0.0
            # Find best potential cross-interaction for each unplaced item
            for pos in available:
                cross_score = 0.0
                for j in range(len(positions_assigned)):
                    cross_score += (cross_position_pair_matrix[pos, j] *
                                   cross_item_pair_matrix[u_item, j])
                best_cross_score = max(best_cross_score, cross_score)
            max_pair_component += best_cross_score * 2  # Both directions
            n_pairs += 2  # Add pairs for both directions
    
    if n_pairs > 0:
        max_pair_component = max_pair_component / n_pairs
    
    #-------------------------------------------------------------------------
    # Combine components
    #-------------------------------------------------------------------------
    total_item_score = (current_unweighted_item_score + max_item_component) * item_weight
    total_pair_score = (current_unweighted_item_pair_score + max_pair_component) * item_pair_weight
    combined_score = float(total_item_score + total_pair_score)
    
    if debug_print and depth is not None and depth < 2:
        print("    Max additional:")
        print("      Item: ", max_item_component)
        print("      Pair: ", max_pair_component)
        print("    Final scores:")
        print("      Total item score:", total_item_score)
        print("      Total pair score:", total_pair_score)
        print("      Combined score:", combined_score)
    
    if single_solution:
        margin = combined_score - (best_score - np.abs(best_score) * np.finfo(np.float32).eps)
        return combined_score if margin > 0 else -np.inf
    else:
        return combined_score
               
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
    mapping = np.arange(n_items, dtype=np.int32)
    
    return mapping, item_scores, item_pair_score_matrix, position_score_matrix

def branch_and_bound_optimal_nsolutions(
    arrays: tuple,
    weights: tuple,
    config: dict,
    n_solutions: int = 5,
    norm_item_scores: Dict = None,
    norm_position_scores: Dict = None,
    norm_item_pair_scores: Dict = None,
    norm_position_pair_scores: Dict = None,
    missing_item_pair_norm_score: float = 0.0,
    missing_position_pair_norm_score: float = 0.0
) -> List[Tuple[float, Dict[str, str], Dict]]:
    """
    Branch and bound implementation using depth-first search. 
    
    Uses DFS instead of best-first search because:
    1. With a mathematically sound upper bound for pruning, the search order 
       doesn't affect optimality
    2. DFS requires only O(depth) memory vs O(width^depth) for best-first
    3. Simpler implementation without heap management complexity
    
    Search is conducted in two phases:
    Phase 1:
      - Finds all valid arrangements of constrained items (e.g., 'e', 't') 
        in constrained positions (e.g., F, D, J, K).
      - Each arrangement marks positions as used/assigned.

    Phase 2: For each Phase 1 solution, arrange remaining items
      - For each valid arrangement from Phase 1
        - Uses ONLY the positions that weren't assigned during Phase 1.
        - In other words, if a Phase 1 solution put 'e' in F and 't' in J, 
          then Phase 2 would use remaining positions (not F or J)
          to arrange the remaining items.

    Args:
        arrays: Tuple of (item_scores, item_pair_score_matrix, position_score_matrix)
        weights: Tuple of (item_weight, item_pair_weight)
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
    
    # Unpack arrays based on whether cross-interaction matrices are included
    if len(arrays) > 3:
        item_scores, item_pair_score_matrix, position_score_matrix, cross_item_pair_matrix, cross_position_pair_matrix, *_ = arrays
    else:
        item_scores, item_pair_score_matrix, position_score_matrix = arrays
        cross_item_pair_matrix = None
        cross_position_pair_matrix = None
    
    item_weight, item_pair_weight = weights
    
    # Initialize search structures
    solutions = []  # Will store (score, unweighted_scores, mapping) tuples
    worst_top_n_score = np.float32(-np.inf)
    
    # Initialize mapping and used positions
    initial_mapping = np.full(n_items_to_assign, -1, dtype=np.int32)
    initial_used = np.zeros(n_positions_to_assign, dtype=bool)

    # Track statistics
    processed_perms = 0
    #pruning_stats = {
        #'depth': defaultdict(int),  # Count pruned branches by depth
        #'margin': [],               # Track pruning margins (diff between upper bound and worst score)
        #'total_pruned': 0,
        #'total_explored': 0
    #}

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
    else:
        print("\nSkipping Phase 1 (no constrained items)")
        print(f"Phase 2: {total_perms_phase2:,} permutations")
    
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
                new_mapping = mapping.copy()
                new_mapping[current_item_idx] = pos
                new_used = used.copy()
                new_used[pos] = True
                solutions.extend(phase1_dfs(new_mapping, new_used, depth + 1, pbar))
        
        return solutions

    def phase2_dfs(
        mapping: np.ndarray,
        used: np.ndarray,
        depth: int,
        pbar: tqdm
    ) -> None:
        """DFS for Phase 2 (remaining items)."""
        nonlocal solutions, worst_top_n_score, processed_perms
        
        # Process complete solutions
        if depth == n_items_to_assign:
            processed_perms += 1
            if processed_perms % 1000 == 0:  # Only update every 1000 permutations
                pbar.update(1000)
            
            if n_constrained:
                if not validate_mapping(mapping, constrained_item_indices, constrained_positions):
                    return

            # Calculate score with cross-interactions
            total_score, item_component, item_pair_component = calculate_score(
                mapping,
                position_score_matrix,
                item_scores,
                item_pair_score_matrix,
                item_weight,
                item_pair_weight,
                cross_item_pair_matrix,  # Make sure these are passed
                cross_position_pair_matrix,
                items_assigned,
                positions_assigned
            )
            
            margin = total_score - (worst_top_n_score - np.abs(worst_top_n_score) * np.finfo(np.float32).eps)
            if len(solutions) < n_solutions or margin > 0:
                solution = (
                    total_score,
                    item_component,
                    item_pair_component,
                    mapping.tolist()
                )
                solutions.append(solution)
                solutions.sort(key=lambda x: x[0])  # Sort by total_score
                if len(solutions) > n_solutions:
                    solutions.pop(0)  # Remove worst solution
                worst_top_n_score = solutions[0][0]
            return
       
        # Find next unassigned item
        current_item_idx = get_next_item(mapping)
        if current_item_idx == -1: 
            return

        # Get valid positions for this item
        if n_constrained and current_item_idx in constrained_item_indices:
            valid_positions = [pos for pos in constrained_positions if not used[pos]]
        else:
            valid_positions = [pos for pos in range(n_positions_to_assign) if not used[pos]]
            
        if debug_print:
            print(f"\nPlacing item {items_to_assign[current_item_idx]} (idx {current_item_idx})")
            print("Current mapping: ", mapping)
            print(f"Valid positions: {valid_positions} ({[positions_to_assign[p] for p in valid_positions]})")

        # Try each valid position
        for pos in valid_positions:
            if debug_print:
                print(f"  Trying {items_to_assign[current_item_idx]} in position {positions_to_assign[pos]}")
            new_mapping = mapping.copy()
            new_mapping[current_item_idx] = pos
            new_used = used.copy()
            new_used[pos] = True
            
            # Only prune if we have n solutions to compare against
            if len(solutions) >= n_solutions:
                upper_bound = calculate_upper_bound(
                    new_mapping, new_used,
                    position_score_matrix, item_scores,
                    item_pair_score_matrix,
                    item_weight, item_pair_weight,
                    best_score=worst_top_n_score,
                    single_solution=False,
                    depth=depth + 1
                )
                
                margin = upper_bound - worst_top_n_score - np.abs(worst_top_n_score) * np.finfo(np.float32).eps
                if margin < 0:  # We can safely prune
                    #pruning_stats['depth'][depth] += 1
                    #pruning_stats['margin'].append(margin)
                    #pruning_stats['total_pruned'] += 1
                    continue
                #pruning_stats['total_explored'] += 1

            # Recursien:
            phase2_dfs(new_mapping, new_used, depth + 1, pbar)

    #-------------------------------------------------------------------------
    # Phase 1: Find all valid arrangements of constrained items
    #-------------------------------------------------------------------------
    if n_constrained:
        phase1_solutions = []
        with tqdm(total=total_perms_phase1, desc="Phase 1", unit='perms') as pbar:
            phase1_solutions = phase1_dfs(initial_mapping, initial_used, 0, pbar)      
        print(f"\nFound {len(phase1_solutions)} valid phase 1 arrangements")
    
    #-------------------------------------------------------------------------
    # Phase 2: For each Phase 1 solution, arrange remaining items
    #-------------------------------------------------------------------------
    current_phase1_solution_index = 0

    with tqdm(total=total_perms_phase2//1000, desc="Phase 2", unit='Kperms') as pbar:
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
    for score, unweighted_item_score, unweighted_item_pair_score, mapping_list in reversed(solutions):
        mapping = np.array(mapping_list, dtype=np.int32)
        item_mapping = dict(zip(items_to_assign, [positions_to_assign[i] for i in mapping]))
        
        # Here's where we need to recalculate the score for the complete layout
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
            complete_score, complete_unweighted_item_score, complete_unweighted_item_pair_score = (
                calculate_score(
                    complete_mapping_array,
                    complete_position_score_matrix, 
                    complete_item_scores,
                    complete_item_pair_score_matrix,
                    item_weight, 
                    item_pair_weight
                )
            )
            
            # Use the complete scores
            score = complete_score
            unweighted_item_score = complete_unweighted_item_score
            unweighted_item_pair_score = complete_unweighted_item_pair_score
        
        return_solutions.append((
            score,
            item_mapping,
            {'total': {
                'total_score': score,
                'unweighted_item_pair_score': unweighted_item_pair_score,
                'unweighted_item_score': unweighted_item_score
            }}
        ))

    return return_solutions, processed_perms

#-----------------------------------------------------------------------------
# Main function and pipeline
#-----------------------------------------------------------------------------
def optimize_layout(config: dict, verbose: bool = False) -> None:
    """
    Main optimization function. Uses specialized single-solution search when nlayouts=1,
    otherwise uses original branch_and_bound_optimal search.
    """
    start_time = time.time()
    # Get parameters from config
    items_to_assign = config['optimization']['items_to_assign']
    positions_to_assign = config['optimization']['positions_to_assign']
    items_to_constrain = config['optimization'].get('items_to_constrain', '')
    positions_to_constrain = config['optimization'].get('positions_to_constrain', '')
    items_assigned = config['optimization'].get('items_assigned', '')
    positions_assigned = config['optimization'].get('positions_assigned', '')
    n_layouts = config['optimization'].get('nlayouts', 5)
    # Get visualization settings
    print_keyboard = config['visualization'].get('print_keyboard', True)
    # Get scoring weights and missing values
    item_weight = config['optimization']['scoring']['item_weight']
    item_pair_weight = config['optimization']['scoring']['item_pair_weight']
    missing_item_pair_norm_score = config['optimization']['scoring'].get('missing_item_pair_norm_score', 1.0)
    missing_position_pair_norm_score = config['optimization']['scoring'].get('missing_position_pair_norm_score', 1.0)
    # Validate configuration
    validate_config(config)
    # Calculate exact search space size
    search_space = calculate_total_perms(
        n_items=len(items_to_assign),
        n_positions=len(positions_to_assign),
        items_to_constrain=set(items_to_constrain),
        positions_to_constrain=set(positions_to_constrain),
        items_assigned=set(items_assigned),
        positions_assigned=set(positions_assigned)
    )
    # Print detailed search space analysis
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
    # Load and normalize scores
    print("\nNormalization of scores:")
    norm_item_scores, norm_item_pair_scores, norm_position_scores, norm_position_pair_scores = load_scores(config)
    # Get scoring weights
    item_weight = config['optimization']['scoring']['item_weight']
    item_pair_weight = config['optimization']['scoring']['item_pair_weight']
    # Prepare arrays for optimization - include items_assigned and positions_assigned
    arrays = prepare_arrays(
        items_to_assign, positions_to_assign,
        norm_item_scores, norm_item_pair_scores, 
        norm_position_scores, norm_position_pair_scores,
        missing_item_pair_norm_score, missing_position_pair_norm_score,
        items_assigned, positions_assigned
    )
    weights = (item_weight, item_pair_weight)
    print(f"\nFinding top {n_layouts} solutions using branch and bound:")
    if items_to_constrain:
        print(f"  - {len(items_to_constrain)} constrained items: {items_to_constrain}")
        print(f"  - {len(positions_to_constrain)} constrained positions: {positions_to_constrain}")
    # Run multi-solution optimization
    results, processed_perms = branch_and_bound_optimal_nsolutions(
        arrays=arrays,
        weights=weights,
        config=config,
        n_solutions=n_layouts,
        norm_item_scores=norm_item_scores,
        norm_position_scores=norm_position_scores, 
        norm_item_pair_scores=norm_item_pair_scores,  
        norm_position_pair_scores=norm_position_pair_scores,
        missing_item_pair_norm_score=missing_item_pair_norm_score,
        missing_position_pair_norm_score=missing_position_pair_norm_score
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
        complete_score, complete_unweighted_item_score, complete_unweighted_item_pair_score = calculate_score(
            complete_mapping,
            complete_position_score_matrix, 
            complete_item_scores,
            complete_item_pair_score_matrix,
            item_weight, 
            item_pair_weight
        )
        
        # Update the score and detailed scores
        updated_detailed_scores = {'total': {
            'total_score': complete_score,
            'unweighted_item_pair_score': complete_unweighted_item_pair_score,
            'unweighted_item_score': complete_unweighted_item_score
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
        items_to_display=items_assigned,
        positions_to_display=positions_assigned,
        print_keyboard=print_keyboard,
        verbose=verbose
    )
    
    # Save the updated results
    save_results_to_csv(updated_results, config)
    
    # Final statistics reporting
    elapsed_time = time.time() - start_time
    if processed_perms >= search_space['total_perms']:
        print(f"\nTotal permutations processed: {processed_perms:,} (100% of solution space explored) in {timedelta(seconds=int(elapsed_time))}")
    else:
        percent_explored = (processed_perms / search_space['total_perms']) * 100
        print(f"Total permutations processed: {processed_perms:,} ({percent_explored:.1f}% of solution space explored) in {timedelta(seconds=int(elapsed_time))}")

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
        
        # Optimize the layout
        optimize_layout(config, verbose=args.verbose)
        
        elapsed = time.time() - start_time
        print(f"Total runtime: {timedelta(seconds=int(elapsed))}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
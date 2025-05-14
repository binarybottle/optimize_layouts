#!/usr/bin/env python3
"""
Find phase 2 keyboard layout candidates

This script evaluates whether a keyboard layout, after removing specific positions (;, O, A, W) 
and their corresponding letters, can achieve a target score when optimally arranged.

For each layout in the input CSV:
1. Parse the layout (16 letters in 16 positions)
2. Remove positions ;, O, A, W and their corresponding letters (leaving 12 letters in 12 positions)
3. Keep these 12 letters fixed in their positions
4. Evaluate if the 12 remaining letters (from the 24-letter alphabet, excluding q and z) can be 
   optimally arranged in the 12 open positions (including ;, O, A, W which were deliberately vacated)
   to achieve a score that equals or exceeds the target score
5. Output the results to a new CSV file with a "possible_improvement" column (1=yes, 0=no)

*The upper bound calculation provides a theoretical maximum score possible with the 
fixed letters and the optimal arrangement of the remaining letters.

Usage: 
    Basic usage:
    python find_phase2_candidates.py input.csv output.csv
    
    With target score:
    python find_phase2_candidates.py input.csv output.csv --target-score 0.05
    
    With custom item mapping:
    python find_phase2_candidates.py input.csv output.csv --items etaoinsrhldcumfp
    
    With debugging:
    python find_phase2_candidates.py input.csv output.csv --debug

Example:
    poetry run python3 find_phase2_candidates.py layout_scores.csv candidates_for_phase2.csv --target-score 0.05
"""

import sys
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from numba import jit

# Import functions from optimize_layout.py
from optimize_layout import (
    calculate_upper_bound,
    load_scores,
    prepare_arrays,
    calculate_score,
    load_config
)

def parse_layout_entry(layout_entry, items_str="etaoinsrhldcumfp"):
    """
    Parse a layout entry from the CSV file.
    
    Args:
        layout_entry: String with positions (e.g., "FDSEVJKILMUO;RWA")
        items_str: String with items in order (e.g., "etaoinsrhldcumfp")
        
    Returns:
        Dictionary mapping letters to positions
    """
    # Make sure layout_entry and items_str have the same length
    min_len = min(len(layout_entry), len(items_str))
    
    # If lengths don't match, log a warning
    if len(layout_entry) != len(items_str):
        print(f"Warning: Layout length ({len(layout_entry)}) doesn't match items length ({len(items_str)})")
        print(f"Using only the first {min_len} positions/items")
    
    # Create a dictionary with the available positions and items
    return {item: pos for item, pos in zip(items_str[:min_len], layout_entry[:min_len])}


def remove_positions_and_letters(layout_dict, positions_to_remove=["W", "A", ";", "O"]):
    """
    Remove specified positions and their corresponding letters from a layout.
    
    Args:
        layout_dict: Dictionary mapping letters to positions
        positions_to_remove: List of positions to remove (default: W, A, ;, O)
        
    Returns:
        Tuple of (modified layout dict, removed letters)
    """
    # Find letters that correspond to the positions to remove
    letters_to_remove = []
    for letter, position in layout_dict.items():
        if position in positions_to_remove:
            letters_to_remove.append(letter)
    
    # Create a new dictionary without the removed positions/letters
    new_layout = {k: v for k, v in layout_dict.items() 
                 if v not in positions_to_remove and k not in letters_to_remove}
    
    return new_layout, letters_to_remove

def can_improve_score(
    layout_dict, 
    removed_letters, 
    target_score, 
    config, 
    norm_item_scores, 
    norm_item_pair_scores, 
    norm_position_scores, 
    norm_position_pair_scores,
    debug=False
):
    """
    Check if it's possible to achieve a target score after removing specified positions and their letters.
    
    The function evaluates whether the kept letters plus the optimal arrangement of the remaining letters
    can achieve a score that equals or exceeds the target score.
    
    Args:
        layout_dict: Dictionary with current layout (after removal of specified positions)
        removed_letters: Letters that were assigned to removed positions
        target_score: Target score to beat
        config: Configuration dictionary
        norm_*: Normalized score dictionaries
        debug: Whether to print debug information
        
    Returns:
        Boolean indicating if improvement is possible
    """
    try:
        # Get all 24 letters (26 letters minus 'q' and 'z')
        all_letters = "abcdefghijklmnoprstuvwxy"  
        
        # Get all 24 positions on the keyboard
        all_positions = "QWERTYUIOPASDFGHJKL;ZXCVBNM,"
        
        # Find letters kept from the original layout
        kept_letters = set(layout_dict.keys())
        
        # Find all letters remaining to assign
        # These are letters that are:
        # 1. In the full alphabet (24 letters minus q,z)
        # 2. Not already in the kept layout
        # 3. Not already removed (because they were assigned to positions we removed)
        removed_set = set(removed_letters)
        letters_to_assign = [l for l in all_letters 
                          if l not in kept_letters and l not in removed_set]
        
        # Find all available positions 
        # These are positions that:
        # 1. Are in the full keyboard (24 positions)
        # 2. Not already used in the kept layout
        positions_to_assign = [p for p in all_positions 
                            if p not in set(layout_dict.values())]
        
        # Debug output to understand what's happening with positions and letters
        if debug:
            print(f"\nLetter Analysis:")
            print(f"  All 24 letters: {all_letters}")
            print(f"  Original layout letters: {sorted(layout_dict.keys())} ({len(layout_dict.keys())})")
            print(f"  Kept letters: {sorted(kept_letters)} ({len(kept_letters)})")
            print(f"  Removed letters: {sorted(removed_set)} ({len(removed_set)})")
            print(f"  Letters to assign: {sorted(letters_to_assign)} ({len(letters_to_assign)})")
            
            print(f"\nPosition Analysis:")
            print(f"  All 24 positions: {all_positions}")
            print(f"  Kept positions: {sorted(list(layout_dict.values()))} ({len(layout_dict.values())})")
            print(f"  Positions to assign: {sorted(positions_to_assign)} ({len(positions_to_assign)})")
            
            # Check for duplicates in the original layout
            letter_counts = {}
            for letter in layout_dict.keys():
                letter_counts[letter] = letter_counts.get(letter, 0) + 1
            duplicate_letters = {k: v for k, v in letter_counts.items() if v > 1}
            
            position_counts = {}
            for position in layout_dict.values():
                position_counts[position] = position_counts.get(position, 0) + 1
            duplicate_positions = {k: v for k, v in position_counts.items() if v > 1}
            
            if duplicate_letters:
                print(f"\nWarning: Found duplicate letters in layout: {duplicate_letters}")
            
            if duplicate_positions:
                print(f"\nWarning: Found duplicate positions in layout: {duplicate_positions}")
        
        # Update expectations based on the actual layout, not the theoretical calculation
        expected_positions_to_assign = 24 - len(set(layout_dict.values()))
        expected_letters_to_assign = 24 - len(kept_letters) - len(removed_set)
        
        # Verify we have the expected number of positions to assign
        if len(positions_to_assign) != expected_positions_to_assign:
            if debug:
                print(f"Warning: Expected {expected_positions_to_assign} positions to assign, but found {len(positions_to_assign)}")
        
        # Verify we have the expected number of letters to assign
        if len(letters_to_assign) != expected_letters_to_assign:
            if debug:
                print(f"Warning: Expected {expected_letters_to_assign} letters to assign, but found {len(letters_to_assign)}")
        
        # Set up optimization config
        test_config = config.copy()
        test_config['optimization']['items_to_assign'] = ''.join(letters_to_assign)
        test_config['optimization']['positions_to_assign'] = ''.join(positions_to_assign)
        
        # Add assigned items to the config
        test_config['optimization']['items_assigned'] = ''.join(kept_letters)
        test_config['optimization']['positions_assigned'] = ''.join(layout_dict.values())
        
        # Set scoring weights from config
        item_weight = config['optimization']['scoring']['item_weight']
        item_pair_weight = config['optimization']['scoring']['item_pair_weight']
        
        # Prepare arrays for optimization with the partial layout
        try:
            arrays = prepare_arrays(
                test_config['optimization']['items_to_assign'], 
                test_config['optimization']['positions_to_assign'],
                norm_item_scores, 
                norm_item_pair_scores, 
                norm_position_scores, 
                norm_position_pair_scores,
                items_assigned=list(layout_dict.keys()),
                positions_assigned=list(layout_dict.values())
            )
        except Exception as e:
            print(f"Error preparing arrays: {e}")
            if debug:
                print(f"items_to_assign: {test_config['optimization']['items_to_assign']}")
                print(f"positions_to_assign: {test_config['optimization']['positions_to_assign']}")
                print(f"items_assigned: {list(layout_dict.keys())}")
                print(f"positions_assigned: {list(layout_dict.values())}")
            return False
        
        # Initial mapping for unassigned items (-1 means not yet assigned)
        n_items_to_assign = len(test_config['optimization']['items_to_assign'])
        n_positions_to_assign = len(test_config['optimization']['positions_to_assign'])
        
        if n_items_to_assign == 0 or n_positions_to_assign == 0:
            if debug:
                print("No items to assign or no positions available")
            return False
            
        # Check if there's a mismatch between items and positions to assign
        if n_items_to_assign != n_positions_to_assign:
            if debug:
                print(f"Warning: Number of items to assign ({n_items_to_assign}) doesn't match number of positions to assign ({n_positions_to_assign})")
                # If there are more positions than items, we can still run the calculation
                if n_items_to_assign < n_positions_to_assign:
                    print(f"Proceeding with {n_items_to_assign} items and positions (ignoring {n_positions_to_assign - n_items_to_assign} extra positions)")
                    # Truncate positions_to_assign to match number of items
                    test_config['optimization']['positions_to_assign'] = test_config['optimization']['positions_to_assign'][:n_items_to_assign]
                    # Re-prepare arrays
                    arrays = prepare_arrays(
                        test_config['optimization']['items_to_assign'], 
                        test_config['optimization']['positions_to_assign'],
                        norm_item_scores, 
                        norm_item_pair_scores, 
                        norm_position_scores, 
                        norm_position_pair_scores,
                        items_assigned=list(layout_dict.keys()),
                        positions_assigned=list(layout_dict.values())
                    )
                    n_positions_to_assign = n_items_to_assign
            
        initial_mapping = np.full(n_items_to_assign, -1, dtype=np.int16)
        initial_used = np.zeros(n_positions_to_assign, dtype=np.bool_)
        
        # Get the arrays needed for upper bound calculation
        if len(arrays) > 3:
            item_scores, item_pair_score_matrix, position_score_matrix, *rest = arrays
        else:
            item_scores, item_pair_score_matrix, position_score_matrix = arrays
        
        # Calculate upper bound score
        try:
            upper_bound = calculate_upper_bound(
                initial_mapping, 
                initial_used, 
                position_score_matrix, 
                item_scores,
                item_pair_score_matrix, 
                item_weight, 
                item_pair_weight,
                best_score=float('-inf'),  # No pruning
                depth=0,
                cross_item_pair_matrix=arrays[3] if len(arrays) > 3 else None,
                cross_position_pair_matrix=arrays[4] if len(arrays) > 4 else None,
                items_assigned=list(layout_dict.keys()) if layout_dict else None,
                positions_assigned=list(layout_dict.values()) if layout_dict else None
            )
        except Exception as e:
            print(f"Error calculating upper bound: {e}")
            return False
        
        if debug:
            print(f"Upper bound score: {upper_bound}, Target score: {target_score}")
        
        # Check if upper bound score can exceed or equal the target score
        return upper_bound >= target_score
        
    except Exception as e:
        print(f"Error in can_improve_score: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_csv(input_csv, output_csv, config_path="find_phase2_candidates.yaml", debug=False, target_score=None, positions_to_remove=["W", "A", ";", "O"]):
    """
    Process the CSV file, check each layout, and output results.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        config_path: Path to configuration file
        debug: Whether to print debug information
        target_score: Target score to beat (if None, uses each layout's original score)
        positions_to_remove: List of positions to remove from the layout
    """
    # Load config
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return
    
    # Load scores
    try:
        norm_item_scores, norm_item_pair_scores, norm_position_scores, norm_position_pair_scores = load_scores(config)
    except Exception as e:
        print(f"Error loading scores: {e}")
        return
    
    # Check for missing score files in config
    required_files = [
        config['paths']['input'].get('item_scores_file'),
        config['paths']['input'].get('item_pair_scores_file'),
        config['paths']['input'].get('position_scores_file'),
        config['paths']['input'].get('position_pair_scores_file')
    ]
    
    missing_files = [f for f in required_files if not f]
    if missing_files:
        print(f"Warning: Missing file paths in config: {missing_files}")
    
    # Read input CSV
    try:
        layouts = []
        with open(input_csv, 'r') as f:
            reader = csv.reader(f)
            # Skip the header row
            headers = next(reader, None)
            # Get the column indices (in case columns are in different order)
            score_idx = 0
            layout_idx = 1
            if headers:
                for i, header in enumerate(headers):
                    if header and 'score' in header.lower():
                        score_idx = i
                    elif header and 'position' in header.lower():
                        layout_idx = i
                
                if debug:
                    print(f"Found headers: {headers}")
                    print(f"Using column {score_idx} for score and column {layout_idx} for layout")
            
            # Process data rows
            for row in reader:
                if len(row) > max(score_idx, layout_idx):  # Ensure we have both columns
                    try:
                        score = float(row[score_idx].strip())
                        layout_str = row[layout_idx].strip()
                        layouts.append((score, layout_str))
                    except ValueError:
                        print(f"Warning: Could not convert '{row[score_idx]}' to float, skipping row")
                        if debug:
                            print(f"Row contents: {row}")
    except Exception as e:
        print(f"Error reading input CSV {input_csv}: {e}")
        return
    
    # Check if we have any layouts to process
    if not layouts:
        print(f"No valid layouts found in {input_csv}")
        return
    
    print(f"Processing {len(layouts)} layouts...")
    
    # Item string to use for mapping - default is "etaoinsrhldcumfp"
    items_str = "etaoinsrhldcumfp"
    
    # Process each layout
    results = []
    for score, layout_str in tqdm(layouts):
        try:
            # Check that layout string is long enough
            if len(layout_str) < len(items_str):
                print(f"Warning: Layout {layout_str} is too short for the items string {items_str}")
                results.append((score, layout_str, 0))  # Mark as not improvable by default
                continue
                
            # Parse the layout
            layout_dict = parse_layout_entry(layout_str, items_str)
            
            # Remove specified positions and their corresponding letters
            modified_layout, removed_letters = remove_positions_and_letters(layout_dict, positions_to_remove)
            
            if debug:
                print(f"\nProcessing layout: {layout_str} with score {score}")
                print(f"Modified layout: {modified_layout}")
                print(f"Removed letters: {removed_letters}")
            
            # Check if improvement is possible
            # If target_score is provided, use it instead of the layout's original score
            score_to_beat = target_score if target_score is not None else score
            
            improvement_possible = can_improve_score(
                modified_layout,
                removed_letters,
                score_to_beat,
                config, 
                norm_item_scores, 
                norm_item_pair_scores, 
                norm_position_scores, 
                norm_position_pair_scores,
                debug=debug
            )
            
            # Add result
            results.append((score, layout_str, 1 if improvement_possible else 0))
            
            if debug:
                print(f"Improvement possible: {improvement_possible}")
                
        except Exception as e:
            print(f"Error processing layout {layout_str}: {e}")
            # Add to results as "no improvement possible" to be safe
            results.append((score, layout_str, 0))
            import traceback
            traceback.print_exc()
    
    # Write output CSV
    try:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["total_score", "positions_for_items", "possible_improvement"])
            for score, layout, possible in results:
                writer.writerow([score, layout, possible])
        
        print(f"Results written to {output_csv}")
    except Exception as e:
        print(f"Error writing output CSV {output_csv}: {e}")
        return
    
    # Summary stats
    possible_count = sum(1 for _, _, p in results if p == 1)
    
    if target_score is not None:
        print(f"Summary: {possible_count} out of {len(results)} layouts can potentially beat the target score of {target_score}.")
    else:
        print(f"Summary: {possible_count} out of {len(results)} layouts can potentially be improved from their original scores.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze keyboard layouts')
    parser.add_argument('input_csv', help='Input CSV file with layouts')
    parser.add_argument('output_csv', help='Output CSV file for results')
    parser.add_argument('--config', default='find_phase2_candidates.yaml', help='Configuration file path')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    parser.add_argument('--items', default='etaoinsrhldcumfp', help='String of letters to map to positions')
    parser.add_argument('--target-score', type=float, help='Target score to beat (if not provided, uses each layout\'s own score)')
    parser.add_argument('--positions-to-remove', default='WA;O', help='Positions to remove from layout (default: W,A,;,O)')
    
    args = parser.parse_args()
    
    # Update the global items_str if provided
    items_str = args.items
    
    # Get positions to remove
    positions_to_remove = list(args.positions_to_remove)
    
    # Update the process_csv function to pass positions_to_remove
    process_csv(args.input_csv, args.output_csv, args.config, args.debug, args.target_score, positions_to_remove)
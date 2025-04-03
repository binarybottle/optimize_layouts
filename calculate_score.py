#!/usr/bin/env python
"""
Layout Score Calculator

This script takes a list of items and positions, then calculates the score
using the same methods as optimize_layout.py.

Example usage:
    python calculate_score.py --items "etaoinsrhld" --positions "FJDSVERAWCQ" --no-details
"""
import argparse
import numpy as np
import yaml
from optimize_layout import (
    load_config, 
    load_and_normalize_scores, 
    prepare_arrays, 
    calculate_score, 
    validate_config,
    visualize_keyboard_layout
)

def split_mapping(items_str, positions_str):
    """Split input strings into lists ensuring equal length."""
    items = list(items_str.lower())
    positions = list(positions_str.upper())
    
    if len(items) != len(positions):
        raise ValueError(f"Mismatch between items ({len(items)}) and positions ({len(positions)})")
    
    return items, positions

def calculate_layout_score(items_str, positions_str, config_path="config.yaml", detailed=True):
    """
    Calculate the score for a given layout mapping.
    
    Args:
        items_str: String of items (e.g., "etaoinsrhld")
        positions_str: String of positions (e.g., "FJDSVERAWCQ")
        config_path: Path to the configuration file
        detailed: Whether to print detailed scoring information
    
    Returns:
        Tuple of (total_score, unweighted_item_score, unweighted_item_pair_score)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Extract the items and positions
    items, positions = split_mapping(items_str, positions_str)
    
    # Save original config settings
    original_items_to_assign = config['optimization'].get('items_to_assign', '')
    original_positions_to_assign = config['optimization'].get('positions_to_assign', '')
    original_items_assigned = config['optimization'].get('items_assigned', '')
    original_positions_assigned = config['optimization'].get('positions_assigned', '')
    
    # Update config to include all items/positions in the input
    config['optimization']['items_to_assign'] = ''.join(items)
    config['optimization']['positions_to_assign'] = ''.join(positions)
    config['optimization']['items_assigned'] = ''
    config['optimization']['positions_assigned'] = ''
    
    # Validate the modified config
    validate_config(config)
    
    # Load and normalize scores
    if detailed:
        print("\nLoading and normalizing scores...")
    
    norm_item_scores, norm_item_pair_scores, norm_position_scores, norm_position_pair_scores = (
        load_and_normalize_scores(config)
    )
    
    # Extract scoring settings
    item_weight = config['optimization']['scoring']['item_weight']
    item_pair_weight = config['optimization']['scoring']['item_pair_weight']
    missing_item_pair_norm_score = config['optimization']['scoring']['missing_item_pair_norm_score']
    missing_position_pair_norm_score = config['optimization']['scoring']['missing_position_pair_norm_score']
    
    # Prepare arrays for scoring
    arrays = prepare_arrays(
        config['optimization']['items_to_assign'],
        config['optimization']['positions_to_assign'],
        norm_item_scores, 
        norm_item_pair_scores, 
        norm_position_scores, 
        norm_position_pair_scores,
        missing_item_pair_norm_score,
        missing_position_pair_norm_score
    )
    item_scores, item_pair_score_matrix, position_score_matrix = arrays
    
    # Create the mapping array
    n_items = len(items)
    mapping = np.arange(n_items, dtype=np.int32)  # Direct mapping since we reordered the arrays
    
    # Calculate score
    total_score, unweighted_item_score, unweighted_item_pair_score = calculate_score(
        mapping, 
        position_score_matrix,
        item_scores, 
        item_pair_score_matrix,
        item_weight, 
        item_pair_weight
    )
    
    # Display detailed scoring information if requested
    if detailed:
        print("\nDetailed Scoring Information:")
        print(f"Items:     {items_str}")
        print(f"Positions: {positions_str}")
        print("\nScoring weights:")
        print(f"  Item weight:      {item_weight:.2f}")
        print(f"  Item-pair weight: {item_pair_weight:.2f}")
        
        print("\nScores:")
        print(f"  Unweighted item score:      {unweighted_item_score:.12f}")
        print(f"  Unweighted item-pair score: {unweighted_item_pair_score:.12f}")
        print(f"  Total weighted score:       {total_score:.12f}")
        
        # Show individual item and position scores
        print("\nIndividual Item Scores:")
        print("  Item | Position | Item Score | Position Score | Combined")
        print("  " + "-" * 55)
        for i, (item, position) in enumerate(zip(items, positions)):
            item_score = item_scores[i]
            position_score = position_score_matrix[i, i]
            combined = item_score * position_score
            print(f"  {item:4} | {position:8} | {item_score:10.6f} | {position_score:14.6f} | {combined:8.6f}")
        
        # Show item pair contributions
        print("\nTop Item Pair Contributions:")
        
        # Calculate all pair scores
        pair_scores = []
        for i in range(n_items):
            for j in range(i+1, n_items):
                item1, item2 = items[i], items[j]
                pos1, pos2 = positions[i], positions[j]
                
                # Forward direction (i->j)
                fwd_item_score = item_pair_score_matrix[i, j]
                fwd_pos_score = position_score_matrix[i, j]
                fwd_combined = fwd_item_score * fwd_pos_score
                
                # Backward direction (j->i)
                bck_item_score = item_pair_score_matrix[j, i]
                bck_pos_score = position_score_matrix[j, i]
                bck_combined = bck_item_score * bck_pos_score
                
                # Average of both directions
                avg_combined = (fwd_combined + bck_combined) / 2
                
                pair_scores.append((
                    (item1, item2), 
                    (pos1, pos2), 
                    avg_combined,
                    fwd_item_score,
                    fwd_pos_score,
                    bck_item_score,
                    bck_pos_score
                ))
        
        # Sort by contribution (highest first)
        pair_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Print top 10 pairs
        num_to_show = min(10, len(pair_scores))
        print(f"  Items | Positions | Avg Score | Fwd Item | Fwd Pos | Bck Item | Bck Pos")
        print("  " + "-" * 75)
        for i in range(num_to_show):
            (item1, item2), (pos1, pos2), avg, fwd_item, fwd_pos, bck_item, bck_pos = pair_scores[i]
            print(f"  {item1}{item2:3} | {pos1}{pos2:7} | {avg:9.6f} | {fwd_item:8.6f} | {fwd_pos:7.6f} | {bck_item:8.6f} | {bck_pos:7.6f}")
        
        # Display as keyboard layout
        print("\nKeyboard Layout Visualization:")
        item_mapping = dict(zip(items, positions))
        visualize_keyboard_layout(
            mapping=item_mapping,
            title="Scoring Layout",
            config=config
        )
        
        # Restore original config
        config['optimization']['items_to_assign'] = original_items_to_assign
        config['optimization']['positions_to_assign'] = original_positions_to_assign
        config['optimization']['items_assigned'] = original_items_assigned
        config['optimization']['positions_assigned'] = original_positions_assigned
    
    return total_score, unweighted_item_score, unweighted_item_pair_score

def main():
    parser = argparse.ArgumentParser(description='Calculate layout scoring.')
    parser.add_argument('--items', type=str, required=True, help='String of items (e.g., "etaoinsrhld")')
    parser.add_argument('--positions', type=str, required=True, help='String of positions (e.g., "FJDSVERAWCQ")')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--no-details', action='store_true', help='Suppress detailed output')
    
    args = parser.parse_args()
    calculate_layout_score(args.items, args.positions, args.config, not args.no_details)

if __name__ == "__main__":
    main()
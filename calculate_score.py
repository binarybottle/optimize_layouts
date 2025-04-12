#!/usr/bin/env python
"""
Layout Score Calculator

This script takes a list of items and positions, then calculates the score
using the same methods as optimize_layout.py.

Example usage:
    python calculate_score.py --items "etaoinsrhld" --positions "FJDSVERAWCQ" --details
"""
import argparse
import numpy as np
import yaml
from optimize_layout import (
    load_config, 
    load_scores, 
    calculate_score, 
    visualize_keyboard_layout
)

def split_mapping(items_str, positions_str):
    """Split input strings into lists ensuring equal length."""
    items = list(items_str.lower())
    positions = list(positions_str.upper())
    
    if len(items) != len(positions):
        raise ValueError(f"Mismatch between items ({len(items)}) and positions ({len(positions)})")
    
    return items, positions

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
    
    # Verify all scores are normalized [0,1]
    arrays_to_check = [
        (item_scores, "Item scores"),
        (item_pair_score_matrix, "Item pair scores"),
        (position_score_matrix, "Position scores")
    ]
    
    for arr, name in arrays_to_check:
        if not np.all(np.isfinite(arr)):
            print(f"Warning: {name} contains non-finite values")
        if np.any(arr < 0) or np.any(arr > 1):
            print(f"Warning: {name} values outside [0,1] range: min={np.min(arr)}, max={np.max(arr)}")
    
    # Create direct mapping for evaluating a complete layout
    mapping = np.arange(n_items, dtype=np.int16)
    
    return mapping, item_scores, item_pair_score_matrix, position_score_matrix

def calculate_layout_score(items_str, positions_str, config, detailed=True):
    """
    Calculate the score for a given layout mapping.
    
    Args:
        items_str: String of items (e.g., "etaoinsrhld")
        positions_str: String of positions (e.g., "FJDSVERAWCQ")
        config: configuration file
        detailed: Whether to print detailed scoring information
    
    Returns:
        Tuple of (total_score, unweighted_item_score, unweighted_item_pair_score)
    """
    # Extract the items and positions
    items, positions = split_mapping(items_str, positions_str)
    
    # Create layout mapping for later visualization
    layout_mapping = dict(zip(items, positions))
    
    # Load and normalize scores
    if detailed:
        print("\nLoading normalized scores...")
    
    norm_item_scores, norm_item_pair_scores, norm_position_scores, norm_position_pair_scores = (
        load_scores(config)
    )
    
    # Extract scoring settings
    item_weight = config['optimization']['scoring']['item_weight']
    item_pair_weight = config['optimization']['scoring']['item_pair_weight']
    missing_item_pair_norm_score = config['optimization']['scoring'].get('missing_item_pair_norm_score', 1.0)
    missing_position_pair_norm_score = config['optimization']['scoring'].get('missing_position_pair_norm_score', 1.0)
    
    # Prepare arrays for the complete layout
    mapping, item_scores, item_pair_score_matrix, position_score_matrix = prepare_complete_arrays(
        items, positions, 
        norm_item_scores, norm_item_pair_scores,
        norm_position_scores, norm_position_pair_scores,
        missing_item_pair_norm_score, missing_position_pair_norm_score
    )
    
    # Calculate scores
    total_score, unweighted_item_score, unweighted_item_pair_score = calculate_score(
        mapping,
        position_score_matrix, 
        item_scores,
        item_pair_score_matrix,
        item_weight, 
        item_pair_weight
    )
    
    if detailed:
        print("\nDetailed Scoring Information:")
        print(f"Items:     {''.join(items)}")
        print(f"Positions: {''.join(positions)}")
        print("Scoring weights:")
        print(f"  Item weight:      {item_weight:.2f}")
        print(f"  Item-pair weight: {item_pair_weight:.2f}")
        print("Scores:")
        print(f"  Unweighted item score:      {unweighted_item_score:.12f}")
        print(f"  Unweighted item-pair score: {unweighted_item_pair_score:.12f}")
        print(f"  Total weighted score:       {total_score:.12f}")
        
        # Print individual item scores
        print("\nIndividual Item Scores:")
        print("  Item | Position | Item Score | Position Score | Combined")
        print("  " + "-" * 55)
        
        print("\nLayout:")
        print(f"  items: {''.join(items)}")
        print(f"  positions: {''.join(positions).upper()}")

        # Create a list of item details for sorting
        item_details = []
        for i, (item, pos) in enumerate(zip(items, positions)):
            item_score = item_scores[i]
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
        
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j: continue
                
                item_i = items[i]
                item_j = items[j]
                pos_i = positions[i]
                pos_j = positions[j]
                
                # Forward direction (item_i->item_j)
                fwd_item_score = norm_item_pair_scores.get((item_i.lower(), item_j.lower()), missing_item_pair_norm_score)
                fwd_pos_score = norm_position_pair_scores.get((pos_i.lower(), pos_j.lower()), 
                                                             missing_position_pair_norm_score)
                
                # Backward direction (item_j->item_i)
                bck_item_score = norm_item_pair_scores.get((item_j.lower(), item_i.lower()), missing_item_pair_norm_score)
                bck_pos_score = norm_position_pair_scores.get((pos_j.lower(), pos_i.lower()), 
                                                             missing_position_pair_norm_score)
                
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
        
        # Display as keyboard layout
        print("\nKeyboard Layout Visualization:")
        
    # Save original config values
    original_items_assigned = config['optimization'].get('items_assigned', '')
    original_positions_assigned = config['optimization'].get('positions_assigned', '')

    # Temporarily clear config assigned values to prevent double display
    config['optimization']['items_assigned'] = ''
    config['optimization']['positions_assigned'] = ''

    # Now visualize
    visualize_keyboard_layout(
        mapping=layout_mapping,
        title="Scoring Layout",
        config=config
    )

    # Restore original values
    config['optimization']['items_assigned'] = original_items_assigned
    config['optimization']['positions_assigned'] = original_positions_assigned
    
    return total_score, unweighted_item_score, unweighted_item_pair_score

def main():
    parser = argparse.ArgumentParser(description="Calculate layout score for a given mapping.")
    parser.add_argument("--items", required=True, help="String of items (e.g., 'etaoinsrhld')")
    parser.add_argument("--positions", required=True, help="String of positions (e.g., 'FJDSVERAWCQ')")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--details", action="store_true", help="Show details")
    
    args = parser.parse_args()
    
    # Load config to display
    config = load_config(args.config)
    
    # Display current evaluation information
    print("\nEvaluating layout:")
    print(f"{len(args.items)} items: {args.items}")
    print(f"{len(args.positions)} positions: {args.positions}")
    
    # Calculate score
    total_score, item_score, pair_score = calculate_layout_score(
        args.items,
        args.positions,
        config,
        detailed=args.details
    )
    
    if not args.details:
        print(f"\nTotal score: {total_score:.12f}")
        print(f"Item score: {item_score:.12f}")
        print(f"Item-pair score: {pair_score:.12f}")

if __name__ == "__main__":
    main()
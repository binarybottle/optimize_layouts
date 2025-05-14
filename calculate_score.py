#!/usr/bin/env python
"""
Layout Score Calculator (Correct optimize_layout.py Version)

This script calculates layout scores using the same method as optimize_layout.py,
which applies scaling factors to item and pair scores.

Example usage:
    python calculate_score.py --items "ecumfponsrithlabdgjkqvwx" --positions "FDESVRWJKILMUO;,P/ZCXA.Q" --details
"""
import argparse
import yaml
import os
from optimize_layout import (
    load_scores, 
    calculate_score,
    visualize_keyboard_layout,
    prepare_complete_arrays
)

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from yaml file without printing details."""
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
    
    return config

def split_mapping(items_str, positions_str):
    """Split input strings into lists ensuring equal length."""
    items = list(items_str.lower())
    positions = list(positions_str.upper())
    
    if len(items) != len(positions):
        raise ValueError(f"Mismatch between items ({len(items)}) and positions ({len(positions)})")
    
    return items, positions

def calculate_layout_score(items_str, positions_str, config, detailed=True,
                           missing_item_pair_norm_score=1.0,
                           missing_position_pair_norm_score=1.0):
    """
    Calculate the score for a given layout using optimize_layout.py's method.
    
    Args:
        items_str: String of items (e.g., "etaoinsrhld")
        positions_str: String of positions (e.g., "FJDSVERAWCQ")
        config: configuration file
        detailed: Whether to print detailed scoring information
    
    Returns:
        Tuple of (total_score, item_score, item_pair_score)
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
    
    # Print only the relevant scoring parameters
    if detailed:
        print("\nScoring Configuration:")
        print(f"  Missing item pair score:     {missing_item_pair_norm_score:.6f}")
        print(f"  Missing position pair score: {missing_position_pair_norm_score:.6f}")
        print(f"\nScoring {len(items)} items: {''.join(items)}")
        print(f"In {len(positions)} positions: {''.join(positions)}")
    
    # Prepare arrays for scoring
    mapping, item_scores, item_pair_score_matrix, position_score_matrix = prepare_complete_arrays(
        items, positions, 
        norm_item_scores, norm_item_pair_scores,
        norm_position_scores, norm_position_pair_scores,
        missing_item_pair_norm_score, missing_position_pair_norm_score
    )
    
    # Calculate raw scores using calculate_score
    total_score, item_score, item_pair_score = calculate_score(
        mapping,
        position_score_matrix, 
        item_scores,
        item_pair_score_matrix
    )
    
    if detailed:
        print("\nScoring Calculation:")
        print(f"  Item score:      {item_score:.12f}")
        print(f"  Item pair score: {item_pair_score:.12f}")
        print(f"  Total score:     {total_score:.12f}")
        
        # Print individual item scores if detailed output is requested
        if detailed:
            print("\nIndividual Item Scores:")
            print("  Item | Position | Item Score | Position Score | Combined")
            print("  " + "-" * 55)
            
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
    
    # Return the correct score values
    return total_score, item_score, item_pair_score

def main():
    parser = argparse.ArgumentParser(description="Calculate layout score using optimize_layout.py's method.")
    parser.add_argument("--items", required=True, help="String of items (e.g., 'etaoinsrhld')")
    parser.add_argument("--positions", required=True, help="String of positions (e.g., 'FJDSVERAWCQ')")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--details", action="store_true", help="Show details")
    
    args = parser.parse_args()
    
    # Load config 
    config = load_config(args.config)
    
    # Display current evaluation information
    print("\nEvaluating layout:")
    print(f"{len(args.items)} items: {args.items}")
    print(f"{len(args.positions)} positions: {args.positions}")
    
    # Calculate score using optimizer's method
    total_score, item_score, item_pair_score = calculate_layout_score(
        args.items,
        args.positions,
        config,
        detailed=args.details,
        missing_item_pair_norm_score=1.0,
        missing_position_pair_norm_score=1.0
    )
    
    # Always show the final scores
    print(f"\nFinal Scores (Optimizer Method):")
    print(f"Total score:              {total_score:.12f}")
    print(f"Item score:               {item_score:.12f}")
    print(f"Item-pair score:          {item_pair_score:.12f}")
    
    # Show the CSV output format that matches optimize_layout.py's output
    print(f"\nCSV Values:")
    print(f"total_score item_score item_pair_score")
    print(f"{total_score:.6f} {item_score:.6f} {item_pair_score:.6f}")

if __name__ == "__main__":
    main()
# display_and_score_layout.py
"""
Layout score calculator that shows complete layout scores.

Example usage:
    python display_and_score_layout.py --items "etaoinsrhld" --positions "FJDSVERAWCQ" --details
    python display_and_score_layout.py --items "abc" --positions "FDJ" --config config.yaml --verbose
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Import consolidated modules
from config import load_config, Config
from scoring import LayoutScorer, prepare_scoring_arrays
from display import visualize_keyboard_layout
from validation import validate_specific_layout

#-----------------------------------------------------------------------------
# Score loading functions
#-----------------------------------------------------------------------------
def load_normalized_scores(config: Config) -> tuple:
    """
    Load normalized scores from CSV files.
    
    Args:
        config: Configuration object containing file paths
        
    Returns:
        Tuple of (item_scores, item_pair_scores, position_scores, position_pair_scores)
    """
    def load_score_dict(filepath: str, key_col: str, score_col: str = 'score') -> dict:
        """Helper to load score dictionary from CSV."""
        df = pd.read_csv(filepath)
        return {row[key_col].lower(): float(row[score_col]) for _, row in df.iterrows()}
    
    def load_pair_score_dict(filepath: str, pair_col: str, score_col: str = 'score') -> dict:
        """Helper to load pair score dictionary from CSV."""
        df = pd.read_csv(filepath)
        result = {}
        for _, row in df.iterrows():
            pair_str = str(row[pair_col])
            if len(pair_str) == 2:
                key = (pair_str[0].lower(), pair_str[1].lower())
                result[key] = float(row[score_col])
        return result
    
    # Load all score dictionaries
    item_scores = load_score_dict(config.paths.item_scores_file, 'item')
    item_pair_scores = load_pair_score_dict(config.paths.item_pair_scores_file, 'item_pair')
    position_scores = load_score_dict(config.paths.position_scores_file, 'position')  
    position_pair_scores = load_pair_score_dict(config.paths.position_pair_scores_file, 'position_pair')
    
    return item_scores, item_pair_scores, position_scores, position_pair_scores

#-----------------------------------------------------------------------------
# Complete layout scoring
#-----------------------------------------------------------------------------
def create_complete_layout_mapping(items_str: str, positions_str: str, config: Config) -> dict:
    """Create complete layout mapping including pre-assigned items."""
    mapping = {}
    
    # Add pre-assigned items
    if config.optimization.items_assigned and config.optimization.positions_assigned:
        mapping.update(dict(zip(config.optimization.items_assigned, 
                               config.optimization.positions_assigned.upper())))
    
    # Add optimized items
    mapping.update(dict(zip(items_str.lower(), positions_str.upper())))
    
    return mapping

def calculate_complete_layout_score(complete_mapping: dict, config: Config,
                                   score_dicts: tuple) -> tuple:
    """
    Calculate the complete layout score including all items and pairs.
    
    Args:
        complete_mapping: Complete item->position mapping  
        config: Configuration object
        score_dicts: Tuple of score dictionaries
        
    Returns:
        Tuple of (total_score, item_score, pair_score, cross_score)
    """
    # Get all items and positions from the complete mapping
    all_items = list(complete_mapping.keys())
    all_positions = list(complete_mapping.values())
    
    # Create arrays for the complete layout (no pre-assigned items since this IS the complete layout)
    complete_arrays = prepare_scoring_arrays(
        items_to_assign=all_items,
        positions_to_assign=all_positions,
        norm_item_scores=score_dicts[0],
        norm_item_pair_scores=score_dicts[1], 
        norm_position_scores=score_dicts[2],
        norm_position_pair_scores=score_dicts[3],
        items_assigned=None,  # No pre-assigned since this is the complete layout
        positions_assigned=None
    )
    
    complete_scorer = LayoutScorer(complete_arrays, mode='combined')
    
    # Build position lookup
    pos_to_idx = {pos: idx for idx, pos in enumerate(all_positions)}
    
    # Create mapping array
    mapping_array = np.array([pos_to_idx[complete_mapping[item]] 
                             for item in all_items], dtype=np.int32)
    
    # Score the complete layout
    return complete_scorer.score_layout(mapping_array, return_components=True)

#-----------------------------------------------------------------------------
# Score calculation functions
#-----------------------------------------------------------------------------
def calculate_layout_score(items_str: str, positions_str: str, config: Config, 
                         detailed: bool = True, mode: str = 'combined') -> tuple:
    """
    Calculate complete layout score including all items and pairs.
    
    Args:
        items_str: String of items (e.g., "etaoinsrhld")
        positions_str: String of positions (e.g., "FJDSVERAWCQ")
        config: Configuration object
        detailed: Whether to print detailed information
        mode: Scoring mode ('combined', 'item_only', 'pair_only', 'multi_objective')
    
    Returns:
        Tuple of (total_score, item_score, pair_score, cross_score)
    """
    # Validate input lengths
    items = list(items_str.lower())
    positions = list(positions_str.upper())
    
    if len(items) != len(positions):
        raise ValueError(f"Mismatch between items ({len(items)}) and positions ({len(positions)})")
    
    if detailed:
        print(f"\nCalculating COMPLETE layout score:")
        print(f"  Items ({len(items)}): {''.join(items)}")
        print(f"  Positions ({len(positions)}): {''.join(positions)}")
        print(f"  Scoring mode: {mode}")
    
    # Load normalized scores
    if detailed:
        print("\nLoading normalized scores...")
    
    score_dicts = load_normalized_scores(config)
    
    # Create complete layout mapping
    complete_mapping = create_complete_layout_mapping(items_str, positions_str, config)
    
    if detailed:
        all_items = ''.join(complete_mapping.keys())
        all_positions = ''.join(complete_mapping.values())
        print(f"  Complete layout: {all_items} → {all_positions}")
    
    # Calculate complete layout score
    total_score, item_score, pair_score, cross_score = calculate_complete_layout_score(
        complete_mapping, config, score_dicts)
    
    if detailed:
        print(f"\nComplete Layout Results:")
        print(f"  Total score:      {total_score:.12f}")
        print(f"  Item component:   {item_score:.12f}")
        print(f"  Pair component:   {pair_score:.12f}")
        print(f"  Cross component:  {cross_score:.12f}")
    
    # Create a dummy scorer for backward compatibility
    # (This is a temporary solution for functions that expect a scorer)
    try:
        arrays = prepare_scoring_arrays(
            items_to_assign=list(complete_mapping.keys()),
            positions_to_assign=list(complete_mapping.values()),
            norm_item_scores=score_dicts[0],
            norm_item_pair_scores=score_dicts[1],
            norm_position_scores=score_dicts[2],
            norm_position_pair_scores=score_dicts[3]
        )
        scorer = LayoutScorer(arrays, mode=mode)
    except Exception:
        scorer = None
    
    return total_score, item_score, pair_score, cross_score, scorer

def print_detailed_breakdown(items_str: str, positions_str: str, config: Config, scorer: LayoutScorer):
    """Print detailed item-by-item scoring breakdown."""
    items = list(items_str.lower())
    positions = list(positions_str.upper())
    
    print(f"\nDetailed Item Breakdown:")
    print(f"  {'Item':<4} | {'Pos':<3} | {'Item Score':<10} | {'Pos Score':<10} | {'Combined':<10}")
    print(f"  {'-'*4}-+-{'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    
    # Get individual scores (simplified for complete layout)
    if scorer and hasattr(scorer, 'arrays'):
        item_details = []
        
        for i, (item, pos) in enumerate(zip(items, positions)):
            if i < len(scorer.arrays.item_scores):
                item_score = scorer.arrays.item_scores[i]
                pos_score = scorer.arrays.position_matrix[i, i] if i < scorer.arrays.position_matrix.shape[0] else 0.0
                combined = item_score * pos_score
                item_details.append((item, pos, item_score, pos_score, combined))
        
        # Sort by combined score (highest first) for better display
        item_details.sort(key=lambda x: x[4], reverse=True)
        
        for item, pos, item_score, pos_score, combined in item_details:
            print(f"  {item:<4} | {pos:<3} | {item_score:<10.6f} | {pos_score:<10.6f} | {combined:<10.6f}")
    else:
        print("  Detailed breakdown not available with current scorer")

#-----------------------------------------------------------------------------
# Main function and CLI
#-----------------------------------------------------------------------------
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate COMPLETE layout score including all items and pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic complete scoring
  python display_and_score_layout.py --items "abc" --positions "FDJ"
  
  # With detailed breakdown
  python display_and_score_layout.py --items "etaoinsrhld" --positions "FJDSVERAWCQ" --details
  
  # Different scoring modes
  python display_and_score_layout.py --items "abc" --positions "FDJ" --mode item_only
  python display_and_score_layout.py --items "abc" --positions "FDJ" --mode multi_objective
  
  # With validation
  python display_and_score_layout.py --items "abc" --positions "FDJ" --validate
        """
    )
    
    parser.add_argument("--items", required=True, 
                       help="String of items (e.g., 'etaoinsrhld')")
    parser.add_argument("--positions", required=True,
                       help="String of positions (e.g., 'FJDSVERAWCQ')")
    parser.add_argument("--config", default="config.yaml",
                       help="Path to config file (default: config.yaml)")
    parser.add_argument("--mode", choices=['combined', 'item_only', 'pair_only', 'multi_objective'],
                       default='combined', help="Scoring mode (default: combined)")
    parser.add_argument("--details", action="store_true",
                       help="Show detailed scoring breakdown")
    parser.add_argument("--validate", action="store_true", 
                       help="Run validation on this specific layout")
    parser.add_argument("--keyboard", action="store_true",
                       help="Show keyboard visualization")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        print("Complete Layout Score Calculator")
        print("=" * 50)
        
        # Validate inputs
        if len(args.items) != len(args.positions):
            print(f"Error: Item count ({len(args.items)}) != Position count ({len(args.positions)})")
            return
        
        # Run validation if requested
        if args.validate:
            print("\nRunning layout validation...")
            validation_result = validate_specific_layout(args.items, args.positions, config)
            print(f"  {validation_result}")
            print()
        
        # Calculate scores
        total_score, item_score, pair_score, cross_score, scorer = calculate_layout_score(
            args.items, args.positions, config, args.details, args.mode
        )
        
        # Show detailed breakdown if requested
        if args.details and scorer:
            print_detailed_breakdown(args.items, args.positions, config, scorer)
        
        # Show keyboard layout if requested
        if args.keyboard:
            complete_mapping = create_complete_layout_mapping(args.items, args.positions, config)
            print("\nKeyboard Layout:")
            visualize_keyboard_layout(
                mapping=complete_mapping,
                title="Complete Layout",
                config=config
            )
        
        # Final summary
        print(f"\nFinal Complete Layout Scores ({args.mode} mode):")
        print(f"  Total score:       {total_score:.12f}")
        print(f"  Item component:    {item_score:.12f}")
        print(f"  Pair component:    {pair_score:.12f}")
        print(f"  Cross component:   {cross_score:.12f}")
        
        # CSV format output (for compatibility with existing tools)
        print(f"\nCSV Format:")
        print(f"total_score,item_score,pair_score,cross_score")
        print(f"{total_score:.6f},{item_score:.6f},{pair_score:.6f},{cross_score:.6f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the configuration file and input files exist.")
    except ValueError as e:
        print(f"Input Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
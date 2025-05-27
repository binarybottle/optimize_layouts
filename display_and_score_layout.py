# calculate_score.py
"""
Layout Score Calculator using the consolidated scoring system.

This completely rewritten version uses the new unified scoring architecture
for consistent, maintainable score calculations.

Example usage:
    python calculate_score.py --items "etaoinsrhld" --positions "FJDSVERAWCQ" --details
    python calculate_score.py --items "abc" --positions "FDJ" --config config.yaml --verbose
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
# Score Calculation Functions
#-----------------------------------------------------------------------------

def load_normalized_scores_simple(config: Config) -> tuple:
    """
    Load normalized scores from CSV files.
    
    Returns:
        Tuple of (item_scores, item_pair_scores, position_scores, position_pair_scores)
    """
    def load_score_dict(filepath: str, key_col: str, score_col: str = 'score') -> dict:
        df = pd.read_csv(filepath)
        return {row[key_col].lower(): float(row[score_col]) for _, row in df.iterrows()}
    
    def load_pair_score_dict(filepath: str, pair_col: str, score_col: str = 'score') -> dict:
        df = pd.read_csv(filepath)
        result = {}
        for _, row in df.iterrows():
            pair_str = str(row[pair_col])
            if len(pair_str) == 2:
                key = (pair_str[0].lower(), pair_str[1].lower())
                result[key] = float(row[score_col])
        return result
    
    # Load all dictionaries
    item_scores = load_score_dict(config.paths.item_scores_file, 'item')
    item_pair_scores = load_pair_score_dict(config.paths.item_pair_scores_file, 'item_pair')
    position_scores = load_score_dict(config.paths.position_scores_file, 'position')
    position_pair_scores = load_pair_score_dict(config.paths.position_pair_scores_file, 'position_pair')
    
    return item_scores, item_pair_scores, position_scores, position_pair_scores

def calculate_layout_score(items_str: str, positions_str: str, config: Config, 
                         detailed: bool = True, mode: str = 'combined') -> tuple:
    """
    Calculate score for a given layout using the consolidated scoring system.
    
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
        print(f"\nCalculating score for layout:")
        print(f"  Items ({len(items)}): {''.join(items)}")
        print(f"  Positions ({len(positions)}): {''.join(positions)}")
        print(f"  Scoring mode: {mode}")
    
    # Load normalized scores
    if detailed:
        print("\nLoading normalized scores...")
    
    score_dicts = load_normalized_scores_simple(config)
    
    # Get pre-assigned items if any
    items_assigned = list(config.optimization.items_assigned) if config.optimization.items_assigned else None
    positions_assigned = list(config.optimization.positions_assigned) if config.optimization.positions_assigned else None
    
    # Prepare scoring arrays
    arrays = prepare_scoring_arrays(
        items_to_assign=items,
        positions_to_assign=positions,
        norm_item_scores=score_dicts[0],
        norm_item_pair_scores=score_dicts[1],
        norm_position_scores=score_dicts[2],
        norm_position_pair_scores=score_dicts[3],
        items_assigned=items_assigned,
        positions_assigned=positions_assigned
    )
    
    # Create scorer
    scorer = LayoutScorer(arrays, mode=mode)
    
    # Create mapping array (direct 1:1 mapping since we're scoring a complete layout)
    mapping = np.arange(len(items), dtype=np.int16)
    
    # Calculate scores
    if mode == 'multi_objective':
        # MOO mode returns list of objectives
        objectives = scorer.score_layout(mapping)
        total_score = sum(objectives)  # Combined for display
        
        # Get detailed breakdown
        components = scorer.get_components(mapping)
        
        if detailed:
            print(f"\nMulti-Objective Results:")
            print(f"  Item objective:   {objectives[0]:.12f}")
            print(f"  Pair objective:   {objectives[1]:.12f}")
            print(f"  Cross objective:  {objectives[2]:.12f}")
            print(f"  Combined total:   {total_score:.12f}")
        
        return total_score, objectives[0], objectives[1], objectives[2]
    
    else:
        # SOO modes
        total_score, item_score, pair_score, cross_score = scorer.score_layout(mapping, return_components=True)
        
        if detailed:
            print(f"\nSingle-Objective Results ({scorer.mode_name}):")
            print(f"  Total score:      {total_score:.12f}")
            print(f"  Item component:   {item_score:.12f}")
            print(f"  Pair component:   {pair_score:.12f}")
            print(f"  Cross component:  {cross_score:.12f}")
        
        return total_score, item_score, pair_score, cross_score

def print_detailed_breakdown(items_str: str, positions_str: str, config: Config, scorer: LayoutScorer):
    """Print detailed item-by-item scoring breakdown."""
    items = list(items_str.lower())
    positions = list(positions_str.upper())
    
    print(f"\nDetailed Item Breakdown:")
    print(f"  {'Item':<4} | {'Pos':<3} | {'Item Score':<10} | {'Pos Score':<10} | {'Combined':<10}")
    print(f"  {'-'*4}-+-{'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    
    # Get individual scores
    item_details = []
    
    for i, (item, pos) in enumerate(zip(items, positions)):
        item_score = scorer.arrays.item_scores[i]
        pos_score = scorer.arrays.position_matrix[i, i]  # Diagonal for individual position scores
        combined = item_score * pos_score
        
        item_details.append((item, pos, item_score, pos_score, combined))
    
    # Sort by item score (highest first) for better display
    item_details.sort(key=lambda x: x[2], reverse=True)
    
    for item, pos, item_score, pos_score, combined in item_details:
        print(f"  {item:<4} | {pos:<3} | {item_score:<10.6f} | {pos_score:<10.6f} | {combined:<10.6f}")

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

#-----------------------------------------------------------------------------
# Main Function and CLI
#-----------------------------------------------------------------------------

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate layout score using consolidated scoring system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic scoring
  python calculate_score.py --items "abc" --positions "FDJ"
  
  # With detailed breakdown
  python calculate_score.py --items "etaoinsrhld" --positions "FJDSVERAWCQ" --details
  
  # Different scoring modes
  python calculate_score.py --items "abc" --positions "FDJ" --mode item_only
  python calculate_score.py --items "abc" --positions "FDJ" --mode multi_objective
  
  # With validation
  python calculate_score.py --items "abc" --positions "FDJ" --validate
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
        
        print("Layout Score Calculator")
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
        total_score, item_score, pair_score, cross_score = calculate_layout_score(
            args.items, args.positions, config, args.details, args.mode
        )
        
        # Show detailed breakdown if requested
        if args.details and args.mode != 'multi_objective':
            # Create temporary scorer for detailed analysis
            score_dicts = load_normalized_scores_simple(config)
            arrays = prepare_scoring_arrays(
                list(args.items.lower()), list(args.positions.upper()),
                score_dicts[0], score_dicts[1], score_dicts[2], score_dicts[3]
            )
            scorer = LayoutScorer(arrays, args.mode)
            print_detailed_breakdown(args.items, args.positions, config, scorer)
        
        # Show keyboard layout if requested
        if args.keyboard:
            complete_mapping = create_complete_layout_mapping(args.items, args.positions, config)
            print("\nKeyboard Layout:")
            visualize_keyboard_layout(
                mapping=complete_mapping,
                title="Calculated Layout",
                config=config
            )
        
        # Final summary
        print(f"\nFinal Scores ({args.mode} mode):")
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
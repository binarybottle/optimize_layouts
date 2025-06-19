# score_layout.py
"""
Layout score calculator that shows complete layout scores.

Example usage:
    python score_layout.py --items "etaoinsrhldcumfp" --positions "FDESRJKUMIVLA;OW" --details
    python score_layout.py --items "abc" --positions "FDJ" --config config.yaml --verbose
    python score_layout.py --items "etaoinsrhldcumfp" --positions "FDESRJKUMIVLA;OW" --independent
"""

import argparse
import numpy as np

# Import consolidated modules
from config import load_config, Config
from display import visualize_keyboard_layout
from validation import validate_specific_layout
from optimize_layout import load_normalized_scores  # Use existing function
from scoring import (
    calculate_complete_layout_score,
    calculate_complete_layout_score_direct,
    create_complete_layout_scorer,
    apply_default_combination
)

#-----------------------------------------------------------------------------
# Utility functions
#-----------------------------------------------------------------------------
def create_complete_layout_mapping(items_str: str, positions_str: str, config: Config) -> dict:
    """Create layout mapping from provided items and positions only."""
    # Score only the provided items/positions (ignore config pre-assignments)
    mapping = dict(zip(items_str.lower(), positions_str.upper()))
    return mapping

def print_detailed_breakdown(complete_mapping: dict, normalized_scores: tuple):
    """Print detailed item-by-item scoring breakdown."""
    print(f"\nDetailed Item Breakdown:")
    print(f"  {'Item':<4} | {'Pos':<3} | {'Item Score':<10} | {'Pos Score':<10} | {'Combined':<10}")
    print(f"  {'-'*4}-+-{'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    
    # Create scorer for detailed analysis
    scorer = create_complete_layout_scorer(complete_mapping, normalized_scores, mode='combined')
    
    item_details = []
    items = list(complete_mapping.keys())
    positions = list(complete_mapping.values())
    
    # Get individual scores
    for i, (item, pos) in enumerate(zip(items, positions)):
        if i < len(scorer.arrays.item_scores):
            item_score = scorer.arrays.item_scores[i]
            pos_score = scorer.arrays.position_matrix[i, i] if i < scorer.arrays.position_matrix.shape[0] else 0.0
            combined = apply_default_combination(item_score, pos_score)
            item_details.append((item, pos, item_score, pos_score, combined))
    
    # Sort by combined score (highest first) for better display
    item_details.sort(key=lambda x: x[4], reverse=True)
    
    for item, pos, item_score, pos_score, combined in item_details:
        print(f"  {item:<4} | {pos:<3} | {item_score:<10.6f} | {pos_score:<10.6f} | {combined:<10.6f}")

#-----------------------------------------------------------------------------
# Main function and CLI
#-----------------------------------------------------------------------------
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate complete layout score including all items and pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic complete scoring
  python score_layout.py --items "abc" --positions "FDJ"
  
  # Score full layout
  python score_layout.py --items "etaoinsrhldcumfp" --positions "FDESRJKUMIVLA;OW"
  
  # With detailed breakdown
  python score_layout.py --items "etaoinsrhldcumfp" --positions "FDESRJKUMIVLA;OW" --details
  
  # With validation and keyboard display
  python score_layout.py --items "abc" --positions "FDJ" --validate --keyboard
        """
    )
    
    parser.add_argument("--items", required=True, 
                       help="String of items (e.g., 'etaoinsrhldcumfp')")
    parser.add_argument("--positions", required=True,
                       help="String of positions (e.g., 'FDESRJKUMIVLA;OW')")
    parser.add_argument("--config", default="config.yaml",
                       help="Path to config file (default: config.yaml)")
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
        
        # Load normalized scores
        print("Loading normalized scores...")
        normalized_scores = load_normalized_scores(config)
        
        # Create complete layout mapping
        complete_mapping = create_complete_layout_mapping(args.items, args.positions, config)
        
        if len(args.items) > 0:
            all_items = ''.join(complete_mapping.keys())
            all_positions = ''.join(complete_mapping.values())
            print(f"Complete layout: {all_items} → {all_positions}")
        
        # Debug output to match SLURM format
        print(f"\nDEBUG: Scoring breakdown")
        print("Complete mapping:")
        for item, pos in complete_mapping.items():
            print(f"  {item} → {pos}")
        
        print(f"\nDEBUG: Detailed scoring (using direct calculation)")
        print(f"  Mapping length: {len(complete_mapping)}")
        print(f"  Items: {''.join(complete_mapping.keys())}")
        print(f"  Positions: {''.join(complete_mapping.values())}")
        
        # Calculate complete layout score using direct calculation (bypasses optimization scoring)
        total_score, item_score, item_pair_score = calculate_complete_layout_score_direct(
            complete_mapping, normalized_scores
        )
        
        print(f"  Direct calculation results:")
        print(f"    Item component: {item_score:.12f}")
        print(f"    Pair component: {item_pair_score:.12f}")
        print(f"    Total score: {total_score:.12f}")
        
        # Also calculate using old method for comparison
        old_total, old_item, old_pair = calculate_complete_layout_score(complete_mapping, normalized_scores)
        print(f"  Old method results (for comparison):")
        print(f"    Item component: {old_item:.12f}")
        print(f"    Pair component: {old_pair:.12f}")
        print(f"    Total score: {old_total:.12f}")
        
        print(f"\nComplete Layout Results:")
        print(f"  Total score:         {total_score:.12f}")
        print(f"  Item component:      {item_score:.12f}")
        print(f"  Item-Pair component: {item_pair_score:.12f}")
        
        # Show detailed breakdown if requested
        if args.details:
            print_detailed_breakdown(complete_mapping, normalized_scores)
        
        if args.keyboard or config.visualization.print_keyboard:
            print("\nKeyboard Layout:")
            visualize_keyboard_layout(
                mapping=complete_mapping,
                title="Complete Layout",
                config=config
            )
        
        # CSV format output (for compatibility with existing tools)
        print(f"\nCSV Format:")
        print(f"total_score,item_score,item_pair_score")
        print(f"{total_score:.6f},{item_score:.6f},{item_pair_score:.6f}")
        
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
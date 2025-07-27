# score_complete_layout.py
"""
Layout score calculator that calculates complete layout scores.

Note: Since the scoring accepts any characters for items and any positions,
there could be a problem if item or position files are incomplete.
Layouts with items or positions that are missing in these files 
will be assigned default scores. To mitigate issues, we default 
to using only letters for items for this script (positions can use any character).
  - Default (letters only) filters both items and corresponding positions, to "pyaoeuiqjkx" → "rtyuiopasdklzxcvbnm":
    python score_complete_layout.py --items "',.pyaoeui;qjkx" --positions "qwertyuiopasdfg"
  - The argument --nonletter-items uses any characters for items: "',.pyaoeui;qjkx":
    python score_complete_layout.py --items "',.pyaoeui;qjkx" --positions "FDESRJKUMIVLA;" --nonletter-items

Example usage:
    python score_complete_layout.py --items "etaoinsrhldcumfp" --positions "FDESRJKUMIVLA;OW" --details
    python score_complete_layout.py --items "abc" --positions "FDJ" --config config.yaml --validate --keyboard --details
"""

import argparse
import numpy as np

# Import consolidated modules
from config import load_config, Config
from display import visualize_keyboard_layout
from validation import validate_specific_layout
from optimize_layout import load_normalized_scores
from scoring import (
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

def filter_letter_pairs(items_str: str, positions_str: str, allow_all: bool) -> tuple[str, str]:
    """Filter to letter-position pairs only unless allow_all is True."""
    if allow_all:
        return items_str, positions_str  # Current behavior
    else:
        # Filter to letter pairs only
        filtered_items = []
        filtered_positions = []
        removed_pairs = []
        
        for item, pos in zip(items_str, positions_str):
            if item.isalpha():
                filtered_items.append(item)
                filtered_positions.append(pos)
            else:
                removed_pairs.append(f"{item}→{pos}")
        
        if removed_pairs:
            print(f"Note: Removed non-letter pairs: {removed_pairs}")
        
        return ''.join(filtered_items), ''.join(filtered_positions)
        
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
  python score_complete_layout.py --items "abc" --positions "FDJ"
  
  # Score full layout
  python score_complete_layout.py --items "etaoinsrhldcumfp" --positions "FDESRJKUMIVLA;OW"
  
  # With detailed breakdown
  python score_complete_layout.py --items "etaoinsrhldcumfp" --positions "FDESRJKUMIVLA;OW" --details
  
  # With validation and keyboard display
  python score_complete_layout.py --items "abc" --positions "FDJ" --validate --keyboard
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
    parser.add_argument("--nonletter-items", action="store_true",
                   help="Allow non-letter characters in --items (default: letters only)")
    parser.add_argument("--extended-positions", action="store_true",
                       help="Use extended position pair file for additional keyboard coverage")
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
        
        # Validate inputs - must have equal length first
        if len(args.items) != len(args.positions):
            print(f"Error: Item count ({len(args.items)}) != Position count ({len(args.positions)})")
            return

        # Filter to letter pairs
        valid_items, valid_positions = filter_letter_pairs(args.items, args.positions, args.all_characters)

        if len(valid_items) == 0:
            print("Error: No letters found in items string")
            return
        
        # Run validation if requested
        if args.validate:
            print("\nRunning layout validation...")
            validation_result = validate_specific_layout(valid_items, valid_positions, config)
            print(f"  {validation_result}")
            print()
        
        # Load normalized scores
        print("Loading normalized scores...")
        normalized_scores = load_normalized_scores(config)
        
        # Create complete layout mapping
        complete_mapping = create_complete_layout_mapping(valid_items, valid_positions, config)
        
        if len(valid_items) > 0:
            all_items = ''.join(complete_mapping.keys())
            all_positions = ''.join(complete_mapping.values())
            print(f"Complete layout: {all_items} → {all_positions}")
        
        # Debug output
        print("Complete mapping:")
        for item, pos in complete_mapping.items():
            print(f"  {item} → {pos}")
        
        # Calculate complete layout score using direct calculation (bypasses optimization scoring)
        total_score, item_score, item_pair_score = calculate_complete_layout_score_direct(
            complete_mapping, normalized_scores
        )
        
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
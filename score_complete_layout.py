# score_complete_layout.py
"""
Layout score calculator that calculates complete layout scores.

Note: Since the scoring accepts any characters for items and any positions,
there could be a problem if item or position files are incomplete.
Layouts with items or positions that are missing in these files 
will be assigned default scores (see below). To mitigate issues, we default 
to using only letters for items for this script (positions can use any character).
  - Default (letters only) filters both items and corresponding positions, to "pyaoeuiqjkx" → "rtyuiopasdklzxcvbnm":
    python score_complete_layout.py --items "',.pyaoeui;qjkx" --positions "qwertyuiopasdfg"
  - The argument --nonletter-items uses any characters for items: "',.pyaoeui;qjkx":
    python score_complete_layout.py --items "',.pyaoeui;qjkx" --positions "FDESRJKUMIVLA;" --nonletter-items
Also, scores are only calculated for position-pairs that are present in the position-pair file.
In the case of --position-pair-file "input/normalized_key_pair_comfort_scores_extended.csv",
scores are calculated only for within-hand bigrams:
  - prepare_scoring_arrays() will assign default scores of 1.0 for any missing pairs:
    - missing_item_pair_score: float = 1.0,      # ← DEFAULT SCORE FOR MISSING ITEM PAIRS
    - missing_position_pair_score: float = 1.0   # ← DEFAULT SCORE FOR MISSING POSITION PAIRS
The --ignore-keyboard-sides flag can be used to filter out cross-hand bigrams.

Usage:

# Standard usage (uses position-pair file from config.yaml):
>> python score_complete_layout.py --items "',.pyfgcrlaoeuidhtns;qjkxbmwvz" \
    --positions "qwertyuiopasdfghjkl;zxcvbnm,./"

# With all options:
>> python score_complete_layout.py \
    --items     "',.pyfgcrlaoeuidhtns;qjkxbmwvz" \
    --positions "qwertyuiopasdfghjkl;zxcvbnm,./" \
    --position-pair-file "input/normalized_key_pair_comfort_scores_extended.csv" \
    --details --validate

# For keyboard applications, filter out same-hand bigrams and visualize a keyboard:
>> python score_complete_layout.py --items "',.pyfgcrlaoeuidhtns;qjkxbmwvz" \
    --positions "qwertyuiopasdfghjkl;zxcvbnm,./" --keyboard --ignore-keyboard-sides \
    --position-pair-file "input/normalized_key_pair_comfort_scores_extended.csv"

# Your custom file should have format:
# position_pair,score
# qw,0.85
# we,0.72
# ty,0.65

For reference:    
qwerty_layout = "qwertyuiopasdfghjkl;zxcvbnm,./"
dvorak_layout = "',.pyfgcrlaoeuidhtns;qjkxbmwvz"

"""

import argparse
from email import parser
import pandas as pd
import numpy as np
from typing import Tuple

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

# Keyboard hand mapping for filtering cross-hand bigrams  
POSITION_HANDS = {
    # Left hand positions
    '1': 'L', '2': 'L', '3': 'L', '4': 'L', '5': 'L',
    'Q': 'L', 'W': 'L', 'E': 'L', 'R': 'L', 'T': 'L',
    'A': 'L', 'S': 'L', 'D': 'L', 'F': 'L', 'G': 'L',
    'Z': 'L', 'X': 'L', 'C': 'L', 'V': 'L', 'B': 'L',
    # Right hand positions  
    '6': 'R', '7': 'R', '8': 'R', '9': 'R', '0': 'R',
    'Y': 'R', 'U': 'R', 'I': 'R', 'O': 'R', 'P': 'R',
    'H': 'R', 'J': 'R', 'K': 'R', 'L': 'R', ';': 'R',
    'N': 'R', 'M': 'R', ',': 'R', '.': 'R', '/': 'R',
    "'": 'R', '[': 'R',
}

def is_same_hand_pair(pos1: str, pos2: str) -> bool:
    """Check if two positions are on the same hand of the keyboard."""
    pos1_upper = pos1.upper()
    pos2_upper = pos2.upper()
    if pos1_upper not in POSITION_HANDS or pos2_upper not in POSITION_HANDS:
        return False
    return POSITION_HANDS[pos1_upper] == POSITION_HANDS[pos2_upper]

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

def validate_positions_in_pair_file(positions_str: str, position_pair_scores: dict, ignore_keyboard_sides: bool = False) -> None:
    """
    Validate that all positions used in the layout have corresponding entries 
    in the position-pair file. Error if individual positions are missing,
    but only warn about missing bigrams.
    
    Args:
        positions_str: String of positions used in layout
        position_pair_scores: Dictionary of position pair scores
        
    Raises:
        ValueError: If individual positions are missing from position-pair file
    """
    # Get unique positions used in layout
    used_positions = set(pos.lower() for pos in positions_str)
    
    # Get positions that appear in the position-pair file
    available_positions = set()
    for (pos1, pos2) in position_pair_scores.keys():
        available_positions.add(pos1)
        available_positions.add(pos2)
    
    # Check for missing individual positions - this is an ERROR
    missing_positions = used_positions - available_positions
    
    if missing_positions:
        missing_str = ''.join(sorted(missing_positions)).upper()
        available_str = ''.join(sorted(available_positions)).upper()
        print(f"Error: The following positions are used in --positions but missing from position-pair file:")
        print(f"  Missing positions: {missing_str}")
        print(f"  Available positions in file: {available_str}")
        raise ValueError(f"Positions {missing_str} not found in position-pair file")
    
    # Check for missing position pairs - filter based on hand if requested
    missing_pairs = []
    positions_list = list(used_positions)
    
    for i in range(len(positions_list)):
        for j in range(len(positions_list)):
            if i != j:
                pos1, pos2 = positions_list[i], positions_list[j]
                
                # Skip cross-hand pairs if filtering is enabled
                if ignore_keyboard_sides and not is_same_hand_pair(pos1, pos2):
                    continue
                    
                if (pos1.lower(), pos2.lower()) not in position_pair_scores:
                    missing_pairs.append(f"{pos1}{pos2}".upper())

    if missing_pairs:
        # Sort for consistent output
        missing_pairs.sort()
        missing_pairs_str = ' '.join(missing_pairs)
        print(f"  Note: Missing position pairs (will use default scores): {missing_pairs_str}")
        if len(missing_pairs) > 20:  # Only show count if there are many
            print(f"  Total missing pairs: {len(missing_pairs)}")
    else:
        print(f"  ✓ All position pairs found in position-pair file")

def load_scores_with_custom_position_pairs(config: Config, custom_position_pair_file: str = None) -> Tuple:
    """
    Load normalized scores, optionally using a custom position-pair file.
    
    Args:
        config: Configuration object
        custom_position_pair_file: Path to custom position-pair file (raw, will be normalized)
        
    Returns:
        Tuple of (item_scores, item_pair_scores, position_scores, position_pair_scores)
    """
    # Load standard scores (item, item_pair, position)
    from scoring import load_normalized_scores
    item_scores, item_pair_scores, position_scores, position_pair_scores = load_normalized_scores(config)
    
    if not custom_position_pair_file:
        # Use standard position pair scores from config
        print("Using standard position-pair file from config")
        return item_scores, item_pair_scores, position_scores, position_pair_scores
    
    # Load and normalize custom position-pair file
    try:
        print(f"Loading position-pair file: {custom_position_pair_file}")
        
        # Load raw custom file
        # Treat "NA" as literal text instead of converting it to NaN
        df = pd.read_csv(custom_position_pair_file, dtype={'position_pair': str}, keep_default_na=False)
        
        if 'score' not in df.columns:
            raise ValueError(f"Custom position-pair file must have 'score' column")
        if 'position_pair' not in df.columns:
            raise ValueError(f"Custom position-pair file must have 'position_pair' column")
        
        # Normalize scores using same method as normalize_input.py
        scores = df['score'].values
        
        # Simple min-max normalization to [0,1]
        if len(scores) > 0 and not np.all(scores == scores[0]):
            min_score = np.min(scores)
            max_score = np.max(scores)
            if max_score != min_score:
                normalized_scores = (scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.ones_like(scores) * 0.5  # All same value -> 0.5
        else:
            normalized_scores = np.zeros_like(scores)  # Empty or constant -> 0
        
        print(f"  Original range: [{np.min(scores):.6f}, {np.max(scores):.6f}]")
        print(f"  Normalized range: [{np.min(normalized_scores):.6f}, {np.max(normalized_scores):.6f}]")
        
        # Convert to dictionary format expected by scoring system
        custom_position_pair_scores = {}
        for i, row in df.iterrows():
            pair_str = str(row['position_pair'])
            if len(pair_str) == 2:
                key = (pair_str[0].lower(), pair_str[1].lower())
                custom_position_pair_scores[key] = float(normalized_scores[i])
        
        print(f"  Loaded {len(custom_position_pair_scores)} position-pair scores")
        
        # Return with custom position-pair scores
        return item_scores, item_pair_scores, position_scores, custom_position_pair_scores
        
    except FileNotFoundError:
        print(f"Error: Custom position-pair file not found: {custom_position_pair_file}")
        print("Using standard position-pair file from config")
        return item_scores, item_pair_scores, position_scores, position_pair_scores
    except Exception as e:
        print(f"Error loading custom position-pair file: {e}")
        print("Using standard position-pair file from config")
        return item_scores, item_pair_scores, position_scores, position_pair_scores

def print_detailed_breakdown(complete_mapping: dict, normalized_scores: tuple, ignore_keyboard_sides: bool = False):
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

def calculate_complete_layout_score_with_filtering(complete_mapping: dict, normalized_scores: tuple, ignore_keyboard_sides: bool = False) -> tuple:
    """Calculate complete layout score with optional same-hand filtering."""
    if not ignore_keyboard_sides:
        # Use existing function for normal scoring
        total_score, item_score, item_pair_score = calculate_complete_layout_score_direct(complete_mapping, normalized_scores)
        return total_score, item_score, item_pair_score, {'cross_hand_pairs_filtered': 0, 'pairs_used': len(complete_mapping) * (len(complete_mapping) - 1)}
    
    # Custom scoring with filtering
    norm_item_scores, norm_item_pair_scores, norm_position_scores, norm_position_pair_scores = normalized_scores
    
    items = list(complete_mapping.keys())
    positions = list(complete_mapping.values())
    n_items = len(items)
    
    # Calculate item component (not affected by filtering)
    item_raw_score = 0.0
    for item, pos in complete_mapping.items():
        item_score = norm_item_scores.get(item.lower(), 0.0)
        pos_score = norm_position_scores.get(pos.lower(), 0.0)
        item_raw_score += item_score * pos_score
    
    item_component = item_raw_score / n_items
    
    # Calculate item-pair component with filtering
    pair_raw_score = 0.0
    pair_count = 0
    cross_hand_pairs_filtered = 0
    
    for i in range(n_items):
        for j in range(n_items):
            if i != j:  # Skip self-pairs
                item1, item2 = items[i], items[j]
                pos1, pos2 = positions[i], positions[j]
                
                # Filter cross-hand pairs if requested
                if not is_same_hand_pair(pos1, pos2):
                    cross_hand_pairs_filtered += 1
                    continue
                
                # Get scores
                item_pair_key = (item1.lower(), item2.lower())
                item_pair_score = norm_item_pair_scores.get(item_pair_key, 1.0)
                
                pos_pair_key = (pos1.lower(), pos2.lower())
                pos_pair_score = norm_position_pair_scores.get(pos_pair_key, 1.0)
                
                pair_raw_score += item_pair_score * pos_pair_score
                pair_count += 1
    
    pair_component = pair_raw_score / max(1, pair_count)
    total_score = item_component * pair_component
    
    filtered_info = {
        'cross_hand_pairs_filtered': cross_hand_pairs_filtered,
        'pairs_used': pair_count
    }
    
    return total_score, item_component, pair_component, filtered_info

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
    parser.add_argument("--csv", action="store_true",
                       help="Output in CSV format (total_score,item_score,item_pair_score)")    
    parser.add_argument("--nonletter-items", action="store_true",
                       help="Allow non-letter characters in --items (default: letters only)")
    parser.add_argument("--position-pair-file", default="input/normalized_key_pair_comfort_scores_extended.csv",
                       help="Path to position-pair file (default overrides optimize_layout's default)")
    parser.add_argument("--validate", action="store_true",
                       help="Run validation on this specific layout")
    parser.add_argument("--keyboard", action="store_true",
                       help="Show keyboard visualization")
    parser.add_argument("--ignore-keyboard-sides", action="store_true",
                       help="For keyboard applications, ignore bigrams that use both hands")

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
        valid_items, valid_positions = filter_letter_pairs(args.items, args.positions, args.nonletter_items)

        if len(valid_items) == 0:
            print("Error: No letters found in items string")
            return
        
        # Run validation if requested
        if args.validate:
            print("\nRunning layout validation...")
            validation_result = validate_specific_layout(valid_items, valid_positions, config)
            print(f"  {validation_result}")
            print()
        
        # Load normalized scores with optional custom position-pair file
        print("Loading normalized scores...")
        normalized_scores = load_scores_with_custom_position_pairs(config, args.position_pair_file)
        item_scores, item_pair_scores, position_scores, position_pair_scores = normalized_scores
        
        # Validate that all positions are available in position-pair file
        print("Validating positions against position-pair file...")
        try:
            validate_positions_in_pair_file(valid_positions, position_pair_scores, args.ignore_keyboard_sides)
        except ValueError as e:
            print(f"\nValidation error: {e}")
            print("\nSuggestions:")
            print("  1. Use a position-pair file that includes all your positions")
            print("  2. Remove positions from --positions that aren't in your file") 
            print("  3. Use the extended position-pair file that covers more keys")
            return
        
        # Create complete layout mapping
        complete_mapping = create_complete_layout_mapping(valid_items, valid_positions, config)

        if len(valid_items) > 0:
            all_items = ''.join(complete_mapping.keys())
            all_positions = ''.join(complete_mapping.values())
            print(f"Complete layout: {all_items} → {all_positions}")
        
        if args.details:
            print("Complete mapping:")
            for item, pos in complete_mapping.items():
                print(f"  {item} → {pos}")
            
        # Calculate complete layout score with optional filtering
        total_score, item_score, item_pair_score, filtered_info = calculate_complete_layout_score_with_filtering(
            complete_mapping, normalized_scores, args.ignore_keyboard_sides
        )

        if args.csv:
            # CSV output only - this is the exact format compare_layouts.py expects
            print("total_score,item_score,item_pair_score")
            print(f"{total_score:.12f},{item_score:.12f},{item_pair_score:.12f}")
        else:
            # Human-readable output (existing code)
            print(f"\nComplete Layout Results:")
            print(f"  Total score:         {total_score:.12f}")
            print(f"  Item component:      {item_score:.12f}")
            print(f"  Item-Pair component: {item_pair_score:.12f}")
        
        # Show filtering information if applicable
        if args.ignore_keyboard_sides:
            pairs_filtered = filtered_info['cross_hand_pairs_filtered']
            pairs_used = filtered_info['pairs_used']
            total_possible = len(complete_mapping) * (len(complete_mapping) - 1)
            print(f"  Total pairs considered:  {total_possible}")
            print(f"  Filtering ratio:           {pairs_filtered/total_possible*100:.1f}% filtered")
            print(f"\nSame-hand filtering (--ignore-keyboard-sides):")
            print(f"  Cross-hand pairs filtered: {pairs_filtered}")
            print(f"  Same-hand pairs used:      {pairs_used}")
            print(f"  Total possible pairs:      {total_possible}")
            print(f"  Filtering ratio:           {pairs_filtered/total_possible*100:.1f}% filtered")
        
        # Show detailed breakdown if requested
        if args.details:
            print_detailed_breakdown(complete_mapping, normalized_scores, args.ignore_keyboard_sides)

        # Visualize keyboard layout if requested
        if args.keyboard:
            visualize_keyboard_layout(
                mapping=complete_mapping,
                title="Complete keyboard layout",
                config=config
            )
        elif hasattr(config, 'visualization') and config.visualization.print_keyboard:
            visualize_keyboard_layout(
                mapping=complete_mapping,
                title="Complete keyboard layout",
                config=config
            )
                    
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
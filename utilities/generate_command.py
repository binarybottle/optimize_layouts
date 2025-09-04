#!/usr/bin/env python3
"""
Generate score_layouts.py commands from MOO analysis results for compare and visualize workflow.

This script creates ready-to-run commands for the score_layouts.py → compare_layouts.py workflow
using results from multi-objective optimization (MOO) analysis.

Usage:
    poetry run python3 generate_command.py  --python-cmd "poetry run python3" \
        --additional-items "qz'\",.-?" \
        --additional-positions "['TYGHBN" \
        --csv moo_results.csv

Then run the generated commands:
    1. Copy/paste from the output file score_layout_commands.txt
    2. Run the score_layouts.py command
    3. Run the compare_layouts.py commands
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

def generate_score_layouts_command(df: pd.DataFrame, 
                                 output_dir: Path, 
                                 additional_items: str = "", 
                                 additional_positions: str = "",
                                 python_cmd: str = "poetry run python3") -> Path:
    """
    Generate score_layouts.py commands from MOO analysis results for visualization.
    
    Creates commands compatible with the score_layouts.py → compare_layouts.py workflow.
    Reorders items to match QWERTY position sequence for consistency.
    
    Args:
        df: DataFrame with MOO solutions (columns: 'items', 'positions')
        output_dir: Output directory for command files
        additional_items: String of additional characters to append to each solution's items
        additional_positions: String of additional positions to append to each solution's positions  
        python_cmd: Python command to use (e.g., "python3", "poetry run python3")
    
    Returns:
        Path to the generated command file
    """
    # Standard QWERTY position order for score_layouts.py
    qwerty_positions = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"
    
    layout_specs = []
    
    for idx, row in df.iterrows():
        # Parse items and positions from the MOO solution
        original_items = list(row['items'])
        original_positions = list(row['positions'])
        
        # Extend with additional characters if provided
        extended_items = original_items + list(additional_items)
        extended_positions = original_positions + list(additional_positions)
        
        # Ensure we don't have mismatched lengths
        if len(extended_items) != len(extended_positions):
            print(f"Warning: Layout {idx+1} has mismatched items/positions lengths after extension")
            print(f"  Items: {len(extended_items)}, Positions: {len(extended_positions)}")
            # Truncate to the shorter length
            min_len = min(len(extended_items), len(extended_positions))
            extended_items = extended_items[:min_len]
            extended_positions = extended_positions[:min_len]
        
        # Create mapping: Position -> Character
        position_to_char = {}
        for char, pos in zip(extended_items, extended_positions):
            position_to_char[pos.upper()] = char
        
        # Build layout string by going through QWERTY positions in order
        layout_string = ""
        missing_positions = []
        for qwerty_pos in qwerty_positions:
            if qwerty_pos in position_to_char:
                layout_string += position_to_char[qwerty_pos]
            else:
                # For positions not in the MOO solution, use the default QWERTY character
                fallback_char = qwerty_pos.lower() if qwerty_pos.isalpha() else qwerty_pos
                layout_string += fallback_char
                missing_positions.append(qwerty_pos)
                
        # Escape double quotes within the layout string for command line
        escaped_layout_string = layout_string.replace('"', '\\"')
        layout_name = f"moo_layout_{idx + 1}"
        layout_specs.append(f'{layout_name}:"{escaped_layout_string}"')
    
    # Write command file in the paste.txt format
    command_file = output_dir / 'score_layout_commands.txt'
    with open(command_file, 'w') as f:
        # Header comments matching paste.txt format
        f.write("# MOO Layout Analysis → score_layouts.py → compare_layouts.py Commands\n")
        f.write("# Three-step workflow for scoring, visualizing, and ranking MOO optimization results\n")
        f.write(f"# Python command used: {python_cmd}\n")
        f.write(f"# Additional items: '{additional_items}'\n")
        f.write(f"# Additional positions: '{additional_positions}'\n")
        f.write(f"# Number of layouts: {len(df)}\n")
        f.write("# Layout numbers correspond to original MOO analysis line numbers\n")
        f.write("# Items reordered to match QWERTY position order: QWERTYUIOPASDFGHJKL;ZXCVBNM,./['\n")
        f.write("\n")
        
        # STEP 1: Single long command with all layouts
        f.write("# STEP 1: Score all MOO layouts using score_layouts.py\n\n")
        f.write(f"{python_cmd} score_layouts.py --compare ")
        f.write(" ".join(layout_specs))
        f.write(" --csv output/moo_layout_scores.csv\n\n\n")
        
        # STEP 2: Compare layouts commands for visualization
        f.write("# STEP 2: Create visualizations and metric-ranked tables using compare_layouts.py\n\n")
        
        # Basic metrics command
        f.write(f"{python_cmd} compare_layouts.py --tables output/moo_layout_scores.csv ")
        f.write("--metrics comfort comfort-key dvorak7 time_total distance_total ")
        f.write("--output output/moo_layout_scores\n\n")
        
        # Detailed metrics command
        f.write(f"{python_cmd} compare_layouts.py --tables output/moo_layout_scores.csv ")
        f.write("--metrics comfort comfort-key dvorak7 dvorak7_repetition dvorak7_movement ")
        f.write("dvorak7_vertical dvorak7_horizontal dvorak7_adjacent dvorak7_weak dvorak7_outward ")
        f.write("time_total time_setup time_interval time_return distance_total distance_setup ")
        f.write("distance_interval distance_return ")
        f.write("--output output/moo_layout_scores_detailed\n\n")

         # STEP 3: Compare layouts commands for ranking
        f.write("# STEP 2: Create visualizations and metric-ranked tables using compare_layouts.py\n\n")
        
        # Basic metrics command
        f.write(f"{python_cmd} compare_layouts.py --tables output/moo_layout_scores.csv ")
        f.write("--metrics dvorak7 time_total distance_total ")
        f.write("--rankings output/moo_layout_scores_rankings.csv\n\n")
        
        # Detailed metrics command
        f.write(f"{python_cmd} compare_layouts.py --tables output/moo_layout_scores.csv ")
        f.write("--metrics dvorak7 dvorak7_repetition dvorak7_movement ")
        f.write("dvorak7_vertical dvorak7_horizontal dvorak7_adjacent dvorak7_weak dvorak7_outward ")
        f.write("time_total time_setup time_interval time_return distance_total distance_setup ")
        f.write("distance_interval distance_return ")
        f.write("--rankings output/moo_layout_scores_rankings_detailed.csv\n\n")
    
    print(f"\nMOO layout scorer commands generated:")
    print(f"  Commands file: {command_file}")
    print(f"Command includes {len(df)} MOO layouts for score_layouts.py → compare_layouts.py workflow")
    print(f"Items reordered to match QWERTY position order")
    if additional_items or additional_positions:
        print(f"Extended each layout with:")
        print(f"  Additional items: '{additional_items}'")
        print(f"  Additional positions: '{additional_positions}'")
    
    print(f"\nUsage:")
    print(f"  1. Copy/paste commands from: {command_file}")
    print(f"  2. Run the STEP 1 command to score all layouts")
    print(f"  3. Run the STEP 2 commands to create visualizations")
    print(f"  4. Run the STEP 3 commands to create rankings")
    print(f"\nOutput files will be:")
    print(f"  - output/moo_layout_scores.csv (CSV with all scores)")
    print(f"  - output/moo_layout_scores_rankings.csv (Basic metrics rankings)")
    print(f"  - output/moo_layout_scores_rankings_detailed.csv (Detailed metrics rankings)")
    print(f"  - Various visualization files in output/ directory")
    
    return command_file

# Example usage for standalone script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate score_layouts.py commands from MOO results')
    parser.add_argument('--csv', required=True, help='MOO results CSV file with items,positions columns')
    parser.add_argument('--output-dir', default='output', help='Output directory for command files')
    parser.add_argument('--additional-items', default='', help='Additional characters to append')
    parser.add_argument('--additional-positions', default='', help='Additional positions to append')
    parser.add_argument('--python-cmd', default='poetry run python3', help='Python command to use')
    
    args = parser.parse_args()
    
    # Load MOO results
    df = pd.read_csv(args.csv)
    
    # Generate commands
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    command_file = generate_score_layouts_command(
        df=df,
        output_dir=output_dir,
        additional_items=args.additional_items,
        additional_positions=args.additional_positions,
        python_cmd=args.python_cmd
    )
    
    print(f"\nGenerated command file: {command_file}")
#!/usr/bin/env python3
"""
Generate score_layouts.py commands from MOO analysis results for visualization workflow.

This script creates ready-to-run commands for the score_layouts.py → compare_layouts.py workflow
using results from multi-objective optimization (MOO) analysis.

Usage:
    from generate_command import generate_score_layouts_command
    
    # After MOO analysis
    command_file = generate_score_layouts_command(
        df=filtered_solutions_df,
        output_dir=Path("output"),
        additional_items="qz'\",.-?",
        additional_positions="['TYGHBN"
    )

Then run the generated commands:
    1. ./output/score_layouts_command.sh
    2. python compare_layouts.py --tables layout_scores.csv --output comparison
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

def generate_score_layouts_command(df: pd.DataFrame, 
                                 output_dir: Path, 
                                 additional_items: str = "", 
                                 additional_positions: str = "",
                                 python_cmd: str = "python3") -> Path:
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
                
        # Escape any special characters for command line
        escaped_layout_string = layout_string.replace('"', '\\"') #.replace("'", "\\'")
        
        # Use original CSV line number as layout name
        layout_name = f"moo_layout_{idx + 1}"
        layout_specs.append(f'{layout_name}:"{escaped_layout_string}"')
    
    # Generate score_layouts.py command (step 1)
    scores_file = "moo_layout_scores.csv"
    step1_parts = [
        f"{python_cmd} score_layouts.py",
        "--compare",
        " ".join(layout_specs),
        "--csv-output"
    ]
    step1_command = " ".join(step1_parts) + f" > {scores_file}"
    
    # Generate compare_layouts.py commands (step 2)
    step2_command = f"{python_cmd} compare_layouts.py --tables {scores_file} --output moo_layout_comparison"
    step2_raw = f"{python_cmd} compare_layouts.py --tables {scores_file} --use-raw --output moo_layout_comparison_raw"
    step2_verbose = f"{python_cmd} compare_layouts.py --tables {scores_file} --verbose"
    
    # Write shell script
    command_file = output_dir / 'score_layouts_command.sh'
    with open(command_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# MOO Layout Analysis → score_layouts.py → compare_layouts.py Workflow\n")
        f.write("# Generated from MOO analysis results for visualization\n")
        f.write(f"# Python command used: {python_cmd}\n")
        if additional_items or additional_positions:
            f.write(f"# Additional items added: '{additional_items}'\n")
            f.write(f"# Additional positions added: '{additional_positions}'\n")
        f.write(f"# Number of layouts: {len(df)}\n")
        f.write("# Layout numbers correspond to original MOO analysis line numbers\n")
        f.write(f"# Items reordered to match QWERTY position order: {qwerty_positions.lower()}\n")
        f.write("\n")
        f.write("echo 'Step 1: Scoring MOO layouts with score_layouts.py...'\n")
        f.write(step1_command + "\n")
        f.write(f"echo 'Scores saved to {scores_file}'\n")
        f.write("\n")
        f.write("echo 'Step 2: Creating visualizations with compare_layouts.py...'\n")
        f.write(step2_command + "\n")
        f.write("echo 'Visualizations saved as moo_layout_comparison_parallel.png and moo_layout_comparison_heatmap.png'\n")
        f.write("\n")
        f.write("echo 'MOO layout visualization workflow complete!'\n")
    
    # Write text file with individual commands
    commands_file = output_dir / 'score_layouts_commands.txt'
    with open(commands_file, 'w') as f:
        f.write("# MOO Layout Analysis → score_layouts.py → compare_layouts.py Commands\n")
        f.write("# Two-step workflow for visualizing MOO optimization results\n")
        f.write(f"# Python command used: {python_cmd}\n")
        if additional_items or additional_positions:
            f.write(f"# Additional items: '{additional_items}'\n")
            f.write(f"# Additional positions: '{additional_positions}'\n")
        f.write(f"# Number of layouts: {len(df)}\n")
        f.write("# Layout numbers correspond to original MOO analysis line numbers\n")
        f.write(f"# Items reordered to match QWERTY position order: {qwerty_positions.lower()}\n")
        f.write("\n")
        f.write("# STEP 1: Score all MOO layouts using score_layouts.py\n")
        f.write("# This creates a CSV file with layout_name,scorer,weighted_score,raw_score\n")
        f.write(step1_command + "\n")
        f.write("\n")
        f.write("# STEP 2: Create visualizations using compare_layouts.py\n")
        f.write("# Basic visualization (weighted scores)\n")
        f.write(step2_command + "\n")
        f.write("\n")
        f.write("# Alternative visualization options:\n")
        f.write("# Using raw scores instead of weighted scores\n")
        f.write(step2_raw + "\n")
        f.write("\n")
        f.write("# Verbose output with detailed statistics\n")
        f.write(step2_verbose + "\n")
        f.write("\n")
        f.write("# View just the scores without visualization\n")
        f.write(f"cat {scores_file}\n")
        f.write("\n")
        f.write("# Clean up intermediate files (optional)\n")
        f.write(f"# rm {scores_file}\n")
    
    # Make shell script executable
    command_file.chmod(0o755)
    
    print(f"\nMOO layout visualization commands generated:")
    print(f"  Shell script: {command_file}")
    print(f"  Commands file: {commands_file}")
    print(f"  Scores file will be: {scores_file}")
    print(f"Command includes {len(df)} MOO layouts for score_layouts.py → compare_layouts.py workflow")
    print(f"Items reordered to match QWERTY position order")
    if additional_items or additional_positions:
        print(f"Extended each layout with:")
        print(f"  Additional items: '{additional_items}'")
        print(f"  Additional positions: '{additional_positions}'")
    
    print(f"\nUsage:")
    print(f"  1. Run the shell script: ./{command_file}")
    print(f"  2. Or run commands individually from: {commands_file}")
    print(f"\nOutput files will be:")
    print(f"  - {scores_file} (CSV with all scores)")
    print(f"  - moo_layout_comparison_parallel.png (parallel coordinates plot)")
    print(f"  - moo_layout_comparison_heatmap.png (heatmap visualization)")
    
    return command_file

# Example usage for standalone script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate score_layouts.py commands from MOO results')
    parser.add_argument('--csv', required=True, help='MOO results CSV file with items,positions columns')
    parser.add_argument('--output-dir', default='output', help='Output directory for command files')
    parser.add_argument('--additional-items', default='', help='Additional characters to append')
    parser.add_argument('--additional-positions', default='', help='Additional positions to append')
    parser.add_argument('--python-cmd', default='python3', help='Python command to use')
    
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
#!/usr/bin/env python3
"""
Analysis and visualization of global Pareto results.

Features:
- Pareto front visualization (colored by source file + clean grayscale version)
- Source file distribution analysis
- Letter-position stability matrix (sparse heatmap)
- Flexible filtering by letter-position constraints
- Layout scorer command generation for external comparison

Dependencies: pandas, matplotlib, numpy, seaborn

Usage: 
    python3 analyze_global_moo_solutions.py output/global_moo_solutions.csv
    python3 analyze_global_moo_solutions.py data.csv --filter-assignments "e:J"
    python3 analyze_global_moo_solutions.py data.csv --filter-assignments "e:J,t:F,a:S"
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from io import StringIO
import seaborn as sns
from collections import defaultdict


def parse_global_pareto_csv(filepath: str) -> pd.DataFrame:
    """Parse the global Pareto results CSV."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find data start - look for actual column headers
    data_start_idx = -1
    for i, line in enumerate(lines):
        if ('config_id' in line or 'items' in line or 'positions' in line or 
            'Complete Item' in line or 'Complete Pair' in line):
            data_start_idx = i
            break
    
    if data_start_idx == -1:
        raise ValueError("Could not find data header in file")
    
    data_lines = ''.join(lines[data_start_idx:])
    return pd.read_csv(StringIO(data_lines))


def detect_objective_columns(df):
    """Detect the objective column names."""
    item_col = None
    pair_col = None
    
    # Try different column name possibilities
    item_possibilities = ['Complete Item', 'item_score', 'Opt Item Score']
    pair_possibilities = ['Complete Pair', 'item_pair_score', 'Opt Item-Pair Score']
    
    for col in item_possibilities:
        if col in df.columns:
            item_col = col
            break
    
    for col in pair_possibilities:
        if col in df.columns:
            pair_col = col
            break
    
    if item_col is None or pair_col is None:
        print(f"Error: Could not find objective columns in data")
        print(f"Available columns: {list(df.columns)}")
        print(f"Looking for item score column from: {item_possibilities}")
        print(f"Looking for pair score column from: {pair_possibilities}")
        return None, None
    
    return item_col, pair_col


def add_ranking_columns(df, item_col, pair_col):
    """
    Add ranking columns to the dataframe.
    
    - source_rank: Original rank from source files (renamed from 'rank')
    - item_rank: New 1-to-N ranking based on item scores in global dataset (1 = highest score)
    - pair_rank: New 1-to-N ranking based on pair scores in global dataset (1 = highest score)  
    - global_rank: Sum of item_rank + pair_rank (lower = better overall)
    """
    # Rename existing 'rank' column to 'source_rank' if it exists
    if 'rank' in df.columns:
        df = df.rename(columns={'rank': 'source_rank'})
        print("Renamed 'rank' column to 'source_rank'")
    
    # Create new 1-to-N ranks based on scores within the global dataset
    # Use method='first' to ensure consecutive 1-to-N ranking (breaks ties by order)
    df['item_rank'] = df[item_col].rank(ascending=False, method='first').astype(int)
    df['pair_rank'] = df[pair_col].rank(ascending=False, method='first').astype(int)
    df['global_rank'] = df['item_rank'] + df['pair_rank']
    
    # Sort by global rank (lower is better since it's sum of ranks)
    df = df.sort_values('global_rank').reset_index(drop=True)
    
    return df


def calculate_stability_metrics(df):
    """
    Calculate letter and position stability metrics.
    
    Returns:
        letter_stability: dict mapping letter -> number of unique positions
        position_stability: dict mapping position -> number of unique letters
        assignment_counts: dict mapping (letter, position) -> count
    """
    # Track letter-position assignments
    letter_positions = defaultdict(set)
    position_letters = defaultdict(set)
    assignment_counts = defaultdict(int)
    
    for _, row in df.iterrows():
        items = list(row['items'])  # Convert string to character list
        positions = list(row['positions'])  # Convert string to character list
        
        for letter, position in zip(items, positions):
            letter_positions[letter].add(position)
            position_letters[position].add(letter)
            assignment_counts[(letter, position)] += 1
    
    # Calculate stability scores (lower = more stable)
    letter_stability = {letter: len(positions) for letter, positions in letter_positions.items()}
    position_stability = {position: len(letters) for position, letters in position_letters.items()}
    
    return letter_stability, position_stability, assignment_counts


def parse_filter_assignments(filter_string):
    """
    Parse filter string into letter-position constraints.
    
    Args:
        filter_string: String like "e:J,t:F,a:S" specifying letter:position constraints
        
    Returns:
        dict: Mapping of letter -> required position
    """
    if not filter_string:
        return {}
    
    constraints = {}
    for constraint in filter_string.split(','):
        constraint = constraint.strip()
        if ':' not in constraint:
            print(f"Warning: Invalid constraint format '{constraint}'. Use 'letter:position' format.")
            continue
        
        letter, position = constraint.split(':', 1)
        letter = letter.strip()
        position = position.strip()
        constraints[letter] = position
    
    return constraints


def create_general_filter(constraints):
    """
    Create a filter function based on letter-position constraints.
    
    Args:
        constraints: dict mapping letter -> required position
        
    Returns:
        function that takes a DataFrame row and returns True if all constraints are met
    """
    def filter_function(row):
        """Filter function to keep only solutions that meet all constraints."""
        if not constraints:
            return True
            
        items = list(row['items'])
        positions = list(row['positions'])
        
        for letter, required_position in constraints.items():
            try:
                letter_index = items.index(letter)
                actual_position = positions[letter_index]
                if actual_position != required_position:
                    return False
            except ValueError:
                # Letter not found in this solution
                return False
        
        return True
    
    return filter_function


def format_constraints_string(constraints):
    """Format constraints dict into a readable string."""
    if not constraints:
        return "No constraints"
    return ", ".join([f"'{letter}' in '{pos}'" for letter, pos in constraints.items()])


def plot_pareto_front(df, item_col, pair_col, output_dir):
    """Create the Pareto front visualization."""
    # Check if we have source file information
    has_source_files = 'source_file' in df.columns
    
    if has_source_files:
        # Get file counts for legend
        file_counts = df['source_file'].value_counts()
        unique_sources = df['source_file'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sources)))
        source_color_map = dict(zip(unique_sources, colors))
        point_colors = [source_color_map[source] for source in df['source_file']]
        
        # Create colored plot with legend
        plt.figure(figsize=(12, 8))  # Original aspect ratio
        plt.scatter(df[item_col], df[pair_col], alpha=0.3, s=50, c=point_colors)
        
        # Add legend for source files (show top 10 most frequent)
        top_sources = file_counts.head(10).index
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=source_color_map[source], 
                                    markersize=8, label=f'{source} ({file_counts[source]})')
                         for source in top_sources]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.xlabel(item_col)
        plt.ylabel(pair_col)
        plt.title('Global Pareto Front - Colored by Source File')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'pareto_front_2d_colored.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create grayscale duplicate without legend
        plt.figure(figsize=(8, 6))  # Original aspect ratio
        plt.scatter(df[item_col], df[pair_col], alpha=0.3, s=50, color='gray')
        
        plt.xlabel(item_col)
        plt.ylabel(pair_col)
        plt.title('Global Pareto Front')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'pareto_front_2d_grayscale.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    else:
        # Single plot when no source file info
        plt.figure(figsize=(8, 6))  # Original aspect ratio
        plt.scatter(df[item_col], df[pair_col], alpha=0.3, s=50, color='blue')
        
        plt.xlabel(item_col)
        plt.ylabel(pair_col)
        plt.title('Global Pareto Front')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'pareto_front_2d.png', dpi=300, bbox_inches='tight')
        plt.show()
        file_counts = None
    
    return file_counts


def plot_source_distribution(df, output_dir):
    """Create source file distribution plot."""
    if 'source_file' not in df.columns:
        print("Warning: No source_file column found, skipping source distribution plot")
        return None
    
    file_counts = df['source_file'].value_counts()
    plt.figure(figsize=(12, 6))
    top_sources = file_counts.head(20)
    plt.bar(range(len(top_sources)), top_sources.values)
    plt.xlabel('Source File Rank')
    plt.ylabel('Number of Pareto Solutions')
    plt.title('Distribution of Pareto Solutions by Source File (Top 20)')
    plt.xticks(range(len(top_sources)), [f'#{i+1}' for i in range(len(top_sources))])
    plt.tight_layout()
    plt.savefig(output_dir / 'source_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return file_counts


def print_statistics(df, item_col, pair_col, file_counts):
    """Print analysis statistics."""
    print("\n=== Global Pareto Analysis ===")
    print(f"Number of solutions: {len(df)}")
    
    if file_counts is not None:
        print(f"Number of source files: {len(file_counts)}")
        print(f"Most productive source: {file_counts.index[0]} ({file_counts.iloc[0]} solutions)")
    
    print(f"\nObjective Ranges:")
    for obj in [item_col, pair_col]:
        values = df[obj]
        print(f"  {obj}: {values.min():.6f} to {values.max():.6f} (range: {values.max()-values.min():.6f})")
    
    if 'global_rank' in df.columns:
        print(f"Global rank range: {df['global_rank'].min()} to {df['global_rank'].max()}")
    
    # Top 10 solutions
    print(f"\nTop 10 Solutions by Global Rank:")
    display_cols = []
    if 'global_rank' in df.columns:
        display_cols.append('global_rank')
    if 'item_rank' in df.columns:
        display_cols.append('item_rank')
    if 'pair_rank' in df.columns:
        display_cols.append('pair_rank')
    display_cols.extend([item_col, pair_col])
    if 'source_rank' in df.columns:
        display_cols.append('source_rank')
    if 'positions' in df.columns:
        display_cols.append('positions')
    
    top_solutions = df.head(10)[display_cols]
    print(top_solutions.to_string(index=False))
    
    return top_solutions

def plot_stability_matrix(df, output_dir, filter_condition=None, title_suffix="", filename_suffix=""):
    """
    Create a stability matrix heatmap showing letter-position assignment frequencies.
    
    Args:
        df: DataFrame with solutions
        output_dir: Output directory for plots
        filter_condition: Optional function to filter solutions (deprecated - filtering should be done before calling)
        title_suffix: Additional text for plot title
        filename_suffix: Additional text for filename
    """
    # Apply filter if provided (for backward compatibility, but should be done externally now)
    if filter_condition:
        filtered_df = df[df.apply(filter_condition, axis=1)].copy()
        print(f"Filtered from {len(df)} to {len(filtered_df)} solutions")
    else:
        filtered_df = df.copy()
    
    if len(filtered_df) == 0:
        print("Warning: No solutions remain after filtering")
        return
    
    # Calculate stability metrics
    letter_stability, position_stability, assignment_counts = calculate_stability_metrics(filtered_df)
    
    # Sort letters and positions by stability (most stable first)
    letters_by_stability = sorted(letter_stability.keys(), key=lambda x: letter_stability[x])
    positions_by_stability = sorted(position_stability.keys(), key=lambda x: position_stability[x])
    
    print(f"Most stable letters: {letters_by_stability[:6]}")
    print(f"Most stable positions: {positions_by_stability[:6]}")
    
    # Create matrix data
    letter_indices = {letter: i for i, letter in enumerate(letters_by_stability)}
    position_indices = {pos: i for i, pos in enumerate(positions_by_stability)}
    
    # Initialize matrix with zeros
    matrix = np.zeros((len(letters_by_stability), len(positions_by_stability)))
    
    # Fill matrix with assignment counts
    for (letter, position), count in assignment_counts.items():
        if letter in letter_indices and position in position_indices:
            matrix[letter_indices[letter], position_indices[position]] = count
    
    # Create the heatmap
    plt.figure(figsize=(16, 12))
    
    # Create annotations matrix (empty strings for zeros, values for non-zeros)
    annot_matrix = np.where(matrix == 0, '', matrix.astype(int).astype(str))
    
    # Create heatmap with annotations for non-zero values only
    sns.heatmap(matrix, 
                xticklabels=positions_by_stability,  # Clean labels without stability scores
                yticklabels=letters_by_stability,    # Clean labels without stability scores
                cmap='Reds',
                linewidths=0.5,
                linecolor='white',
                square=True,
                cbar_kws={'label': 'Assignment Count', 'shrink': 0.5},  # Smaller colorbar
                fmt='',  # No formatting since we're providing custom annotations
                annot=annot_matrix,  # Custom annotations (empty for zeros)
                annot_kws={'size': 8})
    
    # Remove ticks
    plt.tick_params(left=False, bottom=False)
    
    # Customize the plot
    plt.title(f'Letter-Position Stability Matrix{title_suffix}\n'
              f'({len(filtered_df)} solutions, axes ordered by stability)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Positions (ordered by stability)', fontweight='bold')
    plt.ylabel('Letters (ordered by stability)', fontweight='bold')
    
    # Keep labels horizontal (no rotation)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save the plot
    filename = f'stability_matrix{filename_suffix}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print stability statistics
    print(f"\nStability Statistics{title_suffix}:")
    print(f"Most stable letters (fewest positions):")
    for letter in letters_by_stability[:8]:
        print(f"  {letter}: {letter_stability[letter]} unique positions")
    
    print(f"Most stable positions (fewest letters):")
    for position in positions_by_stability[:8]:
        print(f"  {position}: {position_stability[position]} unique letters")
    
    # Calculate sparsity
    total_cells = len(letters_by_stability) * len(positions_by_stability)
    non_zero_cells = np.count_nonzero(matrix)
    sparsity = (total_cells - non_zero_cells) / total_cells * 100
    print(f"Matrix sparsity: {sparsity:.1f}% ({non_zero_cells}/{total_cells} cells have assignments)")
    
    return letter_stability, position_stability, assignment_counts

    
def save_results(df, file_counts, top_solutions, output_dir, filter_constraints=None):
    """Save analysis results to files."""
    # Save summary statistics
    with open(output_dir / 'analysis_summary.txt', 'w') as f:
        f.write(f"Global Pareto Analysis Summary\n")
        f.write(f"==============================\n\n")
        f.write(f"Number of solutions: {len(df)}\n")
        if file_counts is not None:
            f.write(f"Number of source files: {len(file_counts)}\n")
        if filter_constraints:
            f.write(f"Filter constraints: {format_constraints_string(filter_constraints)}\n")
        if 'global_rank' in df.columns:
            f.write(f"Global rank range: {df['global_rank'].min()} to {df['global_rank'].max()}\n")
        f.write(f"\nTop 10 Solutions by Global Rank:\n")
        f.write(top_solutions.to_string(index=False))
    
    # Save full results with rankings to CSV
    if 'global_rank' in df.columns:
        output_csv = output_dir / 'global_moo_solutions_with_ranks.csv'
        df.to_csv(output_csv, index=False)
        print(f"\nFull results with rankings saved to: {output_csv}")
    """Save analysis results to files."""
    # Save summary statistics
    with open(output_dir / 'analysis_summary.txt', 'w') as f:
        f.write(f"Global Pareto Analysis Summary\n")
        f.write(f"==============================\n\n")
        f.write(f"Number of solutions: {len(df)}\n")
        if file_counts is not None:
            f.write(f"Number of source files: {len(file_counts)}\n")
        if filter_constraints:
            f.write(f"Filter constraints: {format_constraints_string(filter_constraints)}\n")
        if 'global_rank' in df.columns:
            f.write(f"Global rank range: {df['global_rank'].min()} to {df['global_rank'].max()}\n")
        f.write(f"\nTop 10 Solutions by Global Rank:\n")
        f.write(top_solutions.to_string(index=False))
    
    # Save full results with rankings to CSV
    if 'global_rank' in df.columns:
        output_csv = output_dir / 'global_moo_solutions_with_ranks.csv'
        df.to_csv(output_csv, index=False)
        print(f"\nFull results with rankings saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description='Analyze global Pareto results')
    parser.add_argument('pareto_file', help='Path to global Pareto CSV file')
    parser.add_argument('--output-dir', default='output', 
                       help='Directory for output plots and files')
    parser.add_argument('--filter-assignments', type=str,
                       help='Filter solutions by letter-position assignments. Format: "letter:position,letter:position" (e.g., "e:J,t:F,a:S")')
    
    args = parser.parse_args()
    
    # Parse filter constraints
    filter_constraints = parse_filter_assignments(args.filter_assignments) if args.filter_assignments else {}
    if filter_constraints:
        print(f"Using filter constraints: {format_constraints_string(filter_constraints)}")
    
    # Load data
    df = parse_global_pareto_csv(args.pareto_file)
    print(f"Loaded {len(df)} global Pareto solutions")
    
    # Detect objective columns
    item_col, pair_col = detect_objective_columns(df)
    if item_col is None or pair_col is None:
        return
    
    print(f"Using objectives: {item_col} and {pair_col}")
    
    # Add ranking columns
    df = add_ranking_columns(df, item_col, pair_col)
    print(f"Added ranking columns and sorted by global rank")
    print(f"Global rank range: {df['global_rank'].min()} to {df['global_rank'].max()}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Apply filtering to the main dataframe if constraints provided
    if filter_constraints:
        filter_function = create_general_filter(filter_constraints)
        original_count = len(df)
        df = df[df.apply(filter_function, axis=1)].copy()
        print(f"Applied filter constraints: {format_constraints_string(filter_constraints)}")
        print(f"Filtered from {original_count} to {len(df)} solutions")
        
        if len(df) == 0:
            print("Error: No solutions remain after filtering. Exiting.")
            return
    
    # Generate plots and collect statistics
    file_counts = plot_pareto_front(df, item_col, pair_col, output_dir)
    if 'source_file' in df.columns:
        plot_source_distribution(df, output_dir)
    
    # Create stability matrix (now using the already-filtered dataframe)
    print(f"\n=== Creating Stability Matrix ===")
    if filter_constraints:
        constraint_desc = format_constraints_string(filter_constraints)
        plot_stability_matrix(df, output_dir, filter_condition=None, 
                            title_suffix=f" ({constraint_desc})", 
                            filename_suffix="_filtered")
    else:
        plot_stability_matrix(df, output_dir, filter_condition=None, 
                            title_suffix=" (All Solutions)",
                            filename_suffix="")
    
    # Print and save statistics
    top_solutions = print_statistics(df, item_col, pair_col, file_counts)
    save_results(df, file_counts, top_solutions, output_dir, filter_constraints)
    

    # Define additional characters to append to each MOO solution
    # CUSTOMIZE THESE AS NEEDED:
    extra_items = "qz'\",.-?"        # Characters to add to items
    extra_positions = "['TYGHBN"     # Positions to add to positions
    
    print(f"\nExtra items: {extra_items}/")
    print(f"Extra positions: {extra_positions}/")    
    print(f"\nAnalysis complete! Plots and files saved to {output_dir}/")
    print(f"Key outputs:")
    print(f"  - global_moo_solutions_with_ranks.csv: Complete results with rankings")
    if 'source_file' in df.columns:
        print(f"  - pareto_front_2d_colored.png: Pareto front colored by source file (with legend)")
        print(f"  - pareto_front_2d_grayscale.png: Pareto front in grayscale (clean version)")
        print(f"  - source_distribution.png: Source file distribution")
    else:
        print(f"  - pareto_front_2d.png: Pareto front visualization") 
    print(f"  - objective_correlation.png: Correlation analysis")
    print(f"  - stability_matrix.png: Letter-position assignment heatmap")
    if filter_constraints:
        constraint_desc = format_constraints_string(filter_constraints)
        print(f"  - stability_matrix_filtered.png: Filtered heatmap ({constraint_desc})")
    print(f"  - analysis_summary.txt: Text summary")


if __name__ == '__main__':
    main()
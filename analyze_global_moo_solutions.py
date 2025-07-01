#!/usr/bin/env python3
"""
Quick analysis and visualization of global Pareto results.

Usage: ``python3 analyze_global_moo_solutions.py output/global_moo_solutions.csv``
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from io import StringIO


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
    """Add ranking columns to the dataframe."""
    # Rank by each objective (higher scores get better ranks, i.e., rank 1 = highest score)
    df['item_rank'] = df[item_col].rank(ascending=False, method='min').astype(int)
    df['pair_rank'] = df[pair_col].rank(ascending=False, method='min').astype(int)
    df['global_rank'] = df['item_rank'] + df['pair_rank']
    
    # Sort by global rank (lower is better since it's sum of ranks)
    df = df.sort_values('global_rank').reset_index(drop=True)
    
    return df


def plot_pareto_front(df, item_col, pair_col, output_dir):
    """Create the Pareto front visualization."""
    plt.figure(figsize=(12, 8))
    
    # Check if we have source file information
    has_source_files = 'source_file' in df.columns
    
    if has_source_files:
        # Color by source file
        unique_sources = df['source_file'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sources)))
        source_color_map = dict(zip(unique_sources, colors))
        point_colors = [source_color_map[source] for source in df['source_file']]
        
        plt.scatter(df[item_col], df[pair_col], alpha=0.7, s=50, c=point_colors)
        
        # Add legend for source files (show top 10 most frequent)
        file_counts = df['source_file'].value_counts()
        top_sources = file_counts.head(10).index
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=source_color_map[source], 
                                    markersize=8, label=f'{source} ({file_counts[source]})')
                         for source in top_sources]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(df[item_col], df[pair_col], alpha=0.7, s=50, color='blue')
    
    plt.xlabel(item_col)
    plt.ylabel(pair_col)
    plt.title('Global Pareto Front - Colored by Source File' if has_source_files else 'Global Pareto Front')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_front_2d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return file_counts if has_source_files else None


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


def plot_correlation(df, item_col, pair_col, output_dir):
    """Create objective correlation plot."""
    plt.figure(figsize=(8, 6))
    correlation = df[item_col].corr(df[pair_col])
    plt.scatter(df[item_col], df[pair_col], alpha=0.6)
    plt.xlabel(item_col)
    plt.ylabel(pair_col)
    plt.title(f'Objective Correlation (r = {correlation:.3f})')
    
    # Add trend line
    z = np.polyfit(df[item_col], df[pair_col], 1)
    p = np.poly1d(z)
    plt.plot(df[item_col], p(df[item_col]), "r--", alpha=0.8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'objective_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlation


def print_statistics(df, item_col, pair_col, file_counts, correlation):
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
    
    print(f"\nObjective Correlation: {correlation:.4f}")
    
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
    if 'positions' in df.columns:
        display_cols.append('positions')
    
    top_solutions = df.head(10)[display_cols]
    print(top_solutions.to_string(index=False))
    
    return top_solutions


def save_results(df, file_counts, correlation, top_solutions, output_dir):
    """Save analysis results to files."""
    # Save summary statistics
    with open(output_dir / 'analysis_summary.txt', 'w') as f:
        f.write(f"Global Pareto Analysis Summary\n")
        f.write(f"==============================\n\n")
        f.write(f"Number of solutions: {len(df)}\n")
        if file_counts is not None:
            f.write(f"Number of source files: {len(file_counts)}\n")
        f.write(f"Objective correlation: {correlation:.4f}\n")
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
                       help='Directory for output plots')
    
    args = parser.parse_args()
    
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
    
    # Generate plots and collect statistics
    file_counts = plot_pareto_front(df, item_col, pair_col, output_dir)
    plot_source_distribution(df, output_dir)
    correlation = plot_correlation(df, item_col, pair_col, output_dir)
    
    # Print and save statistics
    top_solutions = print_statistics(df, item_col, pair_col, file_counts, correlation)
    save_results(df, file_counts, correlation, top_solutions, output_dir)
    
    print(f"\nAnalysis complete! Plots and summary saved to {output_dir}/")


if __name__ == '__main__':
    main()
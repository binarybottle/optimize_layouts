#!/usr/bin/env python3
"""
Analyze Layout Optimization Results and Create Visualizations

This script analyzes keyboard layout optimization results and creates scatter plots
of scores. Updated for the refactored MOO system using optimize_moo.py output format.

Analysis Features:
- Supports new MOO result CSV format from optimize_moo.py
- Creates scatter plots and correlation analysis 
- Handles multiple objective scoring
- Generates summary statistics and best layout identification

Plot Types Generated:
- objective_scores_scatter.png (all objectives vs rank)
- objective_correlation.png (correlation matrix if multiple objectives)
- best_layouts_summary.png (top solutions visualization)

Usage:
    python3 analyze_results.py [options]

Examples:
    python3 analyze_results.py
    python3 analyze_results.py --results-dir output/layouts
    python3 analyze_results.py --file-pattern "moo_results_*.csv"
    python3 analyze_results.py --max-files 100 --debug
"""
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import seaborn as sns

# Default combination strategy for backwards compatibility
DEFAULT_COMBINATION_STRATEGY = "geometric_mean"

def apply_default_combination_vectorized(obj1_scores: np.ndarray, obj2_scores: np.ndarray) -> np.ndarray:
    """Apply default combination strategy to two objective arrays."""
    if DEFAULT_COMBINATION_STRATEGY == "geometric_mean":
        return np.sqrt(obj1_scores * obj2_scores)
    elif DEFAULT_COMBINATION_STRATEGY == "arithmetic_mean":
        return (obj1_scores + obj2_scores) / 2.0
    else:
        # Fallback to geometric mean
        return np.sqrt(obj1_scores * obj2_scores)

#-----------------------------------------------------------------------------
# File processing utilities - updated for new MOO format
#-----------------------------------------------------------------------------
def process_files_batch(results_dir, file_pattern="moo_results_*.csv", max_files=None):
    """Find MOO result files."""
    pattern_path = f"{results_dir}/{file_pattern}"
    files = glob.glob(pattern_path)
    
    if max_files:
        files = files[:max_files]
    
    if not files:
        print(f"No CSV files found matching pattern: {pattern_path}")
        return []
    
    print(f"Found {len(files)} files matching pattern: {pattern_path}")
    return files

def parse_moo_result_csv(filepath, debug=False):
    """
    Parse a MOO results CSV file from optimize_moo.py.
    
    The new format has:
    - Metadata header section
    - Data section with columns: rank, items, positions, layout, [objective_columns], combined_score
    
    Args:
        filepath: Path to CSV file
        debug: Enable debug output
        
    Returns:
        List of result dictionaries or None if parsing fails
    """
    try:
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return None
        
        # Read the entire file to handle the new format
        df = pd.read_csv(filepath, skiprows=0)
        
        # The new MOO format should have these columns
        required_columns = ['rank', 'items', 'positions']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            if debug:
                print(f"Missing required columns in {filepath}: {missing}")
                print(f"Available columns: {list(df.columns)}")
            return None
        
        # Convert to list of dictionaries
        results = []
        config_id = os.path.basename(filepath).replace('moo_results_config_', '').replace('.csv', '')
        if '_' in config_id:
            config_id = config_id.split('_')[0]  # Take first part before timestamp
        
        for _, row in df.iterrows():
            # Find objective columns (exclude metadata columns)
            metadata_cols = ['rank', 'items', 'positions', 'layout', 'combined_score']
            objective_cols = [col for col in df.columns if col not in metadata_cols]
            
            result = {
                'config_id': config_id,
                'rank': int(row['rank']) if 'rank' in row else 0,
                'items': str(row['items']),
                'positions': str(row['positions']),
                'layout': row.get('layout', f"{row['items']} -> {row['positions']}"),
                'objectives': {},
                'combined_score': float(row.get('combined_score', 0.0))
            }
            
            # Add all objective scores
            for obj_col in objective_cols:
                if pd.notna(row[obj_col]):
                    result['objectives'][obj_col] = float(row[obj_col])
            
            results.append(result)
        
        return results
        
    except Exception as e:
        if debug:
            print(f"Error parsing {filepath}: {e}")
        return None

#-----------------------------------------------------------------------------
# Data loading
#-----------------------------------------------------------------------------
def load_moo_results(results_dir, file_pattern="moo_results_*.csv", max_files=None, debug=False):
    """Load MOO result files and return a dataframe."""
    files = process_files_batch(results_dir, file_pattern, max_files)
    if not files:
        return pd.DataFrame()
    
    all_results = []
    successful_files = 0
    
    for i, filepath in enumerate(files):
        if i % 10 == 0:
            print(f"Processing file {i+1}/{len(files)}: {os.path.basename(filepath)}")
        
        results = parse_moo_result_csv(filepath, debug=debug)
        if results:
            all_results.extend(results)
            successful_files += 1
    
    print(f"Successfully parsed {successful_files}/{len(files)} files, yielding {len(all_results)} results")
    
    if not all_results:
        return pd.DataFrame()
    
    # Convert to DataFrame and flatten objectives
    rows = []
    for result in all_results:
        row = {
            'config_id': result['config_id'],
            'rank': result['rank'],
            'items': result['items'],
            'positions': result['positions'],
            'layout': result['layout'],
            'combined_score': result['combined_score']
        }
        
        # Add objective columns
        for obj_name, obj_score in result['objectives'].items():
            row[obj_name] = obj_score
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

#-----------------------------------------------------------------------------
# Plotting functions - updated for multiple objectives
#-----------------------------------------------------------------------------
def plot_objective_scores(df, output_dir=".", save_path=None):
    """Create scatter plot of all objective scores."""
    if df.empty:
        print("No results to plot!")
        return
    
    # Find objective columns (exclude metadata)
    metadata_cols = ['config_id', 'rank', 'items', 'positions', 'layout', 'combined_score']
    objective_cols = [col for col in df.columns if col not in metadata_cols]
    
    if not objective_cols:
        print("No objective columns found!")
        return
    
    # Sort by combined score
    df_sorted = df.sort_values('combined_score', ascending=False)
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(objective_cols)))
    
    x_positions = range(len(df_sorted))
    
    for i, obj_col in enumerate(objective_cols):
        if obj_col in df_sorted.columns:
            plt.scatter(x_positions, df_sorted[obj_col], 
                       marker='.', s=30, alpha=0.7, 
                       label=obj_col, color=colors[i], edgecolors='none')
    
    plt.xlabel('Layout Index (sorted by combined score)')
    plt.ylabel('Objective Score')
    plt.title(f'Multi-Objective Scores ({len(df_sorted)} layouts)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = []
    for obj_col in objective_cols:
        if obj_col in df.columns:
            values = df[obj_col].dropna()
            stats_text.append(f"{obj_col}: {values.min():.4f} to {values.max():.4f}")
    
    plt.figtext(0.02, 0.02, '\n'.join(stats_text), fontsize=9)
    
    # Save plot
    if save_path is None:
        save_path = os.path.join(output_dir, 'objective_scores_scatter.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Generated objective scores plot: {save_path}")
    plt.close()

def plot_objective_correlation(df, output_dir=".", save_path=None):
    """Create correlation matrix of objectives."""
    if df.empty:
        return
    
    # Find objective columns
    metadata_cols = ['config_id', 'rank', 'items', 'positions', 'layout', 'combined_score']
    objective_cols = [col for col in df.columns if col not in metadata_cols]
    
    if len(objective_cols) < 2:
        print("Need at least 2 objectives for correlation analysis")
        return
    
    # Calculate correlation matrix
    obj_data = df[objective_cols].dropna()
    if obj_data.empty:
        print("No valid objective data for correlation")
        return
    
    corr_matrix = obj_data.corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title(f'Objective Correlation Matrix ({len(obj_data)} layouts)')
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(output_dir, 'objective_correlation.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Generated correlation plot: {save_path}")
    plt.close()

def plot_best_layouts(df, output_dir=".", save_path=None, top_n=10):
    """Visualize top layouts."""
    if df.empty or len(df) < top_n:
        return
    
    # Get top layouts by combined score
    top_layouts = df.nlargest(top_n, 'combined_score')
    
    # Find objective columns
    metadata_cols = ['config_id', 'rank', 'items', 'positions', 'layout', 'combined_score']
    objective_cols = [col for col in df.columns if col not in metadata_cols]
    
    if not objective_cols:
        return
    
    # Create subplot for each objective
    fig, axes = plt.subplots(len(objective_cols), 1, figsize=(12, 4 * len(objective_cols)))
    if len(objective_cols) == 1:
        axes = [axes]
    
    for i, obj_col in enumerate(objective_cols):
        ax = axes[i]
        
        # Bar plot of top solutions for this objective
        y_pos = range(len(top_layouts))
        scores = top_layouts[obj_col].values
        
        bars = ax.barh(y_pos, scores, alpha=0.7, color=plt.cm.Set1(i / len(objective_cols)))
        
        # Add layout labels
        labels = [f"{row['items']} → {row['positions']}" for _, row in top_layouts.iterrows()]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        
        ax.set_xlabel(f'{obj_col} Score')
        ax.set_title(f'Top {top_n} Layouts by {obj_col}')
        ax.grid(True, alpha=0.3)
        
        # Add score values on bars
        for j, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + max(scores) * 0.01, j, f'{score:.4f}', 
                   va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(output_dir, 'best_layouts_summary.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Generated best layouts plot: {save_path}")
    plt.close()

#-----------------------------------------------------------------------------
# Analysis functions
#-----------------------------------------------------------------------------
def analyze_moo_results(df):
    """Print comprehensive analysis of MOO results."""
    if df.empty:
        print("No results to analyze!")
        return
    
    print(f"\nAnalyzed {len(df)} MOO layout results")
    
    # Find objective columns
    metadata_cols = ['config_id', 'rank', 'items', 'positions', 'layout', 'combined_score']
    objective_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Objective statistics
    if objective_cols:
        print(f"\nObjective Statistics:")
        for obj_col in objective_cols:
            if obj_col in df.columns:
                values = df[obj_col].dropna()
                print(f"  {obj_col}: {values.min():.6f} to {values.max():.6f} (mean: {values.mean():.6f})")
        
        # Combined score stats
        if 'combined_score' in df.columns:
            combined = df['combined_score'].dropna()
            print(f"  Combined Score: {combined.min():.6f} to {combined.max():.6f} (mean: {combined.mean():.6f})")
        
        # Correlations between objectives
        if len(objective_cols) > 1:
            print(f"\nObjective Correlations:")
            obj_data = df[objective_cols].dropna()
            corr_matrix = obj_data.corr()
            
            for i in range(len(objective_cols)):
                for j in range(i + 1, len(objective_cols)):
                    obj1, obj2 = objective_cols[i], objective_cols[j]
                    corr = corr_matrix.loc[obj1, obj2]
                    print(f"  {obj1} vs {obj2}: {corr:.4f}")
    
    # Best layouts
    print(f"\nTop 5 Layouts by Combined Score:")
    if 'combined_score' in df.columns:
        top_5 = df.nlargest(5, 'combined_score')
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            obj_scores = []
            for obj_col in objective_cols:
                if obj_col in row and pd.notna(row[obj_col]):
                    obj_scores.append(f"{obj_col}: {row[obj_col]:.4f}")
            
            obj_str = ", ".join(obj_scores) if obj_scores else "No objectives"
            print(f"  {i}. {row['items']} → {row['positions']} | Combined: {row['combined_score']:.4f} | {obj_str}")

def save_results_summary(df, output_dir="."):
    """Save results to CSV with proper formatting."""
    if df.empty:
        return
    
    # Sort by combined score
    df_sorted = df.sort_values('combined_score', ascending=False)
    
    try:
        output_path = os.path.join(output_dir, "moo_results_summary.xlsx")
        df_sorted.to_excel(output_path, index=False)
        print(f"Results saved to {output_path}")
    except Exception:
        output_path = os.path.join(output_dir, "moo_results_summary.csv")
        df_sorted.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

#-----------------------------------------------------------------------------
# Main function
#-----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Analyze MOO layout optimization results')
    parser.add_argument('--results-dir', type=str, default='output/layouts', 
                       help='Directory containing result files')
    parser.add_argument('--file-pattern', type=str, default='moo_results_*.csv', 
                       help='File pattern to match')
    parser.add_argument('--max-files', type=int, default=None, 
                       help='Maximum number of files to process')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output')
    parser.add_argument('--output-dir', type=str, default='.', 
                       help='Output directory for plots and summaries')
    
    args = parser.parse_args()
    
    print(f"MOO Results Analysis Starting...")
    print(f"Results directory: {args.results_dir}")
    print(f"File pattern: {args.file_pattern}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max files: {args.max_files or 'all'}")
    
    try:
        # Check directory exists
        if not os.path.exists(args.results_dir):
            print(f"ERROR: Directory '{args.results_dir}' does not exist!")
            return 1
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load results
        print("Loading MOO results...")
        df = load_moo_results(args.results_dir, args.file_pattern, args.max_files, args.debug)
        
        if df.empty:
            print("No valid results found!")
            return 1
        
        print(f"Loaded {len(df)} MOO layout results")
        
        # Generate plots
        print("Generating visualizations...")
        plot_objective_scores(df, args.output_dir)
        plot_objective_correlation(df, args.output_dir)
        plot_best_layouts(df, args.output_dir)
        
        # Analysis and export
        analyze_moo_results(df)
        save_results_summary(df, args.output_dir)
        
        print(f"\nMOO analysis complete! Output saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
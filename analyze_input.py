#!/usr/bin/env python
"""
Compare Raw and Normalized Input Data

This script compares raw input data with its normalized version through visual plots.
It reads raw file paths from config.yaml and handles cases where raw files may not be available.

Usage:
    python compare_normalization.py --config config.yaml --norm-dir path/to/normalized/data --output-dir path/to/output/plots
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from datetime import datetime
from pathlib import Path

def load_config(config_path):
    """Load configuration from yaml file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None

def setup_paths(config_path, norm_dir, output_dir):
    """Set up all paths based on configuration and input directories."""
    # Load configuration
    config = load_config(config_path)
    if not config:
        print("Warning: Could not load config file. Will only analyze normalized data.")
        raw_paths = {}
    else:
        # Extract raw file paths from config
        try:
            raw_paths = {
                'item_scores': config['paths']['input']['raw_item_scores_file'],
                'item_pair_scores': config['paths']['input']['raw_item_pair_scores_file'],
                'position_scores': config['paths']['input']['raw_position_scores_file'],
                'position_pair_scores': config['paths']['input']['raw_position_pair_scores_file']
            }
            print(f"Successfully loaded file paths from config: {config_path}")
        except KeyError as e:
            print(f"Warning: Could not find all required paths in config file: {e}")
            raw_paths = {}
    
    # Standard filenames for normalized data
    norm_filenames = {
        'item_scores': 'normalized_item_scores.csv',
        'item_pair_scores': 'normalized_item_pair_scores.csv',
        'position_scores': 'normalized_position_scores.csv',
        'position_pair_scores': 'normalized_position_pair_scores.csv'
    }
    
    # Normalized file paths
    norm_paths = {
        'item_scores': os.path.join(norm_dir, norm_filenames['item_scores']),
        'item_pair_scores': os.path.join(norm_dir, norm_filenames['item_pair_scores']),
        'position_scores': os.path.join(norm_dir, norm_filenames['position_scores']),
        'position_pair_scores': os.path.join(norm_dir, norm_filenames['position_pair_scores'])
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Normalized data directory: {norm_dir}")
    print(f"Output plots directory: {output_dir}")
    
    # Check which raw files exist
    available_raw = {}
    for key, path in raw_paths.items():
        if os.path.exists(path):
            available_raw[key] = path
            print(f"Raw {key} file found: {path}")
        else:
            print(f"Raw {key} file not found: {path}")
    
    # Check which normalized files exist
    available_norm = {}
    for key, path in norm_paths.items():
        if os.path.exists(path):
            available_norm[key] = path
            print(f"Normalized {key} file found: {path}")
        else:
            print(f"Normalized {key} file not found: {path}")
    
    return available_raw, available_norm, output_dir

def safely_convert_to_uppercase(labels):
    """Safely convert labels to uppercase strings, handling various data types."""
    if isinstance(labels, np.ndarray):
        # Convert each element to string and then uppercase
        string_labels = np.array([str(label) for label in labels])
        return np.char.upper(string_labels)
    elif isinstance(labels, pd.Series):
        return labels.astype(str).str.upper()
    else:
        # If it's some other type, convert to list of strings
        return [str(label).upper() for label in labels]

def load_data(raw_paths, norm_paths):
    """Load raw and normalized data files."""
    print("\nLoading data files...")
    
    data = {}
    
    # Get all unique keys across both raw and normalized data
    all_keys = set(list(raw_paths.keys()) + list(norm_paths.keys()))
    
    # Load each data type
    for key in all_keys:
        data[key] = {'raw': {}, 'norm': {}}
        
        # Load raw data if available
        if key in raw_paths:
            try:
                print(f"Loading raw {key}...")
                raw_df = pd.read_csv(raw_paths[key])
                
                data[key]['raw']['df'] = raw_df
                data[key]['raw']['scores'] = raw_df['score'].values
                
                # Add labels based on data type
                if key == 'item_scores':
                    data[key]['raw']['labels'] = raw_df['item'].values
                elif key == 'position_scores':
                    data[key]['raw']['labels'] = safely_convert_to_uppercase(raw_df['position'].values)
                elif key == 'item_pair_scores':
                    data[key]['raw']['labels'] = raw_df['item_pair'].values
                elif key == 'position_pair_scores':
                    data[key]['raw']['labels'] = safely_convert_to_uppercase(raw_df['position_pair'].values)
                
                print(f"  - Raw: {len(raw_df)} rows, score range: {raw_df['score'].min():.6g} to {raw_df['score'].max():.6g}")
            except Exception as e:
                print(f"Error loading raw {key}: {e}")
                data[key]['raw'] = None
        else:
            print(f"Skipping raw {key} (file not available)")
            data[key]['raw'] = None
        
        # Load normalized data if available
        if key in norm_paths:
            try:
                print(f"Loading normalized {key}...")
                norm_df = pd.read_csv(norm_paths[key])
                
                data[key]['norm']['df'] = norm_df
                data[key]['norm']['scores'] = norm_df['normalized_score'].values
                
                # Add labels based on data type
                if key == 'item_scores':
                    data[key]['norm']['labels'] = norm_df['item'].values
                elif key == 'position_scores':
                    data[key]['norm']['labels'] = safely_convert_to_uppercase(norm_df['position'].values)
                elif key == 'item_pair_scores':
                    data[key]['norm']['labels'] = norm_df['item_pair'].values
                elif key == 'position_pair_scores':
                    data[key]['norm']['labels'] = safely_convert_to_uppercase(norm_df['position_pair'].values)
                
                print(f"  - Normalized: {len(norm_df)} rows, score range: {norm_df['normalized_score'].min():.6g} to {norm_df['normalized_score'].max():.6g}")
            except Exception as e:
                print(f"Error loading normalized {key}: {e}")
                data[key]['norm'] = None
        else:
            print(f"Skipping normalized {key} (file not available)")
            data[key]['norm'] = None
    
    return data

def get_title_for_key(key):
    """Return a properly formatted title for a given key."""
    if key == 'item_scores':
        return 'Item Frequency Scores'
    elif key == 'item_pair_scores':
        return 'Item Pair Frequency Scores'
    elif key == 'position_scores':
        return 'Position Comfort Scores'
    elif key == 'position_pair_scores':
        return 'Position Pair Comfort Scores'
    else:
        return key.replace("_", " ").title()

def plot_distribution_comparison(data, output_dir):
    """Create histogram plots comparing raw vs normalized distributions."""
    print("\nGenerating distribution comparison plots...")
    
    for key, dataset in data.items():
        # Skip if we don't have normalized data
        if dataset['norm'] is None:
            print(f"  - Skipping {key} (no normalized data)")
            continue
        
        print(f"  - Processing {key}...")
        
        # Check if we have raw data for comparison
        has_raw = dataset['raw'] is not None
        
        if has_raw:
            # Create figure with 2 subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f'Score Distribution: {get_title_for_key(key)}', fontsize=16)
            
            # Plot raw distribution
            sns.histplot(dataset['raw']['scores'], ax=ax1, kde=True, color='blue')
            ax1.set_title('Raw Data')
            ax1.set_xlabel('Score')
            ax1.set_ylabel('Count')
            
            # Use log scale if range is large
            score_range = dataset['raw']['scores'].max() / max(dataset['raw']['scores'].min(), 1e-10)
            if score_range > 1000:
                ax1.set_xscale('log')
                ax1.set_title('Raw Data (Log Scale)')
            
            # Plot normalized distribution
            sns.histplot(dataset['norm']['scores'], ax=ax2, kde=True, color='green')
            ax2.set_title('Normalized Data [0-1]')
            ax2.set_xlabel('Score')
            ax2.set_ylabel('Count')
        else:
            # Create figure with only normalized data
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle(f'Score Distribution: {get_title_for_key(key)}', fontsize=16)
            
            # Plot normalized distribution
            sns.histplot(dataset['norm']['scores'], ax=ax, kde=True, color='green')
            ax.set_title('Normalized Data [0-1]')
            ax.set_xlabel('Score')
            ax.set_ylabel('Count')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{key}_distribution_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved to: {output_path}")
        plt.close()

def plot_normalization_effect(data, output_dir):
    """Plot comparison showing how normalization affects the original data ordering."""
    print("\nGenerating normalization effect plots...")
    
    for key, dataset in data.items():
        # Skip if we don't have both raw and normalized data
        if dataset['raw'] is None or dataset['norm'] is None:
            print(f"  - Skipping {key} (missing raw or normalized data)")
            continue
        
        print(f"  - Processing {key}...")
        
        # Get data
        raw_scores = dataset['raw']['scores']
        norm_scores = dataset['norm']['scores']
        labels = dataset['raw']['labels']
        
        # Sort by raw score (descending)
        sort_idx = np.argsort(raw_scores)[::-1]
        sorted_raw = raw_scores[sort_idx]
        sorted_norm = norm_scores[sort_idx]
        sorted_labels = labels[sort_idx]
        
        # Limit to top 30 for readability
        n_show = min(30, len(sorted_raw))
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Create bar positions
        x = np.arange(n_show)
        width = 0.35
        
        # Create two y-axes
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plot raw scores on left y-axis
        ax1.bar(x - width/2, sorted_raw[:n_show], width, color='blue', alpha=0.7, label='Raw')
        ax1.set_ylabel('Raw Score', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Plot normalized scores on right y-axis
        ax2.bar(x + width/2, sorted_norm[:n_show], width, color='green', alpha=0.7, label='Normalized')
        ax2.set_ylabel('Normalized Score [0-1]', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Set y-axis limits for normalized scores
        ax2.set_ylim(0, 1.05)
        
        # Set title and x-axis
        plt.title(f'{get_title_for_key(key)} - Normalization Effect', fontsize=14)
        
        if key.startswith('position'):
            ax1.set_xlabel('PositionS (sorted by raw score)')
        else:
            ax1.set_xlabel('Items (sorted by raw score)')
        
        # Set x-ticks as labels if there aren't too many
        if n_show <= 30:
            ax1.set_xticks(x)
            if len(sorted_labels[0]) <= 2:  # Single characters or pairs
                ax1.set_xticklabels(sorted_labels[:n_show], rotation=90 if key.endswith('pair_scores') else 0)
            else:
                # For longer labels, show every other label
                ax1.set_xticks(x[::2])
                ax1.set_xticklabels(sorted_labels[:n_show:2], rotation=45)
        
        # Add a legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Optional: log scale for raw scores if range is large
        if raw_scores.max() / max(raw_scores.min(), 1e-10) > 1000:
            ax1.set_yscale('log')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{key}_normalization_effect.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved to: {output_path}")
        plt.close()

def plot_normalized_scores(data, output_dir):
    """Plot normalized scores distribution for each data type."""
    print("\nGenerating normalized score analysis plots...")
    
    for key, dataset in data.items():
        # Skip if we don't have normalized data
        if dataset['norm'] is None:
            print(f"  - Skipping {key} (no normalized data)")
            continue
            
        print(f"  - Processing {key}...")
        
        # Get data
        norm_scores = dataset['norm']['scores']
        labels = dataset['norm']['labels']
        
        # Sort by normalized score (descending)
        sort_idx = np.argsort(norm_scores)[::-1]
        sorted_norm = norm_scores[sort_idx]
        sorted_labels = labels[sort_idx]
        
        # Limit to top 30 for readability
        n_show = min(30, len(sorted_norm))
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        x = np.arange(n_show)
        plt.bar(x, sorted_norm[:n_show], color='green', alpha=0.7)
        
        # Add values on top of bars
        for i, val in enumerate(sorted_norm[:n_show]):
            plt.text(i, val + 0.02, f"{val:.3f}", ha='center', va='bottom', fontsize=8, rotation=90)
        
        # Set title and labels
        plt.title(f'Top {n_show} {get_title_for_key(key)} - Normalized', fontsize=14)
        plt.ylabel('Normalized Score [0-1]')
        
        if key.startswith('position'):
            plt.xlabel('PositionS')
        else:
            plt.xlabel('Items')
        
        # Set y-axis limits
        plt.ylim(0, 1.2)
        
        # Set x-ticks as labels
        plt.xticks(x, sorted_labels[:n_show], rotation=90 if key.endswith('pair_scores') else 45)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{key}_normalized_top{n_show}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved to: {output_path}")
        plt.close()

def create_summary_report(data, raw_paths, norm_paths, output_dir, config_path):
    """Create a summary report of the normalization process."""
    print("\nGenerating normalization summary report...")
    
    report_path = os.path.join(output_dir, "normalization_summary.txt")
    
    with open(report_path, 'w') as f:
        f.write("=== NORMALIZATION SUMMARY REPORT ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config file: {config_path}\n\n")
        
        f.write("=== DATA FILES ===\n")
        f.write("Raw Data Files:\n")
        if raw_paths:
            for key, path in raw_paths.items():
                f.write(f"  {key}: {path}\n")
        else:
            f.write("  No raw data files available or specified\n")
        
        f.write("\nNormalized Data Files:\n")
        for key, path in norm_paths.items():
            f.write(f"  {key}: {path}\n")
        
        f.write("\n=== NORMALIZATION STATISTICS ===\n")
        for key, dataset in data.items():
            title = get_title_for_key(key).upper()
            f.write(f"\n{title}:\n")
            
            # Raw statistics if available
            if dataset['raw'] is not None:
                raw_scores = dataset['raw']['scores']
                f.write(f"  Raw Scores:\n")
                f.write(f"    Min: {np.min(raw_scores):.6g}\n")
                f.write(f"    Max: {np.max(raw_scores):.6g}\n")
                f.write(f"    Mean: {np.mean(raw_scores):.6g}\n")
                f.write(f"    Median: {np.median(raw_scores):.6g}\n")
                f.write(f"    Standard Deviation: {np.std(raw_scores):.6g}\n")
            else:
                f.write("  Raw Scores: Not available\n")
            
            # Normalized statistics if available
            if dataset['norm'] is not None:
                norm_scores = dataset['norm']['scores']
                f.write(f"  Normalized Scores:\n")
                f.write(f"    Min: {np.min(norm_scores):.6g}\n")
                f.write(f"    Max: {np.max(norm_scores):.6g}\n")
                f.write(f"    Mean: {np.mean(norm_scores):.6g}\n")
                f.write(f"    Median: {np.median(norm_scores):.6g}\n")
                f.write(f"    Standard Deviation: {np.std(norm_scores):.6g}\n")
            else:
                f.write("  Normalized Scores: Not available\n")
        
        # Top items analysis
        f.write("\n=== TOP SCORES ANALYSIS ===\n")
        for key in data.keys():
            if key not in data or data[key]['norm'] is None:
                continue
                
            norm_scores = data[key]['norm']['scores']
            labels = data[key]['norm']['labels']
            
            # Get rankings
            norm_ranking = np.argsort(norm_scores)[::-1]
            
            # Get top 10
            norm_top10 = [labels[i] for i in norm_ranking[:10]]
            
            title = get_title_for_key(key).upper()
            f.write(f"\n{title} TOP 10:\n")
            
            if key.startswith('position'):
                f.write("  Rank  Position  Score\n")
            else:
                f.write("  Rank  Item      Score\n")
                
            f.write("  ----  --------  -----\n")
            
            for i, item in enumerate(norm_top10):
                if i < len(norm_ranking):
                    idx = norm_ranking[i]
                    f.write(f"  {i+1:4d}  {item:8}  {norm_scores[idx]:.4f}\n")
    
    print(f"Saved summary report to: {report_path}")

def main():
    """Main function to execute the script."""
    parser = argparse.ArgumentParser(description="Compare raw and normalized input data through plots")
    parser.add_argument("--config", type=str, default='config.yaml', help="Path to config.yaml file")
    parser.add_argument("--norm-dir", type=str, default='output/normalized_input', help="Directory containing normalized data files")
    parser.add_argument("--output-dir", type=str, default="output/normalized_input/plots", help="Directory to save output plots")
    
    args = parser.parse_args()
    
    # Setup paths
    raw_paths, norm_paths, output_dir = setup_paths(args.config, args.norm_dir, args.output_dir)
    
    # Load data
    data = load_data(raw_paths, norm_paths)
    
    # Generate plots
    plot_distribution_comparison(data, output_dir)
    plot_normalized_scores(data, output_dir)
    
    # Generate comparison plots (only if we have both raw and normalized data)
    has_comparison = any(dataset['raw'] is not None and dataset['norm'] is not None for dataset in data.values())
    if has_comparison:
        plot_normalization_effect(data, output_dir)
    else:
        print("\nSkipping comparison plots (raw data not available for any data type)")
    
    # Create summary report
    create_summary_report(data, raw_paths, norm_paths, output_dir, args.config)
    
    print(f"\nAll plots and reports generated successfully in: {output_dir}")

if __name__ == "__main__":
    main()


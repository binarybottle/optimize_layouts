#!/usr/bin/env python
"""
Compare raw and normalized input data; analyze item and position participation in pairs.

This script compares raw input data with its normalized version through visual
plots and statistical analysis, and analyzes item and position participation in pairs. 
It reads raw file paths from config.yaml.

Features:
- Loads both raw and normalized score data using config infrastructure
- Creates distribution comparison plots (raw vs normalized)
- Generates top scores visualizations for normalized data  
- Handles item scores, item-pair scores, position scores, and position-pair scores
- Produces summary reports with statistics
- Analyzes item and position participation in pairs

Input Files (via config.yaml):
- Raw data: item frequencies, item-pair frequencies, position scores, position-pair scores
- Normalized data: corresponding normalized CSV files

Output Files:
- Distribution comparison plots (*_distribution_comparison.png)
- Top scores plots (*_normalized_top*.png)  
- Summary report (normalization_summary.txt)
- Item pair participation analysis (item_pair_participation.csv)
- Position pair participation analysis (position_pair_participation.csv)

Usage:
    python analyze_input.py [--config config.yaml] [--output-dir output/normalized_input/plots]

Examples:
    python analyze_input.py
    python analyze_input.py --config config.yaml
    python analyze_input.py --output-dir output/normalized_input/plots
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

from config import load_config, Config
from optimize_layout import load_normalized_scores

#-----------------------------------------------------------------------------
# Shared utilities
#-----------------------------------------------------------------------------
def safely_convert_to_uppercase(labels):
    """Safely convert labels to uppercase strings, handling various data types."""
    if isinstance(labels, np.ndarray):
        string_labels = np.array([str(label) for label in labels])
        return np.char.upper(string_labels)
    elif isinstance(labels, pd.Series):
        return labels.astype(str).str.upper()
    else:
        return [str(label).upper() for label in labels]

def get_title_for_key(key):
    """Return a properly formatted title for a given key."""
    titles = {
        'item_scores': 'Item Frequency Scores',
        'item_pair_scores': 'Item Pair Frequency Scores',
        'position_scores': 'Position Comfort Scores',
        'position_pair_scores': 'Position Pair Comfort Scores'
    }
    return titles.get(key, key.replace("_", " ").title())

#-----------------------------------------------------------------------------
# Load raw data
#-----------------------------------------------------------------------------
def load_raw_scores(config: Config):
    """Load raw scores using the config structure."""
    def load_score_dict(filepath: str, key_col: str, score_col: str = 'score'):
        if not os.path.exists(filepath):
            return {}
        df = pd.read_csv(filepath)
        return {row[key_col].lower(): float(row[score_col]) for _, row in df.iterrows()}
    
    def load_pair_score_dict(filepath: str, pair_col: str, score_col: str = 'score'):
        if not os.path.exists(filepath):
            return {}
        df = pd.read_csv(filepath)
        result = {}
        for _, row in df.iterrows():
            pair_str = str(row[pair_col])
            if len(pair_str) == 2:
                key = (pair_str[0].lower(), pair_str[1].lower())
                result[key] = float(row[score_col])
        return result
    
    # Use config paths instead of manual path construction
    raw_data = {}
    if hasattr(config.paths, 'raw_item_scores_file'):
        raw_data['item_scores'] = load_score_dict(config.paths.raw_item_scores_file, 'item')
        raw_data['item_pair_scores'] = load_pair_score_dict(config.paths.raw_item_pair_scores_file, 'item_pair')
        raw_data['position_scores'] = load_score_dict(config.paths.raw_position_scores_file, 'position')
        raw_data['position_pair_scores'] = load_pair_score_dict(config.paths.raw_position_pair_scores_file, 'position_pair')
    
    return raw_data

#-----------------------------------------------------------------------------
# Pair participation analysis functions
#-----------------------------------------------------------------------------
def analyze_items_in_pairs(item_pair_scores):
    """Analyze item participation in item pairs."""
    if not item_pair_scores:
        return pd.DataFrame()
    
    # Get all unique items
    all_items = set()
    for pair in item_pair_scores.keys():
        all_items.add(pair[0])
        all_items.add(pair[1])
    
    # Get sorted scores for ranking (highest score = rank 1)
    all_scores = sorted(set(item_pair_scores.values()), reverse=True)
    score_to_rank = {score: rank + 1 for rank, score in enumerate(all_scores)}
    
    results = []
    total_pairs = len(item_pair_scores)
    
    for item in sorted(all_items):
        # Find all pairs containing this item
        item_pair_scores_list = []
        for pair, score in item_pair_scores.items():
            if item in pair:
                item_pair_scores_list.append(score)
        
        if item_pair_scores_list:
            count = len(item_pair_scores_list)
            percent = (count / total_pairs) * 100
            highest_score = max(item_pair_scores_list)
            lowest_score = min(item_pair_scores_list)
            highest_rank = score_to_rank[highest_score]
            lowest_rank = score_to_rank[lowest_score]
            
            results.append({
                'item': item,
                'pair_count': count,
                'pair_percent': round(percent, 2),
                'highest_pair_score': highest_score,
                'highest_pair_rank': highest_rank,
                'lowest_pair_score': lowest_score,
                'lowest_pair_rank': lowest_rank
            })
    
    return pd.DataFrame(results)

def analyze_positions_in_pairs(position_pair_scores):
    """Analyze position participation in position pairs."""
    if not position_pair_scores:
        return pd.DataFrame()
    
    # Get all unique positions
    all_positions = set()
    for pair in position_pair_scores.keys():
        all_positions.add(pair[0])
        all_positions.add(pair[1])
    
    # Get sorted scores for ranking (highest score = rank 1)
    all_scores = sorted(set(position_pair_scores.values()), reverse=True)
    score_to_rank = {score: rank + 1 for rank, score in enumerate(all_scores)}
    
    results = []
    total_pairs = len(position_pair_scores)
    
    for position in sorted(all_positions):
        # Find all pairs containing this position
        position_pair_scores_list = []
        for pair, score in position_pair_scores.items():
            if position in pair:
                position_pair_scores_list.append(score)
        
        if position_pair_scores_list:
            count = len(position_pair_scores_list)
            percent = (count / total_pairs) * 100
            highest_score = max(position_pair_scores_list)
            lowest_score = min(position_pair_scores_list)
            highest_rank = score_to_rank[highest_score]
            lowest_rank = score_to_rank[lowest_score]
            
            results.append({
                'position': position.upper(),  # Convert to uppercase for consistency
                'pair_count': count,
                'pair_percent': round(percent, 2),
                'highest_pair_score': highest_score,
                'highest_pair_rank': highest_rank,
                'lowest_pair_score': lowest_score,
                'lowest_pair_rank': lowest_rank
            })
    
    return pd.DataFrame(results)

def analyze_pair_participation(raw_data, output_dir):
    """Analyze how items and positions participate in pairs and save to CSV files."""
    print("Analyzing pair participation...")
    
    # Analyze items
    if 'item_pair_scores' in raw_data and raw_data['item_pair_scores']:
        print("  Analyzing item pair participation...")
        item_analysis = analyze_items_in_pairs(raw_data['item_pair_scores'])
        if not item_analysis.empty:
            item_output_path = os.path.join(output_dir, "item_pair_participation.csv")
            item_analysis.to_csv(item_output_path, index=False)
            print(f"  Saved: {item_output_path}")
        else:
            print("  No item pair data to analyze")
    
    # Analyze positions  
    if 'position_pair_scores' in raw_data and raw_data['position_pair_scores']:
        print("  Analyzing position pair participation...")
        position_analysis = analyze_positions_in_pairs(raw_data['position_pair_scores'])
        if not position_analysis.empty:
            position_output_path = os.path.join(output_dir, "position_pair_participation.csv")
            position_analysis.to_csv(position_output_path, index=False)
            print(f"  Saved: {position_output_path}")
        else:
            print("  No position pair data to analyze")

#-----------------------------------------------------------------------------
# Plotting functions
#-----------------------------------------------------------------------------
def create_comparison_plots(raw_data, normalized_data, config: Config, output_dir: str):
    """Create all comparison plots using existing data."""
    
    # Standard filenames for normalized data
    norm_files = {
        'item_scores': 'normalized_item_scores.csv',
        'item_pair_scores': 'normalized_item_pair_scores.csv', 
        'position_scores': 'normalized_position_scores.csv',
        'position_pair_scores': 'normalized_position_pair_scores.csv'
    }
    
    # Load normalized CSV files for comparison
    norm_dir = os.path.dirname(config.paths.item_scores_file)
    norm_csv_data = {}
    
    for key, filename in norm_files.items():
        filepath = os.path.join(norm_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            norm_csv_data[key] = df
    
    # Generate plots for each data type
    for key in norm_csv_data.keys():
        if key not in raw_data or not raw_data[key]:
            continue
            
        # Distribution comparison
        _plot_distribution_comparison(raw_data[key], norm_csv_data[key], key, output_dir)
        
        # Top scores analysis
        _plot_top_scores(norm_csv_data[key], key, output_dir)

def _plot_distribution_comparison(raw_scores, norm_df, key, output_dir):
    """Plot distribution comparison between raw and normalized."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Score Distribution: {get_title_for_key(key)}', fontsize=16)
    
    # Raw distribution
    raw_values = list(raw_scores.values())
    sns.histplot(raw_values, ax=ax1, kde=True, color='blue')
    ax1.set_title('Raw Data')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Count')
    
    # Use log scale if range is large
    if max(raw_values) / max(min(raw_values), 1e-10) > 1000:
        ax1.set_xscale('log')
        ax1.set_title('Raw Data (Log Scale)')
    
    # Normalized distribution
    sns.histplot(norm_df['score'], ax=ax2, kde=True, color='green')
    ax2.set_title('Normalized Data [0-1]')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{key}_distribution_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def _plot_top_scores(norm_df, key, output_dir, n_show=30):
    """Plot top normalized scores."""
    sorted_df = norm_df.sort_values('score', ascending=False).head(n_show)
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(sorted_df))
    plt.bar(x, sorted_df['score'], color='green', alpha=0.7)
    
    # Add values on bars
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        plt.text(i, row['score'] + 0.02, f"{row['score']:.3f}", 
                ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.title(f'Top {len(sorted_df)} {get_title_for_key(key)} - Normalized')
    plt.ylabel('Normalized Score [0-1]')
    plt.xlabel('Items' if not key.startswith('position') else 'Positions')
    plt.ylim(0, 1.2)
    
    # Set x-tick labels
    if key.startswith('position'):
        labels = safely_convert_to_uppercase(sorted_df.iloc[:, 0])  # First column
    else:
        labels = sorted_df.iloc[:, 0]  # First column
    
    plt.xticks(x, labels, rotation=90 if key.endswith('pair_scores') else 45)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{key}_normalized_top{len(sorted_df)}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_summary_report(raw_data, normalized_data, config: Config, output_dir: str):
    """Create summary report using existing data structures."""
    report_path = os.path.join(output_dir, "normalization_summary.txt")
    
    with open(report_path, 'w') as f:
        f.write("=== NORMALIZATION SUMMARY REPORT ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config file: {config._config_path}\n\n")
        
        # Write statistics for each data type
        for key in ['item_scores', 'item_pair_scores', 'position_scores', 'position_pair_scores']:
            f.write(f"\n{get_title_for_key(key).upper()}:\n")
            
            if key in raw_data and raw_data[key]:
                raw_values = list(raw_data[key].values())
                f.write(f"  Raw Scores: min={min(raw_values):.6g}, max={max(raw_values):.6g}, count={len(raw_values)}\n")
            else:
                f.write("  Raw Scores: Not available\n")
            
            # Normalized stats would come from the loaded CSVs
            f.write("  Normalized Scores: Available\n")
    
    print(f"Summary report saved: {report_path}")

#-----------------------------------------------------------------------------
# Main function
#-----------------------------------------------------------------------------
def main():
    """Streamlined main function using existing infrastructure."""
    parser = argparse.ArgumentParser(description="Compare raw and normalized input data")
    parser.add_argument("--config", type=str, default='config.yaml', help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="output/normalized_input/plots", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        # Use robust config loading from config.py
        print("Loading configuration...")
        config = load_config(args.config)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load data using existing functions
        print("Loading normalized scores...")
        normalized_scores = load_normalized_scores(config)  # Reuse existing function
        
        print("Loading raw scores...")
        raw_data = load_raw_scores(config)  # Simplified raw data loading
        
        # Generate plots and reports
        print("Generating comparison plots...")
        create_comparison_plots(raw_data, normalized_scores, config, args.output_dir)
        
        print("Creating summary report...")
        create_summary_report(raw_data, normalized_scores, config, args.output_dir)
        
        # New: Analyze pair participation
        analyze_pair_participation(raw_data, args.output_dir)
        
        print(f"\nAnalysis complete! Output saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
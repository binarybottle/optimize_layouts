#!/usr/bin/env python
"""
Compare Raw and Normalized Input Data (Standalone Version)

This script compares raw input data with its normalized version through visual
plots and statistical analysis.

Features:
- Creates distribution comparison plots (raw vs normalized)
- Generates top scores visualizations for normalized data  
- Produces summary reports with statistics
- Analyzes participation in pairs (if pair data is provided)

Usage:
    python analyze_raw_vs_norm.py --raw raw_data.csv --norm normalized_data.csv [--output-dir output]

Examples:
    python analyze_raw_vs_norm.py --raw item_pairs.csv --norm normalized_item_pairs.csv
    python analyze_raw_vs_norm.py --raw position_pairs.csv --norm normalized_position_pairs.csv --output-dir plots/
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Set random seed for reproducible plots
np.random.seed(42)

def detect_data_type(df):
    """Detect the type of data based on column names."""
    columns = [col.lower() for col in df.columns]
    
    if any('item_pair' in col for col in columns):
        return 'item_pair_scores'
    elif any('position_pair' in col for col in columns):
        return 'position_pair_scores'
    elif any('item' in col for col in columns):
        return 'item_scores'
    elif any('position' in col for col in columns):
        return 'position_scores'
    else:
        # Try to infer from data patterns
        if len(df.columns) >= 2:
            first_col = df.iloc[:, 0].astype(str)
            # Check if first column looks like pairs (2-character strings)
            if first_col.str.len().eq(2).all():
                return 'item_pair_scores'  # Default to item pairs
        return 'unknown'

def get_title_for_type(data_type):
    """Return a properly formatted title for a given data type."""
    titles = {
        'item_scores': 'Item frequency scores',
        'item_pair_scores': 'Item-pair frequency scores',
        'position_scores': 'Position comfort scores',
        'position_pair_scores': 'Position-pair comfort scores',
        'unknown': 'Data scores'
    }
    return titles.get(data_type, data_type.replace("_", " ").title())

def load_and_validate_data(raw_file, norm_file):
    """Load and validate raw and normalized CSV files."""
    # Load raw data
    try:
        raw_df = pd.read_csv(raw_file)
        print(f"Loaded raw data: {len(raw_df)} rows, {len(raw_df.columns)} columns")
    except Exception as e:
        raise ValueError(f"Could not load raw file '{raw_file}': {e}")
    
    # Load normalized data
    try:
        norm_df = pd.read_csv(norm_file)
        print(f"Loaded normalized data: {len(norm_df)} rows, {len(norm_df.columns)} columns")
    except Exception as e:
        raise ValueError(f"Could not load normalized file '{norm_file}': {e}")
    
    # Detect data type
    data_type = detect_data_type(norm_df)
    print(f"Detected data type: {data_type}")
    
    # Find score columns
    raw_score_col = None
    norm_score_col = None
    
    # Look for score column in raw data
    for col in raw_df.columns:
        if any(keyword in col.lower() for keyword in ['score', 'frequency', 'comfort', 'value']):
            raw_score_col = col
            break
    
    # Look for score column in normalized data
    for col in norm_df.columns:
        if any(keyword in col.lower() for keyword in ['score', 'frequency', 'comfort', 'value']):
            norm_score_col = col
            break
    
    if raw_score_col is None:
        # Use last column as default
        raw_score_col = raw_df.columns[-1]
        print(f"Warning: No score column found in raw data, using '{raw_score_col}'")
    
    if norm_score_col is None:
        # Use last column as default
        norm_score_col = norm_df.columns[-1]
        print(f"Warning: No score column found in normalized data, using '{norm_score_col}'")
    
    return raw_df, norm_df, data_type, raw_score_col, norm_score_col

def create_distribution_comparison(raw_df, norm_df, raw_score_col, norm_score_col, 
                                 data_type, output_dir):
    """Create distribution comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Score distribution: {get_title_for_type(data_type)}', fontsize=16)
    
    # Raw distribution
    raw_values = raw_df[raw_score_col].dropna()
    ax1.hist(raw_values, bins=30, alpha=0.7, color='black', edgecolor='gray')
    ax1.set_title('Raw data')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Count')
    
    # Use log scale if range is large
    if len(raw_values) > 0 and raw_values.max() / max(raw_values.min(), 1e-10) > 1000:
        ax1.set_xscale('log')
        ax1.set_title('Raw data (log scale)')
    
    # Normalized distribution
    norm_values = norm_df[norm_score_col].dropna()
    ax2.hist(norm_values, bins=30, alpha=0.7, color='gray', edgecolor='black')
    ax2.set_title('Normalized data')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{data_type}_distribution_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_top_scores_plot(norm_df, norm_score_col, data_type, output_dir, n_show=30):
    """Create top scores plot for normalized data."""
    # Sort by score and take top N
    sorted_df = norm_df.sort_values(norm_score_col, ascending=False).head(n_show)
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(sorted_df))
    plt.bar(x, sorted_df[norm_score_col], color='black', alpha=0.8, 
            edgecolor='gray', linewidth=0.5)
    
    plt.title(f'{get_title_for_type(data_type)}: normalized (top {len(sorted_df)})')
    plt.ylabel('Normalized score')
    plt.xlabel('Items')
    
    # Set x-tick labels (use first column as labels)
    if len(sorted_df) > 0:
        labels = sorted_df.iloc[:, 0].astype(str)
        # Convert to uppercase if it looks like position data
        if data_type.startswith('position'):
            labels = labels.str.upper()
        
        plt.xticks(x, labels, rotation=45 if len(labels) > 15 else 0)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{data_type}_normalized_top{len(sorted_df)}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def analyze_pair_participation(raw_df, norm_df, raw_score_col, norm_score_col, 
                             data_type, output_dir):
    """Analyze participation in pairs if this is pair data."""
    if not data_type.endswith('_pair_scores'):
        return
    
    print(f"Analyzing {data_type} participation...")
    
    # Get the pair column (first column)
    pair_col = raw_df.columns[0]
    
    # Extract individual entities from pairs
    entities = set()
    for pair_str in raw_df[pair_col].astype(str):
        if len(pair_str) >= 2:
            entities.add(pair_str[0].lower())
            entities.add(pair_str[1].lower())
    
    # Analyze each entity
    results = []
    for entity in sorted(entities):
        # Find pairs containing this entity in raw data
        entity_raw_scores = []
        entity_norm_scores = []
        
        for i, pair_str in enumerate(raw_df[pair_col].astype(str)):
            if len(pair_str) >= 2 and entity in pair_str.lower():
                entity_raw_scores.append(raw_df.iloc[i][raw_score_col])
                
                # Find corresponding normalized score
                if i < len(norm_df):
                    entity_norm_scores.append(norm_df.iloc[i][norm_score_col])
        
        if entity_raw_scores:
            result = {
                'entity': entity.upper() if data_type.startswith('position') else entity,
                'pair_count': len(entity_raw_scores),
                'raw_min': min(entity_raw_scores),
                'raw_max': max(entity_raw_scores),
                'norm_min': min(entity_norm_scores) if entity_norm_scores else 0,
                'norm_max': max(entity_norm_scores) if entity_norm_scores else 0,
            }
            results.append(result)
    
    if results:
        # Save participation analysis
        results_df = pd.DataFrame(results)
        entity_type = 'item' if data_type.startswith('item') else 'position'
        output_path = os.path.join(output_dir, f'{entity_type}_pair_participation.csv')
        results_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        
        # Create visualization
        create_participation_plot(results_df, data_type, output_dir)

def create_participation_plot(results_df, data_type, output_dir):
    """Create a plot showing entity participation in pairs."""
    if results_df.empty:
        return
    
    # Sort by normalized max score
    sorted_df = results_df.sort_values('norm_max', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(sorted_df) * 0.3)))
    
    y_pos = np.arange(len(sorted_df))
    
    # Create horizontal bars showing score ranges
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        width = row['norm_max'] - row['norm_min']
        ax.barh(i, width, left=row['norm_min'], height=0.6, 
                alpha=0.8, color='black', edgecolor='gray', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_df['entity'])
    ax.set_xlabel('Normalized score range')
    ax.set_title(f'{get_title_for_type(data_type)}: Entity participation ranges')
    ax.grid(True, alpha=0.3)
    
    entity_type = 'item' if data_type.startswith('item') else 'position'
    output_path = os.path.join(output_dir, f'{entity_type}_pair_score_ranges.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_summary_report(raw_df, norm_df, raw_score_col, norm_score_col, 
                         data_type, raw_file, norm_file, output_dir):
    """Create a summary report."""
    report_path = os.path.join(output_dir, "comparison_summary.txt")
    
    with open(report_path, 'w') as f:
        f.write("=== DATA COMPARISON SUMMARY REPORT ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Raw file: {raw_file}\n")
        f.write(f"Normalized file: {norm_file}\n")
        f.write(f"Data type: {data_type}\n\n")
        
        # Raw data statistics
        raw_values = raw_df[raw_score_col].dropna()
        f.write("RAW DATA STATISTICS:\n")
        f.write(f"  Count: {len(raw_values)}\n")
        if len(raw_values) > 0:
            f.write(f"  Min: {raw_values.min():.6g}\n")
            f.write(f"  Max: {raw_values.max():.6g}\n")
            f.write(f"  Mean: {raw_values.mean():.6g}\n")
            f.write(f"  Std: {raw_values.std():.6g}\n")
        
        # Normalized data statistics
        norm_values = norm_df[norm_score_col].dropna()
        f.write("\nNORMALIZED DATA STATISTICS:\n")
        f.write(f"  Count: {len(norm_values)}\n")
        if len(norm_values) > 0:
            f.write(f"  Min: {norm_values.min():.6g}\n")
            f.write(f"  Max: {norm_values.max():.6g}\n")
            f.write(f"  Mean: {norm_values.mean():.6g}\n")
            f.write(f"  Std: {norm_values.std():.6g}\n")
        
        # Data quality checks
        f.write("\nDATA QUALITY:\n")
        f.write(f"  Raw missing values: {raw_df[raw_score_col].isna().sum()}\n")
        f.write(f"  Normalized missing values: {norm_df[norm_score_col].isna().sum()}\n")
        f.write(f"  Row count match: {len(raw_df) == len(norm_df)}\n")
    
    print(f"Summary report saved: {report_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Compare raw and normalized input data")
    parser.add_argument("--raw", type=str, required=True, help="Path to raw CSV file")
    parser.add_argument("--norm", type=str, required=True, help="Path to normalized CSV file")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        # Validate input files
        if not os.path.exists(args.raw):
            raise FileNotFoundError(f"Raw file not found: {args.raw}")
        if not os.path.exists(args.norm):
            raise FileNotFoundError(f"Normalized file not found: {args.norm}")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load and validate data
        print("Loading and validating data...")
        raw_df, norm_df, data_type, raw_score_col, norm_score_col = load_and_validate_data(
            args.raw, args.norm)
        
        # Create comparison plots
        print("Creating distribution comparison...")
        create_distribution_comparison(raw_df, norm_df, raw_score_col, norm_score_col, 
                                     data_type, args.output_dir)
        
        print("Creating top scores plot...")
        create_top_scores_plot(norm_df, norm_score_col, data_type, args.output_dir)
        
        # Analyze pairs if applicable
        analyze_pair_participation(raw_df, norm_df, raw_score_col, norm_score_col, 
                                 data_type, args.output_dir)
        
        # Create summary report
        print("Creating summary report...")
        create_summary_report(raw_df, norm_df, raw_score_col, norm_score_col, 
                            data_type, args.raw, args.norm, args.output_dir)
        
        print(f"\nAnalysis complete! Output saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
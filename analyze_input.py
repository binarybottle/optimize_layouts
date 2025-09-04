#!/usr/bin/env python
"""
Compare Raw and Normalized Input Data

This script compares raw input data with its normalized version through visual
plots and statistical analysis. It reads raw file paths from config.yaml.

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
- Item-pair participation analysis (item_pair_participation.csv)
- Position-pair participation analysis (position_pair_participation.csv)
- Item-pair score range visualization (item_pair_score_ranges.png)
- Item-pair score scatter plot (item_pair_score_scatter.png)
- Item-pair cumulative distribution (item_pair_cumulative.png)
- Position-pair score range visualization (position_pair_score_ranges.png)
- Position-pair score scatter plot (position_pair_score_scatter.png)
- Position-pair cumulative distribution (position_pair_cumulative.png)

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

# Set random seed for reproducible jitter in scatter plots
np.random.seed(42)

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
        'item_scores': 'Item frequency scores',
        'item_pair_scores': 'Item-pair frequency scores',
        'position_scores': 'Position comfort scores',
        'position_pair_scores': 'Position-pair comfort scores'
    }
    return titles.get(key, key.replace("_", " ").title())

#-----------------------------------------------------------------------------
# Load raw data (simplified for new architecture)
#-----------------------------------------------------------------------------
def load_raw_scores(config: Config):
    """Load raw scores using the config structure (simplified version)."""
    
    # For this refactored version, we'll focus on the main files from config
    # and create placeholder data structures for compatibility
    
    raw_data = {}
    
    # Try to load item-pair scores from config path
    item_pair_file = config.paths.item_pair_score_table
    if os.path.exists(item_pair_file):
        try:
            df = pd.read_csv(item_pair_file)
            # Try to find the appropriate columns
            pair_col = None
            score_col = None
            
            for col in df.columns:
                if 'pair' in col.lower():
                    pair_col = col
                if 'score' in col.lower() or 'frequency' in col.lower():
                    score_col = col
            
            if pair_col and score_col:
                pair_scores = {}
                for _, row in df.iterrows():
                    pair_str = str(row[pair_col])
                    if len(pair_str) == 2:
                        key = (pair_str[0].lower(), pair_str[1].lower())
                        pair_scores[key] = float(row[score_col])
                raw_data['item_pair_scores'] = pair_scores
                print(f"Loaded {len(pair_scores)} item-pair scores from {item_pair_file}")
        except Exception as e:
            print(f"Warning: Could not load item-pair scores: {e}")
            raw_data['item_pair_scores'] = {}
    
    # Try to load position-pair scores from config path
    position_pair_file = config.paths.position_pair_score_table
    if os.path.exists(position_pair_file):
        try:
            df = pd.read_csv(position_pair_file)
            # Use first objective as example
            if 'position_pair' in df.columns and len(config.moo.default_objectives) > 0:
                obj_col = config.moo.default_objectives[0]
                if obj_col in df.columns:
                    pair_scores = {}
                    for _, row in df.iterrows():
                        pair_str = str(row['position_pair'])
                        if len(pair_str) == 2:
                            key = (pair_str[0].lower(), pair_str[1].lower())
                            pair_scores[key] = float(row[obj_col])
                    raw_data['position_pair_scores'] = pair_scores
                    print(f"Loaded {len(pair_scores)} position-pair scores from {position_pair_file}")
        except Exception as e:
            print(f"Warning: Could not load position-pair scores: {e}")
            raw_data['position_pair_scores'] = {}
    
    # Initialize empty dictionaries for missing data types
    for key in ['item_scores', 'position_scores']:
        if key not in raw_data:
            raw_data[key] = {}
    
    return raw_data

def load_normalized_scores(config: Config):
    """Load normalized scores from the expected output directory."""
    normalized_data = {}
    
    # Standard paths for normalized data
    norm_dir = "output/normalized_input"
    
    norm_files = {
        'item_scores': 'normalized_item_scores.csv',
        'item_pair_scores': 'normalized_item_pair_scores.csv',
        'position_scores': 'normalized_position_scores.csv',
        'position_pair_scores': 'normalized_position_pair_scores.csv'
    }
    
    for key, filename in norm_files.items():
        filepath = os.path.join(norm_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                normalized_data[key] = df
                print(f"Loaded normalized {key}: {len(df)} entries")
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
                normalized_data[key] = pd.DataFrame()
        else:
            print(f"Warning: Normalized file not found: {filepath}")
            normalized_data[key] = pd.DataFrame()
    
    return normalized_data

#-----------------------------------------------------------------------------
# Pair participation analysis functions (updated)
#-----------------------------------------------------------------------------
def plot_score_ranges(df, title, output_dir, filename, normalized_scores=None):
    """Plot horizontal bars showing score ranges for each item/position."""
    if df.empty:
        return
    
    # Use normalized scores if provided, otherwise use original scores
    if normalized_scores is not None:
        # Create a copy of df with normalized scores
        df_plot = df.copy()
        df_plot = add_normalized_scores_to_df(df_plot, normalized_scores)
        high_col, low_col = 'highest_norm_score', 'lowest_norm_score'
        xlabel = 'Normalized pair score range [0-1]'
        plot_scores = normalized_scores
    else:
        df_plot = df.copy()
        high_col, low_col = 'highest_pair_score', 'lowest_pair_score'
        xlabel = 'Pair score range'
        plot_scores = None
    
    # Sort by highest score for better visualization
    df_sorted = df_plot.sort_values(high_col, ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(df_sorted) * 0.35)))
    
    y_pos = np.arange(len(df_sorted))
    
    # Calculate x-axis limits to match scatter plot
    if plot_scores is not None:
        all_values = list(plot_scores.values())
    else:
        all_values = []
        for _, row in df_sorted.iterrows():
            all_values.extend([row[high_col], row[low_col]])
    
    x_min, x_max = min(all_values), max(all_values)
    x_range = x_max - x_min
    x_margin = x_range * 0.05  # 5% margin on each side
    
    # Text distance from bars (adjust these values to change spacing)
    text_distance = x_margin * 0.1  # Change this multiplier: 0.1=closer, 1.0=farther
    
    # Create horizontal bars from lowest to highest score (black and white)
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax.barh(i, row[high_col] - row[low_col], 
                left=row[low_col], height=0.6, alpha=0.8, 
                color='black', edgecolor='gray', linewidth=0.5)
        
        # Add text annotations for the range
        ax.text(row[high_col] + text_distance, i, f"{row[high_col]:.3f}", 
                va='center', fontsize=8)
        ax.text(row[low_col] - text_distance, i, f"{row[low_col]:.3f}", 
                va='center', ha='right', fontsize=8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted.iloc[:, 0])  # First column (item/position)
    ax.set_xlabel(xlabel)
    ax.set_title(f'{title}: Score ranges')
    ax.grid(True, alpha=0.3)
    
    # Set identical x-axis limits as scatter plot
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {output_path}")
    plt.close()

def plot_score_scatter(pair_scores, df, title, output_dir, filename, normalized_scores=None):
    """Plot horizontal scatter showing all individual pair scores for each item/position."""
    if df.empty:
        return
    
    # Use normalized scores if provided
    if normalized_scores is not None:
        df_plot = df.copy()
        df_plot = add_normalized_scores_to_df(df_plot, normalized_scores)
        sort_col = 'highest_norm_score'
        xlabel = 'Normalized pair scores [0-1]'
        # Use normalized pair scores
        plot_pair_scores = normalized_scores
    else:
        df_plot = df.copy()
        sort_col = 'highest_pair_score'
        xlabel = 'Pair scores'
        plot_pair_scores = pair_scores
    
    # Sort by highest score for consistency
    df_sorted = df_plot.sort_values(sort_col, ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(df_sorted) * 0.35)))
    
    y_pos = np.arange(len(df_sorted))
    
    # Calculate x-axis limits to match bar plot
    all_values = list(plot_pair_scores.values())
    x_min, x_max = min(all_values), max(all_values)
    x_range = x_max - x_min
    x_margin = x_range * 0.05  # 5% margin on each side
    
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        entity = row.iloc[0].lower() if 'item' in df.columns else row.iloc[0].lower()
        
        # Find all scores for this entity
        entity_scores = []
        for pair, score in plot_pair_scores.items():
            if entity in pair:
                entity_scores.append(score)
        
        if entity_scores:
            # Use transparency instead of jitter to handle overlapping dots
            ax.scatter(entity_scores, [i] * len(entity_scores), 
                      alpha=0.5, color='black', s=25, edgecolors='gray', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted.iloc[:, 0])  # First column
    ax.set_xlabel(xlabel)
    ax.set_title(f'{title}')
    ax.grid(True, alpha=0.3)
    
    # Set identical x-axis limits as bar plot
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {output_path}")
    plt.close()

def plot_cumulative_scores(pair_scores, df, title, output_dir, filename, normalized_scores=None):
    """Plot cumulative distribution showing where each item/position maxes out."""
    if df.empty:
        return
    
    # Use normalized scores if provided
    if normalized_scores is not None:
        df_plot = df.copy()
        df_plot = add_normalized_scores_to_df(df_plot, normalized_scores)
        sort_col = 'highest_norm_score'
        ylabel = 'Normalized pair score [0-1]'
        plot_pair_scores = normalized_scores
    else:
        df_plot = df.copy()
        sort_col = 'highest_pair_score'
        ylabel = 'Pair score'
        plot_pair_scores = pair_scores
    
    # Sort by highest score for better visualization
    df_sorted = df_plot.sort_values(sort_col, ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use different line styles instead of colors for black/white
    line_styles = ['-', '--', '-.', ':']
    line_widths = [2, 1.5, 2, 1.5]
    
    for i, (_, row) in enumerate(df_sorted.head(20).iterrows()):  # Limit to top 20 for readability
        entity = row.iloc[0].lower() if 'item' in df.columns else row.iloc[0].lower()
        
        # Find all scores for this entity
        entity_scores = []
        for pair, score in plot_pair_scores.items():
            if entity in pair:
                entity_scores.append(score)
        
        if entity_scores:
            # Sort scores in descending order
            sorted_scores = sorted(entity_scores, reverse=True)
            x_values = range(1, len(sorted_scores) + 1)
            
            # Plot the cumulative "max-out" curve with different line styles
            style_idx = i % len(line_styles)
            ax.plot(x_values, sorted_scores, 
                   linestyle=line_styles[style_idx],
                   linewidth=line_widths[style_idx],
                   marker='o' if i < 10 else 's', 
                   markersize=3 if i < 10 else 2,
                   color='black',
                   alpha=0.8 if i < 5 else 0.6,
                   label=f"{row.iloc[0]} (max: {max(sorted_scores):.3f})")
    
    ax.set_xlabel('Pair rank (sorted by score)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title}: Cumulative score distribution (top 20)')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Use same spacing as other plots to maintain consistency
    plt.subplots_adjust(left=0.20, right=0.95, top=0.95, bottom=0.1)
    
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def load_normalized_pair_scores(config: Config):
    """Load normalized pair scores from CSV files."""
    normalized_pairs = {}
    
    # Standard filenames for normalized pair data
    norm_files = {
        'item_pair_scores': 'normalized_item_pair_scores.csv',
        'position_pair_scores': 'normalized_position_pair_scores.csv'
    }
    
    norm_dir = "output/normalized_input"
    
    for key, filename in norm_files.items():
        filepath = os.path.join(norm_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            
            # Convert to dictionary format
            pair_scores = {}
            for _, row in df.iterrows():
                if key == 'item_pair_scores':
                    pair_str = str(row['item_pair'])
                else:  # position_pair_scores
                    pair_str = str(row['position_pair'])
                
                if len(pair_str) == 2:
                    pair_key = (pair_str[0].lower(), pair_str[1].lower())
                    pair_scores[pair_key] = float(row['score'])
            
            normalized_pairs[key] = pair_scores
    
    return normalized_pairs

def add_normalized_scores_to_df(df, normalized_scores):
    """Add normalized score columns to the dataframe."""
    if 'item' in df.columns:
        # Item analysis
        entity_col = 'item'
    else:
        # Position analysis
        entity_col = 'position'
    
    # Initialize new columns
    df['highest_norm_score'] = 0.0
    df['lowest_norm_score'] = 0.0
    df['highest_norm_rank'] = 0
    df['lowest_norm_rank'] = 0
    
    # Get sorted normalized scores for ranking (highest score = rank 1)
    all_norm_scores = sorted(set(normalized_scores.values()), reverse=True)
    norm_score_to_rank = {score: rank + 1 for rank, score in enumerate(all_norm_scores)}
    
    for idx, row in df.iterrows():
        entity = row[entity_col].lower()
        
        # Find all normalized scores for this entity
        entity_norm_scores = []
        for pair, score in normalized_scores.items():
            if entity in pair:
                entity_norm_scores.append(score)
        
        if entity_norm_scores:
            highest_norm = max(entity_norm_scores)
            lowest_norm = min(entity_norm_scores)
            
            df.at[idx, 'highest_norm_score'] = highest_norm
            df.at[idx, 'lowest_norm_score'] = lowest_norm
            df.at[idx, 'highest_norm_rank'] = norm_score_to_rank[highest_norm]
            df.at[idx, 'lowest_norm_rank'] = norm_score_to_rank[lowest_norm]
    
    return df

def analyze_items_in_pairs(item_pair_scores, normalized_scores=None):
    """Analyze item participation in item-pairs."""
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
            
            result = {
                'item': item,
                'pair_count': count,
                'pair_percent': round(percent, 2),
                'highest_pair_score': highest_score,
                'highest_pair_rank': highest_rank,
                'lowest_pair_score': lowest_score,
                'lowest_pair_rank': lowest_rank
            }
            
            # Add normalized scores if available
            if normalized_scores:
                item_norm_scores = []
                for pair, score in normalized_scores.items():
                    if item in pair:
                        item_norm_scores.append(score)
                
                if item_norm_scores:
                    # Get normalized score rankings
                    all_norm_scores = sorted(set(normalized_scores.values()), reverse=True)
                    norm_score_to_rank = {score: rank + 1 for rank, score in enumerate(all_norm_scores)}
                    
                    highest_norm = max(item_norm_scores)
                    lowest_norm = min(item_norm_scores)
                    
                    result.update({
                        'highest_norm_score': highest_norm,
                        'highest_norm_rank': norm_score_to_rank[highest_norm],
                        'lowest_norm_score': lowest_norm,
                        'lowest_norm_rank': norm_score_to_rank[lowest_norm]
                    })
            
            results.append(result)
    
    return pd.DataFrame(results)

def analyze_positions_in_pairs(position_pair_scores, normalized_scores=None):
    """Analyze position participation in position-pairs."""
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
            
            result = {
                'position': position.upper(),  # Convert to uppercase for consistency
                'pair_count': count,
                'pair_percent': round(percent, 2),
                'highest_pair_score': highest_score,
                'highest_pair_rank': highest_rank,
                'lowest_pair_score': lowest_score,
                'lowest_pair_rank': lowest_rank
            }
            
            # Add normalized scores if available
            if normalized_scores:
                position_norm_scores = []
                for pair, score in normalized_scores.items():
                    if position in pair:
                        position_norm_scores.append(score)
                
                if position_norm_scores:
                    # Get normalized score rankings
                    all_norm_scores = sorted(set(normalized_scores.values()), reverse=True)
                    norm_score_to_rank = {score: rank + 1 for rank, score in enumerate(all_norm_scores)}
                    
                    highest_norm = max(position_norm_scores)
                    lowest_norm = min(position_norm_scores)
                    
                    result.update({
                        'highest_norm_score': highest_norm,
                        'highest_norm_rank': norm_score_to_rank[highest_norm],
                        'lowest_norm_score': lowest_norm,
                        'lowest_norm_rank': norm_score_to_rank[lowest_norm]
                    })
            
            results.append(result)
    
    return pd.DataFrame(results)

def analyze_pair_participation(raw_data, config, output_dir):
    """Analyze how items and positions participate in pairs and save to CSV files."""
    print("Analyzing pair participation...")
    
    # Load normalized pair scores from CSV files
    normalized_pairs = load_normalized_pair_scores(config)
    
    # Analyze items
    if 'item_pair_scores' in raw_data and raw_data['item_pair_scores']:
        print("  Analyzing item-pair participation...")
        # Get normalized item-pair scores if available
        normalized_item_pairs = normalized_pairs.get('item_pair_scores', {})
        
        item_analysis = analyze_items_in_pairs(raw_data['item_pair_scores'], normalized_item_pairs)
        if not item_analysis.empty:
            item_output_path = os.path.join(output_dir, "item_pair_participation.csv")
            item_analysis.to_csv(item_output_path, index=False)
            print(f"  Saved: {item_output_path}")
            
            # Generate visualizations for items
            print("  Creating item-pair visualizations...")
            plot_score_ranges(item_analysis, "Item-pair participation", output_dir, 
                            "item_pair_score_ranges.png", normalized_item_pairs)
            plot_score_scatter(raw_data['item_pair_scores'], item_analysis, 
                             "Item-pair participation", output_dir, "item_pair_score_scatter.png", 
                             normalized_item_pairs)
            plot_cumulative_scores(raw_data['item_pair_scores'], item_analysis,
                                 "Item-pair participation", output_dir, "item_pair_cumulative.png",
                                 normalized_item_pairs)
        else:
            print("  No item-pair data to analyze")
    
    # Analyze positions  
    if 'position_pair_scores' in raw_data and raw_data['position_pair_scores']:
        print("  Analyzing position-pair participation...")
        # Get normalized position-pair scores if available
        normalized_position_pairs = normalized_pairs.get('position_pair_scores', {})
        
        position_analysis = analyze_positions_in_pairs(raw_data['position_pair_scores'], normalized_position_pairs)
        if not position_analysis.empty:
            position_output_path = os.path.join(output_dir, "position_pair_participation.csv")
            position_analysis.to_csv(position_output_path, index=False)
            print(f"  Saved: {position_output_path}")
            
            # Generate visualizations for positions
            print("  Creating position-pair visualizations...")
            plot_score_ranges(position_analysis, "Position-pair participation", output_dir,
                            "position_pair_score_ranges.png", normalized_position_pairs)
            plot_score_scatter(raw_data['position_pair_scores'], position_analysis,
                             "Position-pair participation", output_dir, "position_pair_score_scatter.png",
                             normalized_position_pairs)
            plot_cumulative_scores(raw_data['position_pair_scores'], position_analysis,
                                 "Position-pair participation", output_dir, "position_pair_cumulative.png",
                                 normalized_position_pairs)
        else:
            print("  No position-pair data to analyze")

#-----------------------------------------------------------------------------
# Plotting functions (updated for new architecture)
#-----------------------------------------------------------------------------
def create_comparison_plots(raw_data, normalized_data, config: Config, output_dir: str):
    """Create all comparison plots using existing data."""
    
    # Generate plots for each data type
    for key in ['item_pair_scores', 'position_pair_scores']:
        if key not in raw_data or not raw_data[key]:
            continue
            
        if key not in normalized_data or normalized_data[key].empty:
            continue
            
        # Distribution comparison
        _plot_distribution_comparison(raw_data[key], normalized_data[key], key, output_dir)
        
        # Top scores analysis
        _plot_top_scores(normalized_data[key], key, output_dir)

def _plot_distribution_comparison(raw_scores, norm_df, key, output_dir):
    """Plot distribution comparison between raw and normalized."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Score distribution: {get_title_for_key(key)}', fontsize=16)
    
    # Raw distribution (black and white)
    raw_values = list(raw_scores.values())
    ax1.hist(raw_values, bins=30, alpha=0.7, color='black', edgecolor='gray')
    ax1.set_title('Raw data')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Count')
    
    # Use log scale if range is large
    if max(raw_values) / max(min(raw_values), 1e-10) > 1000:
        ax1.set_xscale('log')
        ax1.set_title('Raw data (log scale)')
    
    # Normalized distribution (black and white)
    ax2.hist(norm_df['score'], bins=30, alpha=0.7, color='gray', edgecolor='black')
    ax2.set_title('Normalized data [0-1]')
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
    plt.bar(x, sorted_df['score'], color='black', alpha=0.8, edgecolor='gray', linewidth=0.5)
    
    plt.title(f'{get_title_for_key(key)}: normalized (top {len(sorted_df)})')
    plt.ylabel('Normalized score [0-1]')
    plt.xlabel('Items' if not key.startswith('position') else 'Positions')
    plt.ylim(0, 1.0)
    
    # Set x-tick labels
    if key.startswith('position'):
        labels = safely_convert_to_uppercase(sorted_df.iloc[:, 0])  # First column
    else:
        labels = sorted_df.iloc[:, 0]  # First column
    
    plt.xticks(x, labels, rotation=0 if key.endswith('pair_scores') else 0)
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
                f.write(f"  Raw scores: min={min(raw_values):.6g}, max={max(raw_values):.6g}, count={len(raw_values)}\n")
            else:
                f.write("  Raw scores: Not available\n")
            
            # Normalized stats 
            if key in normalized_data and not normalized_data[key].empty:
                norm_df = normalized_data[key]
                f.write(f"  Normalized scores: min={norm_df['score'].min():.6g}, max={norm_df['score'].max():.6g}, count={len(norm_df)}\n")
            else:
                f.write("  Normalized scores: Not available\n")
    
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
        normalized_scores = load_normalized_scores(config)
        
        print("Loading raw scores...")
        raw_data = load_raw_scores(config)
        
        # Generate plots and reports
        print("Generating comparison plots...")
        create_comparison_plots(raw_data, normalized_scores, config, args.output_dir)
        
        print("Creating summary report...")
        create_summary_report(raw_data, normalized_scores, config, args.output_dir)
        
        # Analyze pair participation
        analyze_pair_participation(raw_data, config, args.output_dir)
        
        print(f"\nAnalysis complete! Output saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
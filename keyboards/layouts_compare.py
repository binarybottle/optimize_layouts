#!/usr/bin/env python3
"""
Comprehensive Keyboard Layout Comparison and Visualization Tool

Combines metric comparison, visualization, and analysis for keyboard layouts.
Creates parallel coordinates, heatmaps, scatter plots, Pareto fronts, correlation
matrices, and stability analysis. Supports multiple CSV formats with auto-detection.

Data Format Support (Auto-Detected)
===============================================
CSV Input Formats Supported:
1. Preferred: layout_qwerty column (layout string in QWERTY key order)
2. Standard: letters column (layout string in QWERTY key order) + positions
3. MOO: items + positions columns (letters in assignment order + where they go)
4. Legacy: items + keys columns (alternative names)

Column Meanings:
- layout_qwerty: What's at each QWERTY position (e.g., "  cr  du  oinl  teha   s  m     ")
- letters: What's at each QWERTY position (same as layout_qwerty)  
- items: Letters in assignment order (e.g., "etaoinsrhldcum")
- positions/keys: QWERTY positions where those letters go (e.g., "KJ;ASDVRLFUEIM")

Examples:
    # Basic comparison with auto-detected metrics (all assumed positive):
    python layouts_compare.py --tables layout_scores.csv

    # Specify positive and negative metrics explicitly:
    python layouts_compare.py --tables scores1.csv scores2.csv \
        --positive-metrics comfort rolls dvorak7 \
        --negative-metrics same-finger_bigrams scissors lateral_stretch_bigrams

    # Full analysis with all visualizations and report:
    python layouts_compare.py --tables layouts.csv \
        --positive-metrics comfort rolls redirects \
        --negative-metrics same-finger_bigrams skipgrams scissors \
        --plot --report --summary summary.csv

    # Sort all visualizations by specific metric:
    python layouts_compare.py --tables layouts.csv \
        --positive-metrics comfort rolls \
        --negative-metrics scissors skipgrams \
        --plot --sort-by comfort --summary sorted_by_comfort.csv

    # Keyboard layout optimization study commands:
    # Step 2:
    poetry run python3 layouts_compare.py \
        --tables ../output/layouts_filter_patterns.csv \
        --metrics engram_key_preference engram_row_separation engram_same_row engram_same_finger \
        --output ../output/layouts_compare --summary ../output/layouts_compare.csv \
        --report --plot --verbose \
        --sort-by average_score       
    # Step 3:
    poetry run python3 layouts_compare.py \
        --tables ../output/layouts_consolidate.csv \
        --metrics engram_key_preference engram_row_separation engram_same_row engram_same_finger \
        --output ../output/layouts_compare --summary ../output/layouts_compare.csv \
        --report --plot --verbose \
        --sort-by average_score

    # Compare layouts against Engram:
    # All scores:
    poetry run python3 layouts_compare.py \
        --tables ../output/engram_en/scores_31_layouts.csv \
                 ../output/engram_en/scores_engram.csv \
        --output ../output/layouts_compare_all_scores --summary ../output/layouts_compare_all_scores.csv \
        --sort-by average_score --report --plot --verbose \
        --positive-metrics key_preference row_separation same_row same_finger outside_homeblock trigram_sequence \
            dvorak_distribution dvorak_row_span dvorak_homeblocks dvorak_remote_fingers \
            comfort \
        --negative-metrics distance \
                SFBs_1u SFBs_2u SFSs_1u SFSs_2u LSBs_1u LSBs_2u skipgrams_1u skipgrams_2u
    # Engram scores:
    poetry run python3 layouts_compare.py \
        --tables ../output/engram_en/scores_31_layouts.csv \
                 ../output/engram_en/scores_engram.csv \
        --output ../output/layouts_compare_engram_scores --summary ../output/layouts_compare_engram_scores.csv \
        --sort-by average_score --report --plot --verbose \
        --positive-metrics key_preference row_separation same_row same_finger outside_homeblock trigram_sequence
    # Dvorak scores:
    poetry run python3 layouts_compare.py \
        --tables ../output/engram_en/scores_31_layouts.csv \
                 ../output/engram_en/scores_engram.csv \
        --output ../output/layouts_compare_dvorak_scores --summary ../output/layouts_compare_dvorak_scores.csv \
        --sort-by average_score --report --plot --verbose \
        --positive-metrics dvorak_distribution dvorak_row_span dvorak_homeblocks dvorak_remote_fingers 
    # All non-Engram and non-Dvorak scores:
    poetry run python3 layouts_compare.py \
        --tables ../output/engram_en/scores_31_layouts.csv \
                 ../output/engram_en/scores_engram.csv \
        --output ../output/layouts_compare_misc_scores --summary ../output/layouts_compare_misc_scores.csv \
        --sort-by average_score --report --plot --verbose \
        --positive-metrics comfort \
        --negative-metrics distance \
                SFBs_1u SFBs_2u SFSs_1u SFSs_2u LSBs_1u LSBs_2u skipgrams_1u skipgrams_2u
                
Input format examples:
  
  Preferred format:
  layout,layout_qwerty,engram,dvorak7,comfort
  Dvorak,',.pyfgcrlaeoiduhtns;qjkxbmwvz,0.712,0.698,0.654
  
  MOO format (auto-converted):
  config_id,items,positions,engram_key_preference,engram_avg4_score
  2438,etaoinsrhldcum,KJ;ASDVRLFUEIM,0.742,0.960

Summary output:
  CSV with columns: index, layout, [layout_qwerty, positions], average_score, balanced_average, [metric_values]
  Layouts ordered by average performance across selected metrics (higher = better)
  balanced_average penalizes layouts with wide tradeoffs (mean - 0.5 × std_dev)
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
import sys
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime

# High-contrast color palette for visualizations
PLOT_COLORS = [
    '#1f77b4',  # Blue
    '#000000',  # Black  
    '#2ca02c',  # Green
    '#ff9896',  # Light Red
    '#d62728',  # Red
    '#008080',  # Teal
    '#17becf',  # Cyan
    '#800080',  # Purple
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
]

def get_colors(n_colors: int) -> List[str]:
    """Get high-contrast colors, cycling if needed."""
    if n_colors <= len(PLOT_COLORS):
        return PLOT_COLORS[:n_colors]
    else:
        return [PLOT_COLORS[i % len(PLOT_COLORS)] for i in range(n_colors)]

def parse_layout_string(layout_string: str) -> tuple:
    """Parse layout string to extract letters and their positions."""
    if pd.isna(layout_string) or not layout_string:
        return "", ""
    
    # QWERTY reference positions
    qwerty_positions = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"
    
    # Extract letters from layout string (should be same length as QWERTY)
    layout_letters = layout_string.strip('"').replace('\\', '')  # Clean up quotes and escapes
    
    if len(layout_letters) != len(qwerty_positions):
        # Try to pad or truncate to match QWERTY length
        if len(layout_letters) < len(qwerty_positions):
            layout_letters = layout_letters + ' ' * (len(qwerty_positions) - len(layout_letters))
        else:
            layout_letters = layout_letters[:len(qwerty_positions)]
    
    # Create mapping from layout position to letter
    position_to_letter = {}
    
    for i, (pos, letter) in enumerate(zip(qwerty_positions, layout_letters)):
        position_to_letter[pos] = letter
    
    # Letters in QWERTY order (what letter is at each QWERTY position)
    letters = ''.join(position_to_letter.get(pos, ' ') for pos in qwerty_positions)
    
    # QWERTY positions (reference)
    positions = qwerty_positions
    
    return letters, positions

def load_layout_data(file_path: str, verbose: bool = False) -> pd.DataFrame:
    """
    Load CSV data with automatic format detection.
    
    Supports multiple CSV formats:
    - Preferred: layout_qwerty column (layout string in QWERTY key order)
    - Standard: letters + positions columns (letters in QWERTY order + reference)  
    - MOO: items + positions columns (letters in assignment order + where they go)
    - Legacy: items + keys columns (alternative column names)
    
    Args:
        file_path: Path to CSV file
        verbose: Print format detection details
        
    Returns:
        DataFrame with standardized column names (letters, positions)
    """
    try:
        df = pd.read_csv(file_path)
        if verbose:
            print(f"\nLoaded {len(df)} rows from {file_path}")
            print(f"Columns: {list(df.columns)}")
        
        # Check required columns with automatic format detection
        if 'layout' not in df.columns and 'layout_qwerty' in df.columns:
            df['layout'] = df['layout_qwerty']
            if verbose:
                print("  Using 'layout_qwerty' as 'layout' column")

        # Detect and handle different layout column formats
        letters_col = None
        positions_col = None
        
        # Priority order: layout_qwerty > letters > items
        if 'layout_qwerty' in df.columns:
            letters_col = 'layout_qwerty'
            if verbose:
                print("  Format: layout_qwerty (preferred)")
        elif 'letters' in df.columns:
            letters_col = 'letters'
            if verbose:
                print("  Format: letters (standard)")
        elif 'items' in df.columns:
            letters_col = 'items'
            if verbose:
                print("  Format: items (MOO format)")
        else:
            raise ValueError("Missing layout data column: need 'layout_qwerty', 'letters', or 'items'")
        
        # Handle positions column
        for col in ['positions', 'keys']:
            if col in df.columns:
                positions_col = col
                break
        if not positions_col and letters_col == 'items':
            raise ValueError("MOO format requires 'positions' or 'keys' column")
        
        # Standardize column names for internal use
        # Drop the target column if it exists and is different from the source
        if letters_col != 'letters':
            if 'letters' in df.columns and letters_col != 'letters':
                df = df.drop(columns=['letters'])
            df = df.rename(columns={letters_col: 'letters'})
        if positions_col and positions_col != 'positions':
            if 'positions' in df.columns and positions_col != 'positions':
                df = df.drop(columns=['positions'])
            df = df.rename(columns={positions_col: 'positions'})
        
        # Ensure no duplicate columns (safety check)
        if df.columns.duplicated().any():
            duplicates = df.columns[df.columns.duplicated()].tolist()
            print(f"Warning: Found duplicate columns {duplicates}, keeping first occurrence")
            df = df.loc[:, ~df.columns.duplicated()]
        
        # Find scorer columns (numeric columns after layout, letters, positions)
        scorer_columns = []
        for col in df.columns[3:]:  # Skip layout, letters, positions
            if pd.api.types.is_numeric_dtype(df[col]):
                scorer_columns.append(col)
        
        if verbose:
            print(f"Found {len(df)} layouts")
            print(f"Found {len(scorer_columns)} scorer columns: {', '.join(scorer_columns)}")
        
        return df
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)

def find_available_metrics(dfs: List[pd.DataFrame], verbose: bool = False) -> List[str]:
    """Find which scorer metrics are available in the data."""
    # Get all columns that appear to be numeric metrics (excluding 'layout', 'letters', 'positions')
    all_metrics = set()
    for df in dfs:
        for col in df.columns:
            if col not in ['layout', 'layout_qwerty','letters', 'positions'] and pd.api.types.is_numeric_dtype(df[col]):
                all_metrics.add(col)
    
    # Convert to sorted list (alphabetical order)
    available_metrics = sorted(list(all_metrics))
    
    if verbose:
        print(f"\nFound {len(available_metrics)} scorer metrics available:")
        for i, metric in enumerate(available_metrics):
            print(f"  {i+1:2d}. {metric}")
    
    return available_metrics

def filter_and_order_metrics(dfs: List[pd.DataFrame], requested_metrics: Optional[List[str]] = None, 
                           verbose: bool = False) -> List[str]:
    """Filter and order metrics based on user specification."""
    # Get all available metrics
    available_metrics = find_available_metrics(dfs, verbose)
    
    if not requested_metrics:
        # Return all available metrics in alphabetical order
        return available_metrics
    
    # Filter to only requested metrics that are available
    filtered_metrics = []
    missing_metrics = []
    
    for metric in requested_metrics:
        if metric in available_metrics:
            filtered_metrics.append(metric)
        else:
            missing_metrics.append(metric)
    
    if missing_metrics and verbose:
        print(f"\nWarning: Requested metrics not found in data: {', '.join(missing_metrics)}")
    
    if not filtered_metrics:
        print("Error: None of the requested metrics were found in the data")
        print(f"Available metrics: {', '.join(available_metrics)}")
        sys.exit(1)
    
    if verbose:
        print(f"\nUsing {len(filtered_metrics)} metrics in specified order:")
        for i, metric in enumerate(filtered_metrics):
            print(f"  {i+1:2d}. {metric}")
    
    return filtered_metrics

def normalize_data(dfs: List[pd.DataFrame], positive_metrics: List[str], 
                  negative_metrics: List[str]) -> List[pd.DataFrame]:
    """Normalize all data across tables for fair comparison.
    
    Args:
        dfs: List of dataframes to normalize
        positive_metrics: Metrics where higher is better (normalized as-is)
        negative_metrics: Metrics where lower is better (inverted during normalization)
    
    Returns:
        List of normalized dataframes where all metrics are oriented so higher = better
    """
    # Check for duplicate columns in each dataframe before concatenating
    for i, df in enumerate(dfs):
        if df.columns.duplicated().any():
            duplicates = df.columns[df.columns.duplicated()].tolist()
            print(f"Warning: DataFrame {i} has duplicate columns {duplicates}, removing duplicates")
            dfs[i] = df.loc[:, ~df.columns.duplicated()]
        
        # Check for duplicate indices and reset if needed
        if df.index.duplicated().any():
            print(f"Warning: DataFrame {i} has duplicate row indices, resetting index")
            dfs[i] = df.reset_index(drop=True)
    
    # Combine all data to get global min/max for each metric
    all_data = pd.concat(dfs, ignore_index=True)
    
    # Combine all metrics
    all_metrics = positive_metrics + negative_metrics
    
    normalized_dfs = []
    
    for table_idx, df in enumerate(dfs):
        normalized_df = df.copy()
        
        # Process positive metrics (higher is better)
        for metric in positive_metrics:
            if metric in df.columns:
                global_min = all_data[metric].min()
                global_max = all_data[metric].max()
                
                if pd.notna(global_min) and pd.notna(global_max) and global_max != global_min:
                    # Standard normalization: higher score = better performance
                    normalized_df[metric] = (df[metric] - global_min) / (global_max - global_min)
                else:
                    normalized_df[metric] = 0.5  # Default to middle if no variation
            else:
                normalized_df[metric] = 0.0  # Default to 0 if metric is missing
        
        # Process negative metrics (lower is better) - INVERT THEM
        for metric in negative_metrics:
            if metric in df.columns:
                global_min = all_data[metric].min()
                global_max = all_data[metric].max()
                
                if pd.notna(global_min) and pd.notna(global_max) and global_max != global_min:
                    # INVERTED normalization: lower score = better performance
                    # So we flip it: (max - value) / (max - min) 
                    # This makes low values map to high normalized scores
                    normalized_df[metric] = (global_max - df[metric]) / (global_max - global_min)
                else:
                    normalized_df[metric] = 0.5  # Default to middle if no variation
            else:
                normalized_df[metric] = 0.0  # Default to 0 if metric is missing
        
        normalized_dfs.append(normalized_df)
    
    return normalized_dfs

def get_table_colors(num_tables: int) -> List[str]:
    """Get color scheme based on number of tables."""
    if num_tables == 1:
        return ['gray']
    elif num_tables == 2:
        return ['#2196F3', '#F44336']  # Blue, Red
    elif num_tables == 3:
        return ['#2196F3', '#F44336', '#4CAF50']  # Blue, Red, Green
    elif num_tables == 4:
        return ['#2196F3', '#F44336', '#4CAF50', '#FF9800']  # Blue, Red, Green, Orange
    else:
        # Use a colormap for many tables
        cmap = plt.cm.Set3
        return [cmap(i / num_tables) for i in range(num_tables)]

def create_sorted_summary(dfs: List[pd.DataFrame], table_names: List[str], 
                         metrics: List[str], summary_output: Optional[str] = None,
                         sort_by: Optional[str] = None) -> pd.DataFrame:
    """Create summary table sorted by specified metric or average performance."""
    
    all_summaries = []
    
    for df, table_name in zip(dfs, table_names):
        # Create a copy for processing
        summary_df = df.copy()
        
        # Check if we have the layout column
        if 'layout' not in summary_df.columns:
            print(f"Warning: No 'layout' column found in {table_name}, skipping")
            continue
        
        # Use existing letters/positions columns
        has_layout_data = 'letters' in summary_df.columns and 'positions' in summary_df.columns
        
        # Filter to only include layouts with sufficient data
        valid_layouts = []
        for _, row in summary_df.iterrows():
            valid_count = sum(1 for metric in metrics if metric in row and pd.notna(row[metric]))
            if valid_count >= len(metrics) * 0.5:  # Need at least 50% valid data
                valid_layouts.append(row.name)
        
        if not valid_layouts:
            print(f"Warning: No layouts with sufficient data in {table_name}")
            continue
        
        summary_df = summary_df.loc[valid_layouts].copy()
        
        # Calculate average score across all metrics
        valid_metrics = [metric for metric in metrics if metric in summary_df.columns]
        if valid_metrics:
            summary_df['average_score'] = summary_df[valid_metrics].mean(axis=1)
            # Calculate balanced average with std dev penalty (α = 0.5)
            # This penalizes layouts with wide tradeoffs across objectives
            summary_df['balanced_average'] = (
                summary_df[valid_metrics].mean(axis=1) - 
                0.5 * summary_df[valid_metrics].std(axis=1)
            )
        else:
            summary_df['average_score'] = 0
            summary_df['balanced_average'] = 0
        
        # Sort by specified metric if requested (descending - best first)
        if sort_by and sort_by in summary_df.columns:
            summary_df = summary_df.sort_values(sort_by, ascending=False)
        # Otherwise preserve original table order (no sorting by default)
        
        # Prepare output columns in requested order:
        # layout, layout_qwerty (or letters+positions), average_score, balanced_average, then original metrics
        output_columns = ['layout']
        
        # Add layout string column(s) - detect if letters is full QWERTY layout or MOO format
        if has_layout_data:
            # Check if letters is full 32-char QWERTY layout format
            sample_letters = summary_df['letters'].iloc[0] if len(summary_df) > 0 else ""
            if len(sample_letters) == 32:
                # Full QWERTY layout - rename to layout_qwerty for display_layouts.py compatibility
                summary_df = summary_df.rename(columns={'letters': 'layout_qwerty'})
                output_columns.append('layout_qwerty')
            else:
                # MOO format or other - keep both letters and positions
                output_columns.extend(['letters', 'positions'])
        
        # Add average score columns
        output_columns.append('average_score')
        output_columns.append('balanced_average')
        
        # Add original metric columns
        output_columns.extend(metrics)
        
        # Add table identifier if multiple tables
        if len(dfs) > 1:
            summary_df['table'] = table_name
            output_columns.insert(1, 'table')  # Insert after layout name
        
        # Select available columns only
        available_columns = [col for col in output_columns if col in summary_df.columns]
        
        # Select and store summary
        table_summary = summary_df[available_columns].copy()
        all_summaries.append(table_summary)
    
    if not all_summaries:
        print("No valid summary data found")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Combine all tables
    combined_summary = pd.concat(all_summaries, ignore_index=True)
    
    # If multiple tables and sort requested, sort globally by specified metric
    if len(dfs) > 1 and sort_by and sort_by in combined_summary.columns:
        combined_summary = combined_summary.sort_values(sort_by, ascending=False)
    # Otherwise preserve original table order
    
    # Round average scores to remove floating point precision issues
    if 'average_score' in combined_summary.columns:
        combined_summary['average_score'] = combined_summary['average_score'].round(4)
    if 'balanced_average' in combined_summary.columns:
        combined_summary['balanced_average'] = combined_summary['balanced_average'].round(4)
    
    # Add index column (1-based)
    combined_summary.insert(0, 'index', range(1, len(combined_summary) + 1))
    
    # Save to CSV if requested
    if summary_output:
        combined_summary.to_csv(summary_output, index=False)
        print(f"\nSummary saved to {summary_output}")
    
    # Print layouts with appropriate label
    if sort_by and sort_by in combined_summary.columns:
        print(f"Layouts (sorted by {sort_by}):")
        sort_col = sort_by
    else:
        print(f"Layouts (in original table order):")
        sort_col = 'average_score'  # For display purposes
    
    # Show top 10 layouts
    display_cols = ['layout']
    if 'table' in combined_summary.columns:
        display_cols.append('table')
    display_cols.append(sort_col)
    if sort_col != 'average_score' and 'average_score' in combined_summary.columns:
        display_cols.append('average_score')
    if 'balanced_average' in combined_summary.columns and 'balanced_average' not in display_cols:
        display_cols.append('balanced_average')
    
    top_layouts = combined_summary[display_cols].head(10)
    for i, (_, row) in enumerate(top_layouts.iterrows(), 1):
        if 'table' in row:
            if sort_col != 'average_score' and 'average_score' in row:
                if 'balanced_average' in row:
                    print(f"  {i:2d}. {row['layout']} ({row['table']}) - {sort_col}: {row[sort_col]:.3f}, Avg: {row['average_score']:.3f}, Balanced: {row['balanced_average']:.3f}")
                else:
                    print(f"  {i:2d}. {row['layout']} ({row['table']}) - {sort_col}: {row[sort_col]:.3f}, Avg: {row['average_score']:.3f}")
            else:
                if 'balanced_average' in row:
                    print(f"  {i:2d}. {row['layout']} ({row['table']}) - {sort_col}: {row[sort_col]:.3f}, Balanced: {row['balanced_average']:.3f}")
                else:
                    print(f"  {i:2d}. {row['layout']} ({row['table']}) - {sort_col}: {row[sort_col]:.3f}")
        else:
            if sort_col != 'average_score' and 'average_score' in row:
                if 'balanced_average' in row:
                    print(f"  {i:2d}. {row['layout']} - {sort_col}: {row[sort_col]:.3f}, Avg: {row['average_score']:.3f}, Balanced: {row['balanced_average']:.3f}")
                else:
                    print(f"  {i:2d}. {row['layout']} - {sort_col}: {row[sort_col]:.3f}, Avg: {row['average_score']:.3f}")
            else:
                if 'balanced_average' in row:
                    print(f"  {i:2d}. {row['layout']} - {sort_col}: {row[sort_col]:.3f}, Balanced: {row['balanced_average']:.3f}")
                else:
                    print(f"  {i:2d}. {row['layout']} - {sort_col}: {row[sort_col]:.3f}")
    
    return combined_summary

def create_heatmap_plot(dfs: List[pd.DataFrame], table_names: List[str], 
                       metrics: List[str], output_path: Optional[str] = None,
                       summary_df: Optional[pd.DataFrame] = None,
                       sort_by: Optional[str] = None,
                       is_normalized: bool = True) -> None:
    """Create heatmap visualization with layouts on y-axis and metrics on x-axis.
    
    Args:
        dfs: Dataframes (normalized or raw depending on is_normalized)
        is_normalized: If True, data is 0-1 normalized. If False, uses raw values.
    """
    
    # [Layout ordering logic remains the same - using summary_df or sorting within tables]
    if summary_df is not None and len(summary_df) > 0:
        all_data = []
        layout_labels = []
        
        layout_lookup = {}
        for df, table_name in zip(dfs, table_names):
            for _, row in df.iterrows():
                layout_name = row.get('layout', '')
                if layout_name:
                    metric_values = []
                    valid_count = 0
                    for metric in metrics:
                        if metric in row and pd.notna(row[metric]):
                            metric_values.append(row[metric])
                            valid_count += 1
                        else:
                            metric_values.append(0.0 if is_normalized else np.nan)
                    if valid_count >= len(metrics) * 0.5:
                        layout_qwerty = row.get('layout_qwerty', '') or row.get('letters', '')
                        layout_lookup[layout_name] = (metric_values, layout_qwerty)
        
        for _, row in summary_df.iterrows():
            layout_name = row.get('layout', '')
            if layout_name in layout_lookup:
                metric_values, layout_qwerty = layout_lookup[layout_name]
                all_data.append(metric_values)
                if layout_name and layout_name not in ('nan', '', 'None'):
                    layout_labels.append(layout_name)
                elif layout_qwerty and layout_qwerty not in ('nan', '', 'None'):
                    layout_labels.append(layout_qwerty)
                else:
                    layout_labels.append("layout")
    else:
        all_data = []
        layout_labels = []
        
        for i, (df, table_name) in enumerate(zip(dfs, table_names)):
            table_data = []
            table_layout_labels = []
            
            for _, row in df.iterrows():
                metric_values = []
                valid_count = 0
                
                for metric in metrics:
                    if metric in row and pd.notna(row[metric]):
                        metric_values.append(row[metric])
                        valid_count += 1
                    else:
                        metric_values.append(0.0 if is_normalized else np.nan)
                
                if valid_count < len(metrics) * 0.5:
                    continue
                
                table_data.append(metric_values)
                layout_qwerty = row.get('layout_qwerty', '') or row.get('letters', '')
                layout_name = row.get('layout', f'Layout_{len(table_layout_labels)+1}')
                if layout_qwerty and layout_qwerty not in ('nan', '', 'None'):
                    table_layout_labels.append(layout_qwerty)
                else:
                    table_layout_labels.append(layout_name)
            
            if not table_data:
                continue
            
            table_matrix = np.array(table_data)
            
            if sort_by and sort_by in metrics:
                sort_metric_idx = metrics.index(sort_by)
                sort_values = table_matrix[:, sort_metric_idx]
            else:
                sort_values = np.nanmean(table_matrix, axis=1)
            sort_indices = np.argsort(sort_values)[::-1]
            
            sorted_table_data = table_matrix[sort_indices]
            sorted_table_labels = [table_layout_labels[idx] for idx in sort_indices]
            
            all_data.extend(sorted_table_data.tolist())
            layout_labels.extend(sorted_table_labels)
    
    if not all_data:
        print("No valid data found for heatmap")
        return
    
    data_matrix = np.array(all_data)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(max(12, len(metrics) * 0.8), max(8, len(layout_labels) * 0.3)))
    
    # Create heatmap with appropriate scaling
    if is_normalized:
        # Normalized: use 0-1 scale with viridis
        im = ax.imshow(data_matrix, cmap='viridis', aspect='equal', vmin=0, vmax=1)
        cbar_label = 'Normalized Score (0 = worst, 1 = best)'
    else:
        # Raw: auto-scale to data range with RdYlGn (red=low, yellow=mid, green=high)
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='equal')
        cbar_label = 'Raw Metric Value'
    
    # Set ticks and labels
    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(layout_labels)))
    
    metric_display_names = []
    for metric in metrics:
        display_name = metric.replace('_', ' ').title()
        metric_display_names.append(display_name)
    
    ax.set_xticklabels(metric_display_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(layout_labels, fontsize=7, fontfamily='monospace')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)
    
    # Add value annotations for smaller matrices
    if len(layout_labels) <= 20 and len(metrics) <= 15:
        for i in range(len(layout_labels)):
            for j in range(len(metrics)):
                value = data_matrix[i, j]
                if is_normalized:
                    text_color = 'white' if value < 0.5 else 'black'
                    text = f'{value:.2f}'
                else:
                    # For raw values, use adaptive coloring based on colormap
                    vmin, vmax = im.get_clim()
                    normalized_val = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    text_color = 'white' if normalized_val < 0.5 else 'black'
                    text = f'{value:.2f}'
                
                if not np.isnan(value):
                    ax.text(j, i, text, ha='center', va='center', 
                           color=text_color, fontsize=7, weight='bold')
    
    # Title with sorting info
    data_type = "Normalized" if is_normalized else "Raw"
    if summary_df is not None and len(summary_df) > 0:
        if sort_by and sort_by in metrics:
            sort_info = f" (sorted by {sort_by})"
        else:
            sort_info = ""
    else:
        if sort_by and sort_by in metrics:
            sort_info = f" (sorted by {sort_by} within each table)"
        else:
            sort_info = ""
    
    title = f'Keyboard Layout Comparison Heatmap - {data_type}{sort_info}\n{len(layout_labels)} layouts across {len(metrics)} metrics'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xlabel('Scoring Methods', fontsize=12)
    ax.set_ylabel('Keyboard Layouts', fontsize=12)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        suffix = '_normalized' if is_normalized else '_raw'
        if output_path.endswith('.png'):
            heatmap_path = output_path.replace('.png', f'_heatmap{suffix}.png')
        else:
            heatmap_path = output_path + f'_heatmap{suffix}.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Heatmap ({data_type}) saved to {heatmap_path}")
    else:
        plt.show()
    
    plt.close()


def create_parallel_plot(dfs: List[pd.DataFrame], table_names: List[str], 
                        metrics: List[str], output_path: Optional[str] = None,
                        summary_df: Optional[pd.DataFrame] = None,
                        is_normalized: bool = True) -> None:
    """Create parallel coordinates plot with performance-based coloring.
    
    Args:
        dfs: Dataframes (normalized or raw depending on is_normalized)
        is_normalized: If True, data is 0-1 normalized. If False, uses raw values.
    """
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(max(12, len(metrics) * 1.2), 10))
    
    # Determine coloring scheme
    use_performance_colors = summary_df is not None and len(summary_df) > 0
    use_two_table_colors = len(dfs) == 2 and use_performance_colors
    
    if use_performance_colors:
        layout_to_position = {}
        layout_to_table = {}
        
        for idx, (_, row) in enumerate(summary_df.iterrows()):
            layout_name = row['layout']
            layout_to_position[layout_name] = idx
            if 'table' in row:
                layout_to_table[layout_name] = row['table']
        
        total_layouts = len(layout_to_position)
        
        if use_two_table_colors:
            print(f"Using two-table coloring: {table_names[0]} (grayscale), {table_names[1]} (red)")
            
            gray_colormap = cm.Grays
            gray_min = 0.3
            gray_max = 0.9
            
            red_colors = [
                (1.0, 0.8, 0.8),
                (1.0, 0.4, 0.4),
                (1.0, 0.0, 0.0),
                (0.8, 0.0, 0.0),
            ]
            red_colormap = LinearSegmentedColormap.from_list('custom_red', red_colors)
            red_min = 0.0
            red_max = 1.0
            
        else:
            print(f"Using performance-based coloring (grayscale) for {total_layouts} layouts")
            
            gray_colormap = cm.Grays
            gray_min = 0.3
            gray_max = 0.9
    else:
        colors = get_table_colors(len(dfs))
    
    x_positions = range(len(metrics))
    
    # Calculate y-axis limits for raw data
    if not is_normalized:
        all_values = []
        for df in dfs:
            for metric in metrics:
                if metric in df.columns:
                    valid_vals = df[metric].dropna()
                    all_values.extend(valid_vals.tolist())
        
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            y_padding = (y_max - y_min) * 0.05
            y_limits = (y_min - y_padding, y_max + y_padding)
        else:
            y_limits = (0, 1)
    
    # Plot each table's data
    for i, (df, table_name) in enumerate(zip(dfs, table_names)):
        valid_layout_count = 0
        
        if not use_performance_colors:
            color = colors[i] if i < len(colors) else colors[-1]
        
        for _, row in df.iterrows():
            y_values = [row.get(metric, np.nan) for metric in metrics]
            
            valid_values = [val for val in y_values if pd.notna(val)]
            if len(valid_values) < len(metrics) * 0.5:
                continue
            
            # For raw data, keep NaN as NaN; for normalized, replace with 0
            if not is_normalized:
                pass  # Keep NaN values
            else:
                y_values = [val if pd.notna(val) else 0 for val in y_values]
            
            if use_performance_colors:
                layout_name = row.get('layout', '')
                if layout_name in layout_to_position:
                    position = layout_to_position[layout_name]
                    
                    if use_two_table_colors:
                        table_idx = None
                        if layout_name in layout_to_table:
                            layout_table = layout_to_table[layout_name]
                            table_idx = 0 if layout_table == table_names[0] else 1
                        else:
                            table_idx = i
                        
                        if table_idx == 0:
                            color_intensity = gray_max - (position / max(1, total_layouts - 1)) * (gray_max - gray_min)
                            color = gray_colormap(color_intensity)
                        else:
                            color_intensity = red_max - (position / max(1, total_layouts - 1)) * (red_max - red_min)
                            color = red_colormap(color_intensity)
                    else:
                        color_intensity = gray_max - (position / max(1, total_layouts - 1)) * (gray_max - gray_min)
                        color = gray_colormap(color_intensity)
                else:
                    color = 'gray'
            
            ax.plot(x_positions, y_values, color=color, alpha=0.7, linewidth=1.5)
            valid_layout_count += 1
        
        # Add legend entry
        if not use_performance_colors:
            ax.plot([], [], color=color, linewidth=3, label=f"{table_name} ({valid_layout_count} layouts)")
        elif use_two_table_colors:
            if i == 0:
                ax.plot([], [], color=gray_colormap(gray_max), linewidth=3, label=f"{table_names[0]} - Best")
                ax.plot([], [], color=gray_colormap(gray_min), linewidth=3, label=f"{table_names[0]} - Worst")
            else:
                ax.plot([], [], color=red_colormap(red_max), linewidth=3, label=f"{table_names[1]} - Best")
                ax.plot([], [], color=red_colormap(red_min), linewidth=3, label=f"{table_names[1]} - Worst")
        elif i == 0:
            ax.plot([], [], color=gray_colormap(gray_max), linewidth=3, label=f"Best performing layouts")
            ax.plot([], [], color=gray_colormap(gray_min), linewidth=3, label=f"Worst performing layouts")

    # Customize the plot
    ax.set_xlim(-0.5, len(metrics) - 0.5)
    
    if is_normalized:
        ax.set_ylim(-0.05, 1.05)
        ylabel = 'Normalized Score (0 = worst, 1 = best)'
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    else:
        ax.set_ylim(y_limits)
        ylabel = 'Raw Metric Value'
        # Auto-generate y-ticks for raw data
    
    # Set x-axis labels
    metric_display_names = []
    for metric in metrics:
        display_name = metric.replace('_', ' ').title()
        if len(display_name) > 12:
            words = display_name.split()
            if len(words) > 1:
                mid = len(words) // 2
                display_name = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
        metric_display_names.append(display_name)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_display_names, rotation=45, ha='right', fontsize=10)
    
    # Add vertical grid lines
    for x in x_positions:
        ax.axvline(x, color='gray', alpha=0.3, linewidth=0.5)
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Title
    data_type = "Normalized" if is_normalized else "Raw"
    if use_performance_colors:
        if use_two_table_colors:
            if summary_df is not None and 'average_score' in summary_df.columns:
                sort_metric = None
                for metric in metrics:
                    if metric in summary_df.columns:
                        sorted_values = summary_df.sort_values(by=metric, ascending=False)[metric].reset_index(drop=True)
                        current_values = summary_df[metric].reset_index(drop=True)
                        if sorted_values.equals(current_values):
                            sort_metric = metric
                            break
                
                if sort_metric:
                    title_suffix = f'\nPerformance-ordered by {sort_metric}\n{table_names[0]} (grayscale): dark = best, light = worst | {table_names[1]} (red): saturated red = best, light pink = worst'
                else:
                    title_suffix = f'\nPerformance-ordered by average\n{table_names[0]} (grayscale): dark = best, light = worst | {table_names[1]} (red): saturated red = best, light pink = worst'
            else:
                title_suffix = f'\n{table_names[0]} (grayscale) vs {table_names[1]} (red): dark gray/saturated red = best'
        elif summary_df is not None and 'average_score' in summary_df.columns:
            sort_metric = None
            for metric in metrics:
                if metric in summary_df.columns:
                    sorted_values = summary_df.sort_values(by=metric, ascending=False)[metric].reset_index(drop=True)
                    current_values = summary_df[metric].reset_index(drop=True)
                    if sorted_values.equals(current_values):
                        sort_metric = metric
                        break
            
            if sort_metric:
                title_suffix = f'\nPerformance-ordered by {sort_metric} (grayscale): dark gray = best, light gray = worst'
            else:
                title_suffix = f'\nPerformance-ordered by average (grayscale): dark gray = best, light gray = worst'
        else:
            title_suffix = f'\nPerformance-ordered visualization (grayscale): dark gray = best, light gray = worst'
    else:
        title_suffix = f'\nParallel coordinates across {len(metrics)} scoring methods'
    
    ax.set_title(f'Keyboard Layout Comparison - {data_type}{title_suffix}', 
                fontsize=16, fontweight='bold', pad=20)

    if len(dfs) > 1 or len(dfs[0]) <= 10 or use_performance_colors:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        suffix = '_normalized' if is_normalized else '_raw'
        if output_path.endswith('.png'):
            parallel_path = output_path.replace('.png', f'_parallel{suffix}.png')
        else:
            parallel_path = output_path + f'_parallel{suffix}.png'
        plt.savefig(parallel_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
        print(f"Parallel plot ({data_type}) saved to {parallel_path}")
    else:
        plt.show()
    
    plt.close()

def plot_objective_scatter(dfs: List[pd.DataFrame], table_names: List[str],
                          metrics: List[str], output_path: Optional[str] = None,
                          sort_by: Optional[str] = None) -> None:
    """Create scatter plot of objective scores."""
    if not metrics:
        return
    
    # Combine all dataframes
    all_data = []
    for df, table_name in zip(dfs, table_names):
        df_copy = df.copy()
        if len(dfs) > 1:
            df_copy['table'] = table_name
        all_data.append(df_copy)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Determine sort column
    if sort_by and sort_by in metrics:
        sort_col = sort_by
    else:
        sort_col = metrics[0]  # Default to first metric
    
    # Sort by selected metric
    df_sorted = combined_df.sort_values(sort_col, ascending=False)
    
    plt.figure(figsize=(14, 8))
    
    colors = get_colors(len(metrics))
    x_positions = range(len(df_sorted))
    
    for i, metric in enumerate(metrics):
        if metric in df_sorted.columns:
            plt.scatter(x_positions, df_sorted[metric], 
                       marker='.', s=1, alpha=0.7, 
                       label=metric, color=colors[i])
    
    plt.xlabel(f'Solution Index (sorted by {sort_col})')
    plt.ylabel('Objective Score')
    plt.title(f'Multi-Objective Scores ({len(df_sorted):,} solutions)')
    
    try:
        legend = plt.legend(markerscale=20, frameon=True)
        if legend:
            legend.get_frame().set_alpha(0.9)
    except Exception as e:
        print(f"Warning: Could not create legend: {e}")
        
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        if output_path.endswith('.png'):
            scatter_path = output_path.replace('.png', '_scatter.png')
        else:
            scatter_path = output_path + '_scatter.png'
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to {scatter_path}")
    else:
        plt.show()
    
    plt.close()

def plot_pareto_front_2d(dfs: List[pd.DataFrame], table_names: List[str],
                        metrics: List[str], output_path: Optional[str] = None) -> None:
    """Create 2D Pareto front if we have exactly 2 objectives."""
    if len(metrics) != 2:
        return
    
    # Combine all dataframes
    all_data = []
    for df, table_name in zip(dfs, table_names):
        df_copy = df.copy()
        if len(dfs) > 1:
            df_copy['table'] = table_name
        all_data.append(df_copy)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    x_col, y_col = metrics[0], metrics[1]
    has_tables = 'table' in combined_df.columns and len(dfs) > 1
    
    plt.figure(figsize=(12, 8))
    
    if has_tables:
        # Color by table
        colors = get_colors(len(table_names))
        table_color_map = dict(zip(table_names, colors))
        
        for table_name in table_names:
            table_data = combined_df[combined_df['table'] == table_name]
            plt.scatter(table_data[x_col], table_data[y_col], 
                       alpha=0.6, s=50, label=f'{table_name} ({len(table_data)})',
                       color=table_color_map[table_name])
        
        try:
            legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
            if legend:
                legend.get_frame().set_alpha(0.9)
        except Exception as e:
            print(f"Warning: Could not create legend: {e}")
            
        title_suffix = " - Colored by Table"
    else:
        plt.scatter(combined_df[x_col], combined_df[y_col], 
                   alpha=0.6, s=50, color='blue')
        title_suffix = ""
    
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'2D Objective Space{title_suffix} ({len(combined_df):,} solutions)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        if output_path.endswith('.png'):
            pareto_path = output_path.replace('.png', '_pareto2d.png')
        else:
            pareto_path = output_path + '_pareto2d.png'
        plt.savefig(pareto_path, dpi=300, bbox_inches='tight')
        print(f"2D Pareto plot saved to {pareto_path}")
    else:
        plt.show()
    
    plt.close()

def plot_correlation_matrix(dfs: List[pd.DataFrame], table_names: List[str],
                           metrics: List[str], output_path: Optional[str] = None) -> None:
    """Create correlation matrix of objectives."""
    if len(metrics) < 2:
        return
    
    # Combine all dataframes
    all_data = pd.concat(dfs, ignore_index=True)
    
    # Get objective data
    obj_data = all_data[metrics].dropna()
    if obj_data.empty:
        return
        
    corr_matrix = obj_data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
               square=True, linewidths=0.5, fmt='.2f')
    plt.title(f'Metric Correlation Matrix ({len(obj_data):,} solutions)')
    plt.tight_layout()
    
    if output_path:
        if output_path.endswith('.png'):
            corr_path = output_path.replace('.png', '_correlation.png')
        else:
            corr_path = output_path + '_correlation.png'
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved to {corr_path}")
    else:
        plt.show()
    
    plt.close()

def plot_stability_matrix(dfs: List[pd.DataFrame], table_names: List[str],
                         output_path: Optional[str] = None) -> None:
    """Create letter-position stability heatmap.
    
    Only works with MOO format data (items + positions where positions are keyboard keys).
    Skips if data format is not suitable.
    """
    # Combine all dataframes
    all_data = pd.concat(dfs, ignore_index=True)
    
    # Check if we have 'items' and 'positions' columns
    # Note: load_layout_data() renames 'items' to 'letters', but we need the original MOO format
    has_items = 'items' in all_data.columns
    has_letters = 'letters' in all_data.columns
    has_positions = 'positions' in all_data.columns
    
    if not has_positions:
        # No positions data at all - can't create stability matrix
        return
    
    # Determine which column to use for letters
    letters_col = 'items' if has_items else ('letters' if has_letters else None)
    
    if letters_col is None:
        # No letter data - can't create stability matrix
        return
    
    # Check if this is MOO format data by examining samples
    # MOO format has:
    # - items/letters: short strings like "etaoinsrhldcum" (letters in assignment order)
    # - positions: keyboard keys like "KJ;ASDVRLFUEIM" (where they go)
    # 
    # Non-MOO format has:
    # - letters: full 32-char QWERTY layout like "  cr  du  oinl  teha   s  m     "
    # - positions: QWERTY reference (all same) or missing
    
    valid_rows = 0
    for _, row in all_data.head(10).iterrows():  # Check first 10 rows
        if pd.notna(row.get(letters_col)) and pd.notna(row.get('positions')):
            letters_str = str(row[letters_col])
            positions_str = str(row['positions'])
            # MOO format: both should be short (< 30 chars) and positions should be mixed keys
            if len(letters_str) < 30 and len(positions_str) < 30 and not positions_str.startswith('QWERTYUIOP'):
                valid_rows += 1
    
    if valid_rows == 0:
        # Not MOO format data - skip stability matrix
        return
    
    print("Creating stability matrix...")
    
    # Calculate assignment frequencies
    letter_positions = defaultdict(set)
    position_letters = defaultdict(set)
    assignment_counts = defaultdict(int)
    
    rows_processed = 0
    for _, row in all_data.iterrows():
        # Skip rows with missing or invalid data
        if pd.isna(row.get(letters_col)) or pd.isna(row.get('positions')):
            continue
        
        # Convert to string
        letters_str = str(row[letters_col])
        positions_str = str(row['positions'])
        
        # Skip if conversion resulted in 'nan' or empty
        if letters_str in ('nan', '', 'None') or positions_str in ('nan', '', 'None'):
            continue
        
        # Skip if this looks like full QWERTY layout (not MOO format)
        if len(letters_str) > 30 or len(positions_str) > 30:
            continue
        
        # Convert to lists
        try:
            items = list(letters_str)
            positions = list(positions_str)
            
            # Skip if lengths don't match
            if len(items) != len(positions):
                continue
            
            for letter, position in zip(items, positions):
                # Skip whitespace
                if letter.strip() and position.strip():
                    letter_positions[letter].add(position)
                    position_letters[position].add(letter)
                    assignment_counts[(letter, position)] += 1
            
            rows_processed += 1
            
        except (TypeError, ValueError, AttributeError):
            continue
    
    # Check if we collected any data
    if not letter_positions or not position_letters or rows_processed == 0:
        print("Skipping stability matrix: no MOO format data found")
        return
    
    print(f"Processed {rows_processed} layouts for stability matrix")
    
    # Sort by stability (fewer positions/letters = more stable)
    letters_by_stability = sorted(letter_positions.keys(), 
                                key=lambda x: len(letter_positions[x]))
    positions_by_stability = sorted(position_letters.keys(), 
                                  key=lambda x: len(position_letters[x]))
    
    # Create matrix
    matrix = np.zeros((len(letters_by_stability), len(positions_by_stability)))
    letter_indices = {letter: i for i, letter in enumerate(letters_by_stability)}
    position_indices = {pos: i for i, pos in enumerate(positions_by_stability)}
    
    for (letter, position), count in assignment_counts.items():
        if letter in letter_indices and position in position_indices:
            matrix[letter_indices[letter], position_indices[position]] = count
    
    # Plot heatmap
    plt.figure(figsize=(16, 12))
    annot_matrix = np.where(matrix == 0, '', matrix.astype(int).astype(str))
    
    sns.heatmap(matrix, 
               xticklabels=positions_by_stability,
               yticklabels=letters_by_stability,
               cmap='Reds', linewidths=0.5, square=True,
               annot=annot_matrix, fmt='', annot_kws={'size': 8},
               cbar_kws={'label': 'Assignment Count'})
    
    plt.title(f'Letter-Position Stability Matrix ({len(all_data):,} solutions)')
    plt.xlabel('Positions (ordered by stability)')
    plt.ylabel('Letters (ordered by stability)')
    plt.tick_params(left=False, bottom=False)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if output_path:
        if output_path.endswith('.png'):
            stability_path = output_path.replace('.png', '_stability.png')
        else:
            stability_path = output_path + '_stability.png'
        plt.savefig(stability_path, dpi=300, bbox_inches='tight')
        print(f"Stability matrix saved to {stability_path}")
    else:
        plt.show()
    
    plt.close()

def generate_report(dfs: List[pd.DataFrame], table_names: List[str], 
                   metrics: List[str], summary_df: Optional[pd.DataFrame] = None,
                   output_dir: Path = Path('../output')) -> str:
    """Generate comprehensive analysis report."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / f'layouts_compare_report_{timestamp}.txt'
    
    # Combine all dataframes for overall stats
    all_data = pd.concat(dfs, ignore_index=True)
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("KEYBOARD LAYOUT COMPARISON REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total solutions analyzed: {len(all_data):,}\n")
        f.write(f"Number of tables: {len(dfs)}\n")
        f.write(f"Metrics analyzed: {len(metrics)}\n\n")
        
        # Table summary
        f.write("-"*70 + "\n")
        f.write("DATA SOURCES\n")
        f.write("-"*70 + "\n")
        for i, (df, name) in enumerate(zip(dfs, table_names), 1):
            f.write(f"{i}. {name}: {len(df):,} layouts\n")
        f.write("\n")
        
        # Metrics summary
        f.write("-"*70 + "\n")
        f.write("METRICS ANALYZED\n")
        f.write("-"*70 + "\n")
        for i, metric in enumerate(metrics, 1):
            f.write(f"{i:2d}. {metric}\n")
        f.write("\n")
        
        # Statistical summary
        f.write("-"*70 + "\n")
        f.write("METRIC STATISTICS\n")
        f.write("-"*70 + "\n")
        for metric in metrics:
            if metric in all_data.columns:
                values = all_data[metric].dropna()
                f.write(f"\n{metric}:\n")
                f.write(f"  Count:  {len(values):,}\n")
                f.write(f"  Mean:   {values.mean():.6f}\n")
                f.write(f"  Std:    {values.std():.6f}\n")
                f.write(f"  Min:    {values.min():.6f}\n")
                f.write(f"  25%:    {values.quantile(0.25):.6f}\n")
                f.write(f"  Median: {values.median():.6f}\n")
                f.write(f"  75%:    {values.quantile(0.75):.6f}\n")
                f.write(f"  Max:    {values.max():.6f}\n")
        f.write("\n")
        
        # Top performers
        if summary_df is not None and len(summary_df) > 0:
            f.write("-"*70 + "\n")
            f.write("TOP 20 PERFORMING LAYOUTS (by average score)\n")
            f.write("-"*70 + "\n")
            display_cols = ['layout']
            if 'table' in summary_df.columns:
                display_cols.append('table')
            display_cols.append('average_score')
            if 'balanced_average' in summary_df.columns:
                display_cols.append('balanced_average')
            display_cols.extend([m for m in metrics if m in summary_df.columns][:3])
            
            top_20 = summary_df[display_cols].head(20)
            f.write(top_20.to_string(index=False))
            f.write("\n\n")
        
        # Correlation insights
        if len(metrics) >= 2:
            obj_data = all_data[metrics].dropna()
            if not obj_data.empty:
                corr_matrix = obj_data.corr()
                f.write("-"*70 + "\n")
                f.write("METRIC CORRELATIONS\n")
                f.write("-"*70 + "\n")
                f.write("\nHighest positive correlations:\n")
                # Get upper triangle of correlation matrix
                corr_pairs = []
                for i in range(len(metrics)):
                    for j in range(i+1, len(metrics)):
                        corr_pairs.append((metrics[i], metrics[j], corr_matrix.iloc[i, j]))
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                for metric1, metric2, corr in corr_pairs[:5]:
                    f.write(f"  {metric1} <-> {metric2}: {corr:.3f}\n")
                f.write("\n")
        
        # Stability insights (if available)
        # After load_layout_data(), 'items' has been renamed to 'letters'
        has_stability_data = False
        letters_col = None
        
        if 'letters' in all_data.columns and 'positions' in all_data.columns:
            # Check if this is MOO format (not just layout_qwerty format)
            sample_row = all_data.iloc[0] if len(all_data) > 0 else None
            if sample_row is not None and pd.notna(sample_row['positions']):
                sample_positions = str(sample_row['positions'])
                # If positions is not standard QWERTY order, it's probably MOO format
                if not sample_positions.startswith('QWERTYUIOP'):
                    has_stability_data = True
                    letters_col = 'letters'
        elif 'items' in all_data.columns and 'positions' in all_data.columns:
            has_stability_data = True
            letters_col = 'items'
        
        if has_stability_data:
            f.write("-"*70 + "\n")
            f.write("LETTER-POSITION STABILITY\n")
            f.write("-"*70 + "\n")
            
            letter_positions = defaultdict(set)
            position_letters = defaultdict(set)
            
            for _, row in all_data.iterrows():
                if pd.isna(row[letters_col]) or pd.isna(row['positions']):
                    continue
                try:
                    items = list(str(row[letters_col]))
                    positions = list(str(row['positions']))
                    for letter, position in zip(items, positions):
                        letter_positions[letter].add(position)
                        position_letters[position].add(letter)
                except (TypeError, ValueError):
                    continue
            
            if letter_positions and position_letters:
                # Most stable letters
                letters_by_stability = sorted(letter_positions.keys(), 
                                            key=lambda x: len(letter_positions[x]))
                f.write("\nMost stable letters (fewest positions):\n")
                for letter in letters_by_stability[:10]:
                    positions_list = sorted(letter_positions[letter])
                    f.write(f"  {letter}: {len(positions_list)} positions - {', '.join(positions_list)}\n")
                
                # Most stable positions
                positions_by_stability = sorted(position_letters.keys(), 
                                              key=lambda x: len(position_letters[x]))
                f.write("\nMost stable positions (fewest letters):\n")
                for position in positions_by_stability[:10]:
                    letters_list = sorted(position_letters[position])
                    f.write(f"  {position}: {len(letters_list)} letters - {', '.join(letters_list)}\n")
                f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"\nAnalysis report saved: {report_path}")
    return str(report_path)

def print_summary_stats(dfs: List[pd.DataFrame], table_names: List[str], metrics: List[str]) -> None:
    """Print summary statistics for the loaded data."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for df, name in zip(dfs, table_names):
        print(f"\n{name}:")
        print(f"  Layouts: {len(df)}")
        
        # Count missing metrics for available metrics only
        available_metrics = [m for m in metrics if m in df.columns]
        if available_metrics:
            missing_counts = df[available_metrics].isnull().sum()
            if missing_counts.sum() > 0:
                print(f"  Missing data points: {missing_counts.sum()}")
        
        # Sample layout names
        if 'layout' in df.columns:
            sample_layouts = [str(x) for x in df['layout'].head(3).tolist()]
            print(f"  Sample layouts: {', '.join(sample_layouts)}")
        else:
            print("  Warning: No 'layout' column found")

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive keyboard layout comparison and visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison with all available metrics (assumed positive)
  python layouts_compare.py --tables layouts.csv
  
  # Specify positive and negative metrics explicitly
  python layouts_compare.py --tables scores1.csv scores2.csv \\
      --positive-metrics comfort rolls dvorak7 \\
      --negative-metrics same-finger_bigrams scissors lateral_stretch_bigrams \\
      --plot --report
  
  # Create summary table sorted by performance
  python layouts_compare.py --tables layouts.csv \\
      --positive-metrics comfort rolls \\
      --negative-metrics same-finger_bigrams \\
      --summary results.csv
  
  # Full analysis with custom sort
  python layouts_compare.py --tables layouts.csv \\
      --positive-metrics comfort rolls redirects \\
      --negative-metrics scissors skipgrams \\
      --sort-by comfort --plot --report --summary summary.csv

Supported CSV formats (auto-detected):
  - Preferred: layout,layout_qwerty,metric1,metric2,...
  - Standard:  layout,letters,positions,metric1,metric2,...  
  - MOO:       config_id,items,positions,metric1,metric2,...
        """
    )
    
    parser.add_argument('--tables', nargs='+', required=True,
                       help='One or more CSV files with layout data')
    parser.add_argument('--positive-metrics', nargs='*',
                       help='Metrics where HIGHER is BETTER (e.g., comfort, rolls). Normalized as-is.')
    parser.add_argument('--negative-metrics', nargs='*',
                       help='Metrics where LOWER is BETTER (e.g., same-finger_bigrams, scissors). Inverted during normalization.')
    parser.add_argument('--metrics', nargs='*',
                       help='(Deprecated) Metrics to include, assumed positive. Use --positive-metrics instead.')
    parser.add_argument('--output', '-o', 
                       help='Output file path for plots (generates multiple files with suffixes)')
    parser.add_argument('--summary', 
                       help='Create summary table sorted by average performance and save to CSV')
    parser.add_argument('--plot', action='store_true',
                       help='Generate additional visualization plots (scatter, Pareto, correlation, stability)')
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive analysis report')
    parser.add_argument('--sort-by',
                       help='Metric to sort by in all visualizations (heatmap, parallel, scatter). Default: average of all metrics')
    parser.add_argument('--output-dir', default='../output',
                       help='Output directory for report and plots without explicit paths')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed information')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate files exist
    for table_path in args.tables:
        if not Path(table_path).exists():
            print(f"Error: File '{table_path}' not found")
            sys.exit(1)
    
    # Load data
    dfs = []
    table_names = []
    
    for table_path in args.tables:
        df = load_layout_data(table_path, args.verbose)
        
        if len(df) == 0:
            if args.verbose:
                print(f"Warning: No data found in {table_path}")
            continue
            
        dfs.append(df)
        table_names.append(Path(table_path).stem)
    
    if not dfs:
        print("Error: No valid data found in any table")
        sys.exit(1)
    
    # Process metrics arguments
    positive_metrics = []
    negative_metrics = []
    
    # Handle new arguments (--positive-metrics and --negative-metrics)
    if args.positive_metrics is not None or args.negative_metrics is not None:
        # Use new arguments
        if args.positive_metrics:
            positive_metrics = filter_and_order_metrics(dfs, args.positive_metrics, args.verbose)
        if args.negative_metrics:
            negative_metrics = filter_and_order_metrics(dfs, args.negative_metrics, args.verbose)
        
        # Warn if old --metrics argument is also provided
        if args.metrics is not None:
            print("Warning: --metrics is deprecated and will be ignored. Use --positive-metrics and --negative-metrics instead.")
    
    # Fall back to old --metrics argument (assumed positive) for backwards compatibility
    elif args.metrics is not None:
        positive_metrics = filter_and_order_metrics(dfs, args.metrics, args.verbose)
        if args.verbose:
            print("Note: Using deprecated --metrics argument (assumed positive). Consider using --positive-metrics instead.")
    
    # If no metrics specified at all, use all available metrics as positive
    else:
        positive_metrics = filter_and_order_metrics(dfs, None, args.verbose)
        if args.verbose:
            print("Note: No metrics specified. Using all available metrics as positive.")
    
    # Combine for convenience
    all_metrics = positive_metrics + negative_metrics
    
    if not all_metrics:
        print("Error: No valid metrics found")
        sys.exit(1)
    
    if args.verbose:
        if positive_metrics:
            print(f"\nPositive metrics (higher is better): {len(positive_metrics)}")
            for i, m in enumerate(positive_metrics, 1):
                print(f"  {i:2d}. {m}")
        if negative_metrics:
            print(f"\nNegative metrics (lower is better, will be inverted): {len(negative_metrics)}")
            for i, m in enumerate(negative_metrics, 1):
                print(f"  {i:2d}. {m}")
    
    # Print summary
    if args.verbose:
        print_summary_stats(dfs, table_names, all_metrics)

    # Always create summary table (needed for performance-based coloring and sorting)
    if args.verbose:
        print(f"\nCreating performance summary...")
    normalized_dfs = normalize_data(dfs, positive_metrics, negative_metrics)
    summary_df = create_sorted_summary(normalized_dfs, table_names, all_metrics, 
                                      args.summary if args.summary else None,
                                      args.sort_by)
    
    # Create core plots - BOTH raw and normalized versions
    if args.verbose:
        print(f"\nCreating core visualization plots (raw and normalized)...")
        print(f"Tables: {len(dfs)}")
        print(f"Total layouts: {sum(len(df) for df in dfs)}")
        print(f"Metrics to plot: {len(all_metrics)} - {', '.join(all_metrics)}")
        if args.sort_by:
            if args.sort_by in all_metrics:
                print(f"Sorting by: {args.sort_by}")
            else:
                print(f"Warning: sort-by metric '{args.sort_by}' not in selected metrics, using average")
    
    # Determine output path
    plot_output = args.output if args.output else str(output_dir / 'layouts_compare.png')
    
    # Generate RAW (unnormalized) plots
    create_parallel_plot(dfs, table_names, all_metrics, plot_output, summary_df, is_normalized=False)
    create_heatmap_plot(dfs, table_names, all_metrics, plot_output, summary_df, args.sort_by, is_normalized=False)
    
    # Generate NORMALIZED plots
    create_parallel_plot(normalized_dfs, table_names, all_metrics, plot_output, summary_df, is_normalized=True)
    create_heatmap_plot(normalized_dfs, table_names, all_metrics, plot_output, summary_df, args.sort_by, is_normalized=True)
    
    # Generate additional plots if requested (these use original data)
    if args.plot:
        if args.verbose:
            print(f"\nCreating additional visualization plots...")
        
        plot_objective_scatter(dfs, table_names, all_metrics, plot_output, args.sort_by)
        plot_pareto_front_2d(dfs, table_names, all_metrics, plot_output)
        plot_correlation_matrix(dfs, table_names, all_metrics, plot_output)
        plot_stability_matrix(dfs, table_names, plot_output)
    
    # Generate report if requested
    if args.report:
        generate_report(dfs, table_names, all_metrics, summary_df, output_dir)
    
    print(f"\nAnalysis complete!")
    print(f"\nGenerated plots:")
    print(f"  - Parallel coordinate plot (raw): *_parallel_raw.png")
    print(f"  - Parallel coordinate plot (normalized): *_parallel_normalized.png")
    print(f"  - Heatmap (raw): *_heatmap_raw.png")
    print(f"  - Heatmap (normalized): *_heatmap_normalized.png")
    if args.plot:
        print(f"  - Additional plots: scatter, pareto2d, correlation, stability")


if __name__ == "__main__":
    main()
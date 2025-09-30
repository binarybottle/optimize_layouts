#!/usr/bin/env python3
"""
Keyboard layout comparison with metric filtering and average-based sorting

Creates parallel coordinates and heatmap plots comparing keyboard layouts 
across performance metrics, and allows filtering to specific metrics in a specified order.
Layouts are automatically sorted by average performance across selected metrics.

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

Core metrics are recommended by default. Experimental distance/efficiency 
and time/speed metrics can be included but have significant limitations:
- Distance metrics oversimplify biomechanics (ignore lateral stretching, finger strength, etc.)
- Time metrics contain QWERTY practice bias from empirical data

Examples:
    # Standard format with layout_qwerty column (preferred)
    python compare_layouts.py --tables layout_scores.csv

    # MOO format with items + positions (auto-converted)
    python compare_layouts.py --tables moo_results.csv --metrics engram dvorak7 comfort

    # Core metrics only (recommended)
    python compare_layouts.py \
        --metrics engram_avg4_score dvorak7 comfort_combo comfort comfort_key \
        --tables layout_scores.csv

    # Include experimental distance/time metrics (caution: limitations noted above)
    python compare_layouts.py --metrics engram_avg4_score comfort comfort_key dvorak7 \
        --tables layout_scores.csv --experimental-metrics

    # Create plots with specific metrics and save summary
    python compare_layouts.py --tables layouts.csv \
        --metrics engram_key_preference engram_row_separation engram_same_row engram_same_finger engram_outside \
        --summary summary.csv
    
    # Compare multiple tables with core metrics
    python compare_layouts.py --metrics engram_avg4_score comfort comfort_key dvorak7 \
        --output output/compare_layouts.png --tables layout_scores1.csv layout_scores2.csv

    # Study
    poetry run python3 compare_layouts.py \
        --metrics engram_key_preference engram_row_separation engram_same_row engram_same_finger engram_avg4_score \
        --output ../output/compare_layouts.png \
        --tables ../output/layouts_filter_patterns.csv ../output/moo2-1in4-4in8-9in16/layouts_compare_results.csv
    

Input format examples:
  
  Preferred format:
  layout,layout_qwerty,engram,dvorak7,comfort
  Dvorak,',.pyfgcrlaeoiduhtns;qjkxbmwvz,0.712,0.698,0.654
  
  Standard format:
  layout,letters,positions,engram,dvorak7,comfort
  Dvorak,',.pyfgcrlaeoiduhtns;qjkxbmwvz,QWERTYUIOPASDFGHJKL;ZXCVBNM,./[',0.712,0.698,0.654
  
  MOO format (auto-converted):
  config_id,items,positions,engram_key_preference,engram_avg4_score
  2438,etaoinsrhldcum,KJ;ASDVRLFUEIM,0.742,0.960

Summary output:
  CSV with columns: index, layout, [layout_qwerty, positions], average_score, [metric_values]
  - index: 1-based ranking by performance
  - layout_qwerty: what letter is at each QWERTY position (preferred format)
  - positions: QWERTY reference positions  
  Layouts ordered by average performance across selected metrics (higher = better)
  
Performance-based coloring:
  Parallel plot lines are colored from dark red (best) to light red (worst) based on average performance.

"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys
from typing import List, Dict, Tuple, Optional
import matplotlib.cm as cm

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
        if 'layout' not in df.columns:
            raise ValueError("Missing required column: 'layout'")
        
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
        if letters_col != 'letters':
            df = df.rename(columns={letters_col: 'letters'})
        if positions_col and positions_col != 'positions':
            df = df.rename(columns={positions_col: 'positions'})
        
        # Note: The script already handles the conversion internally in parse_layout_string
        # if needed for MOO format items+positions
        
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
            if col not in ['layout', 'letters', 'positions'] and pd.api.types.is_numeric_dtype(df[col]):
                all_metrics.add(col)
    
    # Convert to sorted list (alphabetical order)
    available_metrics = sorted(list(all_metrics))
    
    if verbose:
        print(f"\nFound {len(available_metrics)} scorer metrics available:")
        
        # Categorize metrics for better display
        core_metrics = []
        experimental_metrics = []
        
        for metric in available_metrics:
            metric_lower = metric.lower()
            if (metric_lower.startswith('distance') or metric_lower.startswith('efficiency') or
                metric_lower.startswith('time') or metric_lower.startswith('speed') or
                metric_lower == 'distance' or metric_lower == 'efficiency' or
                metric_lower == 'time' or metric_lower == 'speed'):
                experimental_metrics.append(metric)
            else:
                core_metrics.append(metric)
        
        if core_metrics:
            print(f"  Core metrics ({len(core_metrics)}):")
            for i, metric in enumerate(core_metrics):
                print(f"    {i+1:2d}. {metric}")
        
        if experimental_metrics:
            print(f"  Experimental metrics ({len(experimental_metrics)}) - use with caution:")
            for i, metric in enumerate(experimental_metrics):
                print(f"    {i+1:2d}. {metric}")
    
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

def normalize_data(dfs: List[pd.DataFrame], metrics: List[str]) -> List[pd.DataFrame]:
    """Normalize all data across tables for fair comparison."""
    # Combine all data to get global min/max for each metric
    all_data = pd.concat(dfs, ignore_index=True)
    
    normalized_dfs = []
    
    for table_idx, df in enumerate(dfs):
        normalized_df = df.copy()
        
        for metric in metrics:
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
        
        normalized_dfs.append(normalized_df)
    
    return normalized_dfs

def get_colors(num_tables: int) -> List[str]:
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
                         metrics: List[str], summary_output: Optional[str] = None) -> pd.DataFrame:
    """Create summary table sorted by average performance."""
    
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
        else:
            summary_df['average_score'] = 0
        
        # Sort by average score (descending - best first)
        summary_df = summary_df.sort_values('average_score', ascending=False)
        
        # Prepare output columns in requested order:
        # layout, letters, positions, average_score, then original metrics
        output_columns = ['layout']
        
        # Add layout string columns if available
        if has_layout_data:
            output_columns.extend(['letters', 'positions'])
        
        # Add average score column
        output_columns.append('average_score')
        
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
    
    # If multiple tables, sort globally by average score
    if len(dfs) > 1:
        combined_summary = combined_summary.sort_values('average_score', ascending=False)
    
    # Round average scores to remove floating point precision issues
    if 'average_score' in combined_summary.columns:
        combined_summary['average_score'] = combined_summary['average_score'].round(4)
    
    # Add index column (1-based)
    combined_summary.insert(0, 'index', range(1, len(combined_summary) + 1))
    
    # Save to CSV if requested
    if summary_output:
        combined_summary.to_csv(summary_output, index=False)
        print(f"\nSummary saved to {summary_output}")
    
    print(f"Best performing layouts (by average score):")
    
    # Show top 10 layouts
    display_cols = ['layout']
    if 'table' in combined_summary.columns:
        display_cols.append('table')
    display_cols.append('average_score')
    
    top_layouts = combined_summary[display_cols].head(10)
    for i, (_, row) in enumerate(top_layouts.iterrows(), 1):
        if 'table' in row:
            print(f"  {i:2d}. {row['layout']} ({row['table']}) - Avg Score: {row['average_score']:.3f}")
        else:
            print(f"  {i:2d}. {row['layout']} - Avg Score: {row['average_score']:.3f}")
    
    return combined_summary

def create_heatmap_plot(dfs: List[pd.DataFrame], table_names: List[str], 
                       metrics: List[str], output_path: Optional[str] = None) -> None:
    """Create heatmap visualization with layouts on y-axis and metrics on x-axis."""
    
    # Normalize data across all tables
    normalized_dfs = normalize_data(dfs, metrics)
    
    # Prepare data with sorting within each table by average performance
    all_data = []
    layout_names = []
    
    for i, (df, table_name) in enumerate(zip(normalized_dfs, table_names)):
        # Collect data for this table
        table_data = []
        table_layout_names = []
        
        for _, row in df.iterrows():
            # Get metric values for this layout
            metric_values = []
            valid_count = 0
            
            for metric in metrics:
                if metric in row and pd.notna(row[metric]):
                    metric_values.append(row[metric])
                    valid_count += 1
                else:
                    metric_values.append(0.0)  # Default for missing data
            
            # Skip layouts with too much missing data
            if valid_count < len(metrics) * 0.5:
                continue
            
            table_data.append(metric_values)
            layout_name = row.get('layout', f'Layout_{len(table_layout_names)+1}')
            table_layout_names.append(layout_name)
        
        if not table_data:
            continue
        
        # Convert to numpy array for sorting
        table_matrix = np.array(table_data)
        
        # Sort this table's layouts by average performance (descending)
        table_averages = np.mean(table_matrix, axis=1)
        sort_indices = np.argsort(table_averages)[::-1]  # Descending order
        
        # Apply sorting
        sorted_table_data = table_matrix[sort_indices]
        sorted_table_names = [table_layout_names[idx] for idx in sort_indices]
        
        # Add to combined data
        all_data.extend(sorted_table_data.tolist())
        layout_names.extend(sorted_table_names)
    
    if not all_data:
        print("No valid data found for heatmap")
        return
    
    # Convert to numpy array
    data_matrix = np.array(all_data)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(max(12, len(metrics) * 0.8), max(8, len(layout_names) * 0.3)))
    
    # Create heatmap
    im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(layout_names)))
    
    # Format metric labels
    metric_display_names = []
    for metric in metrics:
        # Clean up scorer names for display
        display_name = metric.replace('_', ' ').title()
        metric_display_names.append(display_name)
    
    ax.set_xticklabels(metric_display_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(layout_names, fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Score (0 = worst, 1 = best)', rotation=270, labelpad=20)
    
    # Add value annotations for smaller matrices
    if len(layout_names) <= 20 and len(metrics) <= 15:
        for i in range(len(layout_names)):
            for j in range(len(metrics)):
                value = data_matrix[i, j]
                # Use white text for dark cells, black for light cells
                text_color = 'white' if value < 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                       color=text_color, fontsize=7, weight='bold')
    
    # Title with sorting info
    sort_info = " (sorted by avg. performance)"
    if len(dfs) > 1:
        sort_info = " (sorted within each table)"
    
    title = f'Keyboard Layout Comparison Heatmap{sort_info}\n{len(layout_names)} layouts across {len(metrics)} metrics'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Labels
    ax.set_xlabel('Scoring Methods', fontsize=12)
    ax.set_ylabel('Keyboard Layouts', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        # Modify output path for heatmap
        if output_path.endswith('.png'):
            heatmap_path = output_path.replace('.png', '_heatmap.png')
        else:
            heatmap_path = output_path + '_heatmap.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Heatmap saved to {heatmap_path}")
    else:
        plt.show()

def create_parallel_plot(dfs: List[pd.DataFrame], table_names: List[str], 
                        metrics: List[str], output_path: Optional[str] = None,
                        summary_df: Optional[pd.DataFrame] = None) -> None:
    """Create parallel coordinates plot with performance-based coloring."""
    # Normalize data across all tables
    normalized_dfs = normalize_data(dfs, metrics)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(max(12, len(metrics) * 1.2), 10))
    
    # Determine coloring scheme
    use_performance_colors = summary_df is not None and len(summary_df) > 0
    
    if use_performance_colors:
        # Create performance-based color mapping
        layout_to_position = {}
        for idx, (_, row) in enumerate(summary_df.iterrows()):
            layout_name = row['layout']
            # Use the index (0-based) as the position for coloring (0 = best)
            layout_to_position[layout_name] = idx
        
        total_layouts = len(layout_to_position)
        
        # Create red color gradient: dark red (best) to light red (worst)
        red_colormap = cm.Reds
        
        # Define color range (avoid pure white, use darker range of reds)
        min_color_val = 0.3  # Light red
        max_color_val = 1.0  # Dark red
        
        print(f"Using performance-based coloring for {total_layouts} layouts")
    else:
        # Use original table-based coloring
        colors = get_colors(len(dfs))
    
    # Plot parameters
    x_positions = range(len(metrics))
    
    # Plot each table's data
    for i, (df, table_name) in enumerate(zip(normalized_dfs, table_names)):
        valid_layout_count = 0
        
        if not use_performance_colors:
            color = colors[i] if i < len(colors) else colors[-1]
        
        for _, row in df.iterrows():
            y_values = [row.get(metric, 0) for metric in metrics]
            
            # Skip rows with too much missing data
            valid_values = [val for val in y_values if pd.notna(val)]
            if len(valid_values) < len(metrics) * 0.5:  # Need at least 50% valid data
                continue
            
            # Replace NaN values with 0
            y_values = [val if pd.notna(val) else 0 for val in y_values]
            
            # Determine line color
            if use_performance_colors:
                layout_name = row.get('layout', '')
                if layout_name in layout_to_position:
                    # Calculate color based on performance position
                    position = layout_to_position[layout_name]
                    # Invert so best performance (0) gets darkest color
                    color_intensity = max_color_val - (position / (total_layouts - 1)) * (max_color_val - min_color_val)
                    color = red_colormap(color_intensity)
                else:
                    # Fallback color for layouts not in summary
                    color = 'gray'
            # else: color already set above for table-based coloring
            
            ax.plot(x_positions, y_values, color=color, alpha=0.7, linewidth=1.5)
            valid_layout_count += 1
        
        # Add legend entry (only for table-based coloring or first table)
        if not use_performance_colors:
            ax.plot([], [], color=color, linewidth=3, label=f"{table_name} ({valid_layout_count} layouts)")
        elif i == 0:  # Only add one legend entry for performance-based coloring
            # Create legend showing color gradient
            ax.plot([], [], color=red_colormap(max_color_val), linewidth=3, label=f"Best performing layouts")
            ax.plot([], [], color=red_colormap(min_color_val), linewidth=3, label=f"Worst performing layouts")

    # Customize the plot
    ax.set_xlim(-0.5, len(metrics) - 0.5)
    ax.set_ylim(-0.05, 1.05)
    
    # Set x-axis labels
    metric_display_names = []
    for metric in metrics:
        # Clean up scorer names for display
        display_name = metric.replace('_', ' ').title()
        # Split long names
        if len(display_name) > 12:
            words = display_name.split()
            if len(words) > 1:
                mid = len(words) // 2
                display_name = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
        metric_display_names.append(display_name)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_display_names, rotation=45, ha='right', fontsize=10)
    
    # Add vertical grid lines for each metric
    for x in x_positions:
        ax.axvline(x, color='gray', alpha=0.3, linewidth=0.5)
    
    # Set y-axis
    ax.set_ylabel('Normalized Score (0 = worst, 1 = best)', fontsize=12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, alpha=0.3)
    
    # Title and legend
    if use_performance_colors:
        title_suffix = f'\nPerformance-ordered visualization: dark red = best, light red = worst'
    else:
        title_suffix = f'\nParallel coordinates across {len(metrics)} scoring methods'
    
    ax.set_title(f'Keyboard Layout Comparison{title_suffix}', 
                fontsize=16, fontweight='bold', pad=20)

    # Show legend
    if len(dfs) > 1 or len(dfs[0]) <= 10 or use_performance_colors:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        # Modify output path for parallel plot
        if output_path.endswith('.png'):
            parallel_path = output_path.replace('.png', '_parallel.png')
        else:
            parallel_path = output_path + '_parallel.png'
        plt.savefig(parallel_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
        print(f"Parallel plot saved to {parallel_path}")
    else:
        plt.show()

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
        description='Create parallel coordinates plots and heatmaps comparing keyboard layouts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preferred format with layout_qwerty column
  python compare_layouts.py --tables layouts.csv
  
  # MOO format with items+positions (auto-converted)
  python compare_layouts.py --tables moo_results.csv --metrics engram dvorak7
  
  # Core metrics only (recommended)
  python compare_layouts.py --tables layouts.csv --metrics engram dvorak7 comfort_combo comfort comfort_key
  
  # Include experimental distance/time metrics (caution: limitations)
  python compare_layouts.py --tables layouts.csv --metrics engram comfort efficiency --experimental-metrics
  
  # Create summary table with performance sorting
  python compare_layouts.py --tables layouts.csv --metrics engram comfort dvorak7 --summary layout_summary.csv
  
  # Multiple tables with filtered metrics and summary
  python compare_layouts.py --tables scores1.csv scores2.csv --metrics comfort engram --summary combined_summary.csv

Input format support (auto-detected):
  Preferred: layout,layout_qwerty,scorer1,scorer2,...
  Standard:  layout,letters,positions,scorer1,scorer2,...  
  MOO:       config_id,items,positions,objective1,objective2,...
  
  - layout_qwerty: what's at each QWERTY position (preferred)
  - letters: what's at each QWERTY position (standard)
  - items: letters in assignment order + positions where they go (MOO)

Summary output:
  CSV with columns: index, layout, [layout_qwerty, positions], average_score, [metric_values]
  - index: 1-based ranking by performance
  - layout_qwerty: what letter is at each QWERTY position
  - positions: QWERTY reference positions
  Layouts ordered by average performance across selected metrics (higher = better)

Core vs Experimental Metrics:
  Core metrics (recommended): engram, comfort, comfort_key, dvorak7
  Experimental metrics (use with caution): efficiency*, speed*
  
  Experimental metrics have significant limitations:
  - Distance/efficiency metrics oversimplify biomechanics (ignore lateral stretching, finger strength, etc.)
  - Time/speed metrics contain QWERTY practice bias from empirical data
        """
    )
    
    parser.add_argument('--tables', nargs='+', required=True,
                       help='One or more CSV files: layout scoring (score_layouts.py --csv) or optimization results (optimize_moo.py)')
    parser.add_argument('--metrics', nargs='*',
                       help='Specific metrics to include (in order). If not specified, all available metrics are used alphabetically.')
    parser.add_argument('--output', '-o', 
                       help='Output file path (if not specified, plots are shown)')
    parser.add_argument('--summary', 
                       help='Create summary table sorted by average performance and save to CSV file (e.g., --summary summary.csv)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed information')
    
    args = parser.parse_args()
    
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
    
    # Filter and order metrics based on user specification
    metrics = filter_and_order_metrics(dfs, args.metrics, args.verbose)
    
    if not metrics:
        print("Error: No valid metrics found")
        sys.exit(1)
    
    # Print summary
    if args.verbose:
        print_summary_stats(dfs, table_names, metrics)

    # Create summary table only if explicitly requested
    summary_df = None
    if args.summary:  # <-- ONLY create summary when --summary is specified
        if args.verbose:
            print(f"\nCreating performance summary...")
        normalized_dfs = normalize_data(dfs, metrics)
        summary_df = create_sorted_summary(normalized_dfs, table_names, metrics, args.summary)
    
    # Create plots (unless only summary requested)
    if args.output is not None or not args.summary:
        if args.verbose:
            print(f"\nCreating visualization plots...")
            print(f"Tables: {len(dfs)}")
            print(f"Total layouts: {sum(len(df) for df in dfs)}")
            print(f"Metrics to plot: {len(metrics)} - {', '.join(metrics)}")
            if summary_df is not None and len(summary_df) > 0:
                print(f"Using performance-based coloring for parallel plot")
        
        # Generate parallel coordinates plot with optional performance-based coloring
        create_parallel_plot(dfs, table_names, metrics, args.output, summary_df)

        # Generate heatmap plot
        create_heatmap_plot(dfs, table_names, metrics, args.output)

if __name__ == "__main__":
    main()
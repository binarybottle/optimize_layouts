#!/usr/bin/env python3
"""
Analyze Layout Optimization Results and Create Visualizations

This script analyzes keyboard layout optimization results and creates scatter plots
of scores with support for both edian-MAD analysis and full dataset analysis. 
Supports both MOO (Multi-Objective Optimization) and 
SOO (Single-Objective Optimization) result formats.

Analysis Modes:
1. Full Analysis: Loads all layouts, creates scatter plots, correlations, summaries
2. Median-MAD Mode: Analysis of file-level statistics with error bars

Key Features:
- Supports multiple file patterns (moo_results_config_*.csv, soo_results_config_*.csv)
- Handles both old and new CSV formats automatically
- Creates scatter plots sorted by different score types
- Generates median-MAD plots with error bars for large datasets
- Validates score combination consistency

Plot Types Generated:
Standard Mode:
- scores_by_total.png (layouts sorted by total score)
- scores_by_item.png (layouts sorted by item score)  
- scores_by_pair.png (layouts sorted by item-pair score)
- 1key_vs_2key.png (correlation plot)

Median-MAD Mode:
- median_by_total.png & median_by_total_sorted.png
- median_by_item.png & median_by_item_sorted.png
- median_by_item_pair.png & median_by_item_pair_sorted.png

Usage:
    python3 analyze_results.py [options]

Examples:
    python3 analyze_results.py --median-mad
    python3 analyze_results.py --results-dir output/layouts
    python3 analyze_results.py --file-pattern "moo_results_config_*.csv"
    python3 analyze_results.py --max-files 1000 --debug
"""
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import csv
import argparse
import numpy as np
from matplotlib.lines import Line2D

from scoring import apply_default_combination_vectorized, DEFAULT_COMBINATION_STRATEGY

# Optional import for visualization
try:
    from optimize_layout import visualize_keyboard_layout
    visualization_available = True
except ImportError:
    print("Warning: Could not import visualization functions")
    visualization_available = False

#-----------------------------------------------------------------------------
# Consolidated file processing utilities
#-----------------------------------------------------------------------------
def process_files_batch(results_dir, file_pattern="moo_results_config_*.csv", max_files=None, progress_step=1000):
    """Generic file finder and iterator with progress tracking."""
    # Use the specified pattern
    pattern_path = f"{results_dir}/{file_pattern}"
    files = glob.glob(pattern_path)
    
    if max_files:
        files = files[:max_files]
    
    if not files:
        print(f"No CSV files found matching pattern: {pattern_path}")
        return []
    
    print(f"Found {len(files)} files matching pattern: {pattern_path}")
    return files

def calculate_mad(values):
    """Calculate Median Absolute Deviation."""
    median_val = np.median(values)
    return np.median(np.abs(values - median_val))

#-----------------------------------------------------------------------------
# Consolidated CSV parsing
#-----------------------------------------------------------------------------
def parse_result_csv(filepath, scores_only=False, debug=False):
    """
    Parse a layout results CSV file (updated for new format).
    
    Args:
        filepath: Path to CSV file
        scores_only: If True, return only scores (memory efficient)
        debug: Enable debug output
        
    Returns:
        List of result dictionaries or scores dictionary
    """
    try:
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return None
            
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            return None
            
        # Skip header section
        header_end = 0
        config_info = {}
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                header_end = i
                break
            parts = line.split(',')
            if len(parts) >= 2:
                key, value = parts[0].strip('"'), parts[1].strip('"')
                config_info[key] = value
        
        # Find data header - look for new format headers
        data_section = lines[header_end+1:]
        header_row = None
        
        for i, line in enumerate(data_section):
            # Look for new format headers
            if ('Complete Layout Score' in line or 'Complete Item' in line or 
                'Total score' in line or 'score' in line.lower()):
                header_row = i
                break
        
        if header_row is None:
            if debug:
                print(f"No header row found in {filepath}")
            return None
        
        # Parse header to get column indices
        header_line = data_section[header_row].strip()
        try:
            reader = csv.reader([header_line])
            headers = [h.strip('"') for h in next(reader)]
        except:
            if debug:
                print(f"Failed to parse header in {filepath}")
            return None
        
        # Find column indices for new format
        column_indices = _get_column_indices(headers, debug, filepath)
        if not column_indices:
            return None
        
        # Process data rows
        if scores_only:
            return _extract_scores_only_new(data_section, header_row, column_indices)
        else:
            return _extract_full_results_new(data_section, header_row, column_indices, config_info, filepath, debug)
            
    except Exception as e:
        if debug:
            print(f"Error parsing {filepath}: {e}")
        return None

def _get_column_indices(headers, debug=False, filepath=""):
    """Get column indices for both old and new formats."""
    indices = {}
    
    # Map header names to our internal names
    header_mappings = {
        # New format
        'Rank': 'rank',
        'Items': 'items', 
        'Positions': 'positions',
        'Complete Layout Score': 'total_score',
        'Complete Item': 'item_score',
        'Complete Pair': 'item_pair_score',
        
        # Old format fallbacks
        'Total score': 'total_score',
        'Item score': 'item_score', 
        'Item-pair score': 'item_pair_score',
        
        # Additional possible variations
        'Opt Item Score': 'opt_item_score',
        'Opt Item-Pair Score': 'opt_item_pair_score',
        'Opt Combined': 'opt_combined'
    }
    
    # Find indices
    for i, header in enumerate(headers):
        header_clean = header.strip()
        if header_clean in header_mappings:
            indices[header_mappings[header_clean]] = i
    
    # Verify we have the essential columns
    required = ['total_score', 'item_score', 'item_pair_score']
    missing = [col for col in required if col not in indices]
    
    if missing:
        if debug:
            print(f"Missing required columns in {filepath}: {missing}")
            print(f"Available headers: {headers}")
        return None
    
    return indices

def _extract_scores_only_new(data_section, header_row, column_indices):
    """Extract only scores using dynamic column indices."""
    total_scores, item_scores, item_pair_scores = [], [], []
    
    total_idx = column_indices['total_score']
    item_idx = column_indices['item_score'] 
    pair_idx = column_indices['item_pair_score']
    
    for row_idx in range(header_row + 1, len(data_section)):
        data_row = data_section[row_idx].strip()
        if not data_row:
            continue
            
        try:
            reader = csv.reader([data_row])
            row_data = next(reader)
            
            if len(row_data) > max(total_idx, item_idx, pair_idx):
                total_scores.append(float(row_data[total_idx].strip('"')))
                item_scores.append(float(row_data[item_idx].strip('"')))
                item_pair_scores.append(float(row_data[pair_idx].strip('"')))
            
        except (ValueError, IndexError):
            continue
    
    return {
        'total_scores': total_scores,
        'item_scores': item_scores,
        'item_pair_scores': item_pair_scores
    }

def _extract_full_results_new(data_section, header_row, column_indices, config_info, filepath, debug):
    """Extract full result data using dynamic column indices."""
    results = []
    
    # Get required indices
    rank_idx = column_indices.get('rank', 0)  # Default to 0 if not found
    items_idx = column_indices.get('items', 1)  # Default to 1 if not found
    positions_idx = column_indices.get('positions', 2)  # Default to 2 if not found
    total_idx = column_indices['total_score']
    item_idx = column_indices['item_score']
    pair_idx = column_indices['item_pair_score']
    
    for row_idx in range(header_row + 1, len(data_section)):
        data_row = data_section[row_idx].strip()
        if not data_row:
            continue
            
        try:
            reader = csv.reader([data_row])
            row_data = next(reader)
            
            # Check we have enough columns
            required_max_idx = max(rank_idx, items_idx, positions_idx, total_idx, item_idx, pair_idx)
            if len(row_data) <= required_max_idx:
                continue
            
            # Extract data
            rank = int(row_data[rank_idx].strip('"')) if rank_idx < len(row_data) else 0
            items = row_data[items_idx].strip('"') if items_idx < len(row_data) else ""
            positions = row_data[positions_idx].strip('"') if positions_idx < len(row_data) else ""
            total_score = float(row_data[total_idx].strip('"'))
            item_score = float(row_data[item_idx].strip('"'))
            item_pair_score = float(row_data[pair_idx].strip('"'))
            
            # Clean special characters in positions
            for special, replacement in [('[semicolon]', ';'), ('[comma]', ','), ('[period]', '.'), ('[slash]', '/')]:
                positions = positions.replace(special, replacement)
            
            results.append({
                'config_id': os.path.basename(filepath).replace('moo_results_config_', '').replace('soo_results_config_', '').replace('layout_results_', '').replace('.csv', ''),
                'items': items,
                'positions': positions,
                'opt_items': "",  # Not in new format
                'opt_positions': "",  # Not in new format
                'total_score': total_score,
                'item_score': item_score, 
                'item_pair_score': item_pair_score,
                'items_to_assign': config_info.get('Items to assign', ''),
                'positions_to_assign': config_info.get('Available positions', ''),
                'items_assigned': config_info.get('Assigned items', ''),
                'positions_assigned': config_info.get('Assigned positions', ''),
                'rank': rank
            })
            
        except (ValueError, IndexError) as e:
            if debug:
                print(f"Error parsing row in {filepath}: {e}")
            continue
    
    return results

#-----------------------------------------------------------------------------
# Consolidated data loading
#-----------------------------------------------------------------------------
def load_results(results_dir, file_pattern="moo_results_config_*.csv", max_files=None, debug=False):
    """Load layout result files and return a dataframe."""
    files = process_files_batch(results_dir, file_pattern, max_files)
    if not files:
        return pd.DataFrame()
    
    all_results = []
    successful_files = 0
    
    for i, filepath in enumerate(files):
        if i % 10 == 0:
            print(f"Processing file {i+1}/{len(files)}: {os.path.basename(filepath)}")
        
        results = parse_result_csv(filepath, scores_only=False, debug=debug)
        if results:
            all_results.extend(results)
            successful_files += 1
    
    print(f"Successfully parsed {successful_files}/{len(files)} files, yielding {len(all_results)} results")
    
    if not all_results:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    
    # Verify score combination consistency
    if not df.empty:
        df['calculated_total'] = apply_default_combination_vectorized(
            df['item_score'].values, df['item_pair_score'].values
        )
        
        tol = 1e-8
        mismatch_count = ((df['total_score'] - df['calculated_total']).abs() > tol).sum()
        if mismatch_count > 0:
            print(f"Warning: {mismatch_count} rows have total scores that don't match current combination strategy ({DEFAULT_COMBINATION_STRATEGY})")
        else:
            print(f"✓ All total scores match current combination strategy ({DEFAULT_COMBINATION_STRATEGY})")
    
    return df

#-----------------------------------------------------------------------------
# Generic plotting functions
#-----------------------------------------------------------------------------
def plot_scores_generic(df, sort_by='total_score', score_types=None, save_path=None, title_suffix=''):
    """
    Generic scatter plot function that replaces multiple similar functions.
    
    Args:
        df: DataFrame with score data
        sort_by: Column to sort by ('total_score', 'item_score', 'item_pair_score')
        score_types: List of score types to plot ['total', 'item', 'pair']
        save_path: Output file path
        title_suffix: Additional text for title
    """
    if df.empty:
        print("No results to plot!")
        return
    
    if score_types is None:
        score_types = ['total', 'item', 'pair']
    
    # Column mapping
    col_map = {
        'total': 'total_score',
        'item': 'item_score', 
        'pair': 'item_pair_score'
    }
    
    label_map = {
        'total': 'Total Score',
        'item': 'Item Score (1-key)',
        'pair': 'Item-pair Score (2-key)'
    }
    
    # Sort data
    df_sorted = df.sort_values(sort_by)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange']
    for i, score_type in enumerate(score_types):
        col = col_map[score_type]
        if col in df.columns:
            plt.scatter(range(len(df_sorted)), df_sorted[col], 
                       marker='.', s=30, alpha=0.6, label=label_map[score_type], 
                       edgecolors='none', color=colors[i % len(colors)])
    
    # Configure plot
    sort_label = sort_by.replace('_', '-')
    plt.xlabel(f'Layout Index (sorted by {sort_label})')
    plt.ylabel('Score')
    plt.title(f'Layout Scores Sorted by {sort_label.title()}{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add score range info
    info_lines = []
    for score_type in score_types:
        col = col_map[score_type]
        if col in df.columns:
            info_lines.append(f"{label_map[score_type]}: {df[col].min():.6f} to {df[col].max():.6f}")
    
    plt.figtext(0.02, 0.02, '\n'.join(info_lines), fontsize=9)
    
    # Save plot
    if save_path is None:
        save_path = f'scores_by_{sort_by.replace("_score", "")}.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Generated scores plot (saved as {save_path})")
    plt.close()

def plot_score_comparison(df, save_path=None):
    """Create 1-key vs 2-key comparison plot."""
    if df.empty:
        return
    
    plt.figure(figsize=(12, 10))
    
    plt.scatter(df['item_score'], df['item_pair_score'], 
                marker='.', s=30, alpha=0.6, edgecolors='none', label='Layout Scores')
    
    plt.xlabel('Item Score (1-key)')
    plt.ylabel('Item-pair Score (2-key)')
    plt.title('1-Key vs 2-Key Scores')
    plt.grid(True, alpha=0.3)
    
    # Calculate and display correlation
    corr = df['item_score'].corr(df['item_pair_score'])
    plt.figtext(0.02, 0.02, f'Correlation: {corr:.4f}', fontsize=9)
    
    if save_path is None:
        save_path = '1key_vs_2key.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Generated comparison plot (saved as {save_path})")
    plt.close()

#-----------------------------------------------------------------------------
# Generic median-MAD plotting
#-----------------------------------------------------------------------------
def plot_median_mad_generic(results_dir, score_type='total', sort_by_total=False, max_files=None, file_pattern="moo_results_config_*.csv"):
    """
    Generic median-MAD plot function that replaces multiple similar functions.
    
    Args:
        results_dir: Directory containing CSV files
        score_type: 'total', 'item', or 'item_pair'
        sort_by_total: If True, sort by total score median instead of file order
        max_files: Maximum number of files to process
        file_pattern: File pattern to match (e.g., "moo_results_config_*.csv", "soo_results_config_*.csv")
    """
    files = process_files_batch(results_dir, file_pattern, max_files)
    if not files:
        return
    
    print(f"Processing {len(files)} files for {score_type} scores{'(sorted by total)' if sort_by_total else ''}")
    
    file_stats = []
    score_key = f'{score_type}_scores'
    
    # Process files
    for i, filepath in enumerate(files):
        if i % 5000 == 0:
            print(f"Processing file {i+1}/{len(files)}: {os.path.basename(filepath)}")
        
        scores_data = parse_result_csv(filepath, scores_only=True)
        
        if scores_data and score_key in scores_data and scores_data[score_key]:
            scores_array = np.array(scores_data[score_key])
            median_score = np.median(scores_array)
            mad_score = calculate_mad(scores_array)
            
            file_key = os.path.basename(filepath).replace('moo_results_config_', '').replace('soo_results_config_', '').replace('layout_results_', '').replace('.csv', '')
            
            stat_entry = {
                'file_key': file_key,
                'median': median_score,
                'mad': mad_score,
                'count': len(scores_data[score_key]),
                'min_score': float(np.min(scores_array)),
                'max_score': float(np.max(scores_array))
            }
            
            # Add total median for sorting if needed
            if sort_by_total and score_type != 'total' and 'total_scores' in scores_data:
                stat_entry['total_median'] = np.median(np.array(scores_data['total_scores']))
            
            file_stats.append(stat_entry)
    
    if not file_stats:
        print(f"No file statistics calculated for {score_type} scores!")
        return
    
    print(f"Successfully processed {len(file_stats)} files")
    
    # Sort files
    if sort_by_total and score_type != 'total':
        file_stats.sort(key=lambda x: x.get('total_median', 0))
        sort_desc = 'sorted by total score'
    elif sort_by_total:
        file_stats.sort(key=lambda x: x['median'])
        sort_desc = 'sorted by median'
    else:
        file_stats.sort(key=lambda x: x['file_key'])
        sort_desc = 'by file order'
    
    # Extract data for plotting
    medians = [stat['median'] for stat in file_stats]
    mads = [stat['mad'] for stat in file_stats]
    counts = [stat['count'] for stat in file_stats]
    file_indices = range(len(file_stats))
    
    # Create plot
    plt.figure(figsize=(24, 8))
    
    plt.errorbar(file_indices, medians, yerr=mads, fmt='none', 
                 capsize=1, elinewidth=1, color='black', alpha=0.6, zorder=1)
    plt.scatter(file_indices, medians, s=4, color='red', alpha=0.8, 
               edgecolors='none', zorder=2)
    
    # Labels and formatting
    score_label = score_type.replace('_', '-').title()
    plt.xlabel(f'File Index ({sort_desc})')
    plt.ylabel(f'Median {score_label} Score')
    plt.title(f'Median {score_label} Scores with MAD Error Bars\n({len(file_stats)} files, avg {np.mean(counts):.0f} layouts each)')
    plt.grid(True, alpha=0.3)
    
    # Set reasonable x-axis ticks
    if len(file_stats) > 50:
        tick_step = max(1, len(file_stats) // 50)
        plt.xticks(range(0, len(file_stats), tick_step))
    
    # Add statistics
    overall_median = np.median(medians)
    overall_mad = calculate_mad(np.array(medians))
    
    stats_text = [
        f"Files processed: {len(file_stats)}",
        f"Total layouts: {sum(counts):,}",
        f"Layouts per file: ~{np.mean(counts):.0f}",
        f"Median of medians: {overall_median:.6f}",
        f"MAD of medians: {overall_mad:.6f}",
        f"Range: {min(medians):.6f} to {max(medians):.6f}"
    ]
    
    plt.figtext(0.05, 0.1, '\n'.join(stats_text), fontsize=10)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label=f'Median {score_label} Score'),
        Line2D([0], [0], color='black', linewidth=2, label='MAD Error Bars')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    # Save plot
    sort_suffix = '_sorted' if sort_by_total else ''
    save_path = f'median_by_{score_type}{sort_suffix}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Generated {score_type} median-MAD plot (saved as {save_path})")
    plt.close()
    
    return file_stats

#-----------------------------------------------------------------------------
# Analysis and summary functions
#-----------------------------------------------------------------------------
def analyze_results(df):
    """Print comprehensive analysis of results."""
    if df.empty:
        print("No results to analyze!")
        return
    
    print(f"\nAnalyzed {len(df)} layout results")
    
    # Score statistics
    print("\nScore Statistics:")
    for score_type, col in [('Total', 'total_score'), ('Item', 'item_score'), ('Item-pair', 'item_pair_score')]:
        print(f"  {score_type}: {df[col].min():.6f} to {df[col].max():.6f} (mean: {df[col].mean():.6f})")
    
    # Correlations
    corr_item_total = df['item_score'].corr(df['total_score'])
    corr_pair_total = df['item_pair_score'].corr(df['total_score'])
    corr_item_pair = df['item_score'].corr(df['item_pair_score'])
    
    print("\nCorrelations:")
    print(f"  Item to Total: {corr_item_total:.4f}")
    print(f"  Item-pair to Total: {corr_pair_total:.4f}")
    print(f"  Item to Item-pair: {corr_item_pair:.4f}")
    
    # Best layouts
    for score_type, col in [('Total', 'total_score'), ('Item', 'item_score'), ('Item-pair', 'item_pair_score')]:
        best_idx = df[col].idxmax()
        print(f"\nBest Layout by {score_type} Score:")
        print(f"  Config ID: {df.loc[best_idx, 'config_id']}")
        print(f"  Scores - Total: {df.loc[best_idx, 'total_score']:.6f}, Item: {df.loc[best_idx, 'item_score']:.6f}, Pair: {df.loc[best_idx, 'item_pair_score']:.6f}")
        print(f"  Layout: {df.loc[best_idx, 'items']} → {df.loc[best_idx, 'positions']}")

def save_results_summary(df):
    """Save results to Excel/CSV with proper formatting."""
    if df.empty:
        return
    
    # Select and order columns
    columns_to_export = [
        'config_id', 'total_score', 'item_score', 'item_pair_score',
        'items', 'positions', 'items_to_assign', 'positions_to_assign',
        'items_assigned', 'positions_assigned', 'opt_items', 'opt_positions', 'rank'
    ]
    
    valid_columns = [col for col in columns_to_export if col in df.columns]
    df_export = df[valid_columns].sort_values('total_score', ascending=False)
    
    # Save to Excel or CSV
    try:
        df_export.to_excel("layout_scores_summary.xlsx", index=False)
        print("Results saved to layout_scores_summary.xlsx")
    except Exception as e:
        print(f"Excel save failed ({e}), saving to CSV...")
        df_export.to_csv("layout_scores_summary.csv", index=False)
        print("Results saved to layout_scores_summary.csv")

#-----------------------------------------------------------------------------
# Main function
#-----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Analyze layout optimization results (streamlined version)')
    parser.add_argument('--results-dir', type=str, default='output/layouts', help='Directory containing result files')
    parser.add_argument('--file-pattern', type=str, default='moo_results_config_*.csv', help='File pattern to match (e.g., "moo_results_config_*.csv", "soo_results_config_*.csv")')
    parser.add_argument('--max-files', type=int, default=None, help='Maximum number of files to process')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--median-mad', action='store_true', help='Only generate median-MAD plots')
    
    args = parser.parse_args()
    
    print(f"Streamlined analyze_results.py starting...")
    print(f"Results directory: {args.results_dir}")
    print(f"File pattern: {args.file_pattern}")
    print(f"Max files: {args.max_files or 'all'}")
    
    try:
        # Check directory exists
        if not os.path.exists(args.results_dir):
            print(f"ERROR: Directory '{args.results_dir}' does not exist!")
            return
        
        # Memory-efficient median-MAD analysis
        if args.median_mad_only:
            print("Running memory-efficient median-MAD analysis...")
            for score_type in ['total', 'item', 'item_pair']:
                plot_median_mad_generic(args.results_dir, score_type, False, args.max_files, args.file_pattern)
                plot_median_mad_generic(args.results_dir, score_type, True, args.max_files, args.file_pattern)
            return
        
        # Full analysis
        print("Loading results for full analysis...")
        df = load_results(args.results_dir, args.file_pattern, args.max_files, args.debug)
        
        if df.empty:
            print("No valid results found!")
            return
        
        print(f"Loaded {len(df)} layout results")
        
        # Generate all plots using consolidated functions
        plot_scores_generic(df, 'total_score', save_path='scores_by_total.png')
        plot_scores_generic(df, 'item_score', save_path='scores_by_item.png')
        plot_scores_generic(df, 'item_pair_score', save_path='scores_by_pair.png')
        plot_score_comparison(df)
        
        # Analysis and export
        analyze_results(df)
        save_results_summary(df)
        
        print("\nStreamlined analysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
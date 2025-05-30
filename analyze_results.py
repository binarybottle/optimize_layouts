#!/usr/bin/env python3
"""
Analyze layout optimization results and create scatter plots of scores.

python3 analyze_results.py --results-dir output/layouts
python3 analyze_results.py --max-files 1000
python3 analyze_results.py --debug
python3 analyze_results.py --median-mad-only
python3 analyze_results.py --scoring-comparison

"""
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import csv
import argparse
import numpy as np
from scipy.stats import spearmanr
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
def process_files_batch(results_dir, max_files=None, progress_step=1000):
    """Generic file finder and iterator with progress tracking."""
    files = glob.glob(f"{results_dir}/layout_results_*.csv")
    if max_files:
        files = files[:max_files]
    
    if not files:
        print(f"No CSV files found matching pattern: {results_dir}/layout_results_*.csv")
        return []
    
    print(f"Found {len(files)} files to process")
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
    Parse a layout results CSV file.
    
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
        
        # Find data header
        data_section = lines[header_end+1:]
        header_row = None
        
        for i, line in enumerate(data_section):
            if 'Total score' in line or 'score' in line.lower():
                header_row = i
                break
        
        if header_row is None:
            return None
        
        # Process data rows
        if scores_only:
            return _extract_scores_only(data_section, header_row)
        else:
            return _extract_full_results(data_section, header_row, config_info, filepath, debug)
            
    except Exception as e:
        if debug:
            print(f"Error parsing {filepath}: {e}")
        return None

def _extract_scores_only(data_section, header_row):
    """Extract only scores for memory-efficient processing."""
    total_scores, item_scores, item_pair_scores = [], [], []
    
    for row_idx in range(header_row + 1, len(data_section)):
        data_row = data_section[row_idx].strip()
        if not data_row:
            continue
            
        try:
            reader = csv.reader([data_row])
            row_data = next(reader)
            
            # Determine indices based on row length
            if len(row_data) >= 8:
                total_idx, item_idx, pair_idx = 5, 6, 7
            elif len(row_data) >= 6:
                total_idx, item_idx, pair_idx = 3, 4, 5
            else:
                continue
            
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

def _extract_full_results(data_section, header_row, config_info, filepath, debug):
    """Extract full result data."""
    results = []
    
    for row_idx in range(header_row + 1, len(data_section)):
        data_row = data_section[row_idx].strip()
        if not data_row:
            continue
            
        try:
            reader = csv.reader([data_row])
            row_data = next(reader)
            
            # Parse based on format
            if len(row_data) >= 8:
                items, positions = row_data[0].strip('"'), row_data[1].strip('"')
                opt_items, opt_positions = row_data[2].strip('"'), row_data[3].strip('"')
                rank, total_score = int(row_data[4].strip('"')), float(row_data[5].strip('"'))
                item_score, item_pair_score = float(row_data[6].strip('"')), float(row_data[7].strip('"'))
            elif len(row_data) >= 6:
                items, positions = row_data[0].strip('"'), row_data[1].strip('"')
                opt_items = opt_positions = ""
                rank, total_score = int(row_data[2].strip('"')), float(row_data[3].strip('"'))
                item_score, item_pair_score = float(row_data[4].strip('"')), float(row_data[5].strip('"'))
            else:
                continue
            
            # Clean special characters in positions
            for special, replacement in [('[semicolon]', ';'), ('[comma]', ','), ('[period]', '.'), ('[slash]', '/')]:
                positions = positions.replace(special, replacement)
                opt_positions = opt_positions.replace(special, replacement)
            
            results.append({
                'config_id': os.path.basename(filepath).replace('layout_results_', '').replace('.csv', ''),
                'items': items, 'positions': positions,
                'opt_items': opt_items, 'opt_positions': opt_positions,
                'total_score': total_score, 'item_score': item_score, 'item_pair_score': item_pair_score,
                'items_to_assign': config_info.get('Items to assign', ''),
                'positions_to_assign': config_info.get('Available positions', ''),
                'items_assigned': config_info.get('Assigned items', ''),
                'positions_assigned': config_info.get('Assigned positions', ''),
                'rank': rank
            })
            
        except (ValueError, IndexError):
            continue
    
    return results

#-----------------------------------------------------------------------------
# Consolidated data loading
#-----------------------------------------------------------------------------
def load_results(results_dir, max_files=None, debug=False):
    """Load layout result files and return a dataframe."""
    files = process_files_batch(results_dir, max_files)
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
def plot_median_mad_generic(results_dir, score_type='total', sort_by_total=False, max_files=None):
    """
    Generic median-MAD plot function that replaces multiple similar functions.
    
    Args:
        results_dir: Directory containing CSV files
        score_type: 'total', 'item', or 'item_pair'
        sort_by_total: If True, sort by total score median instead of file order
        max_files: Maximum number of files to process
    """
    files = process_files_batch(results_dir, max_files)
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
            
            file_key = os.path.basename(filepath).replace('layout_results_', '').replace('.csv', '')
            
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
# Scoring comparison utilities
#-----------------------------------------------------------------------------
def plot_scoring_comparison(results_dir, max_files=None, 
                          item_range=(0.08, 0.13), pair_range=(0.214, 0.228),
                          weights=(0.3, 0.7), save_path=None):
    """Compare current scoring with weighted normalized scoring."""
    files = process_files_batch(results_dir, max_files)
    if not files:
        return
    
    item_min, item_max = item_range
    pair_min, pair_max = pair_range
    item_weight, pair_weight = weights
    
    print(f"Comparing scoring methods on {len(files)} files...")
    print(f"Item range: [{item_min:.3f}, {item_max:.3f}], weight: {item_weight:.1f}")
    print(f"Pair range: [{pair_min:.3f}, {pair_max:.3f}], weight: {pair_weight:.1f}")
    
    # Collect data
    current_scores, weighted_scores = [], []
    
    for i, filepath in enumerate(files):
        if i % 1000 == 0:
            print(f"Processing file {i+1}/{len(files)}")
        
        scores_data = parse_result_csv(filepath, scores_only=True)
        if not scores_data:
            continue
        
        for j in range(len(scores_data['total_scores'])):
            current_total = scores_data['total_scores'][j]
            item_score = scores_data['item_scores'][j]
            pair_score = scores_data['item_pair_scores'][j]
            
            # Calculate weighted normalized score
            item_norm = (item_score - item_min) / (item_max - item_min)
            pair_norm = (pair_score - pair_min) / (pair_max - pair_min)
            weighted_score = item_weight * item_norm + pair_weight * pair_norm
            
            current_scores.append(current_total)
            weighted_scores.append(weighted_score)
    
    if not current_scores:
        print("No data collected!")
        return
    
    # Convert to arrays and analyze
    current_scores = np.array(current_scores)
    weighted_scores = np.array(weighted_scores)
    
    correlation = np.corrcoef(current_scores, weighted_scores)[0, 1]
    rank_correlation, _ = spearmanr(current_scores, weighted_scores)
    
    print(f"Collected {len(current_scores):,} layout scores")
    print(f"Pearson correlation: {correlation:.4f}")
    print(f"Spearman rank correlation: {rank_correlation:.4f}")
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Scatter plot
    ax1.scatter(current_scores, weighted_scores, alpha=0.5, s=1)
    ax1.set_xlabel('Current Total Score (multiplication)')
    ax1.set_ylabel('Weighted Normalized Score')
    ax1.set_title(f'Score Comparison\nPearson r={correlation:.3f}, Spearman ρ={rank_correlation:.3f}')
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram comparison
    ax2.hist(current_scores, bins=50, alpha=0.7, label='Current (mult)', density=True)
    ax2.hist(weighted_scores, bins=50, alpha=0.7, label='Weighted norm', density=True)
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Score Distributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Top N overlap analysis
    n_top = min(1000, len(current_scores) // 10)
    current_top_idx = np.argsort(current_scores)[-n_top:]
    weighted_top_idx = np.argsort(weighted_scores)[-n_top:]
    overlap = len(set(current_top_idx) & set(weighted_top_idx))
    overlap_pct = overlap / n_top * 100
    
    ax3.scatter(current_scores[current_top_idx], weighted_scores[current_top_idx], 
               alpha=0.7, s=2, color='red', label=f'Top {n_top} by current')
    ax3.scatter(current_scores[weighted_top_idx], weighted_scores[weighted_top_idx], 
               alpha=0.7, s=2, color='blue', label=f'Top {n_top} by weighted')
    ax3.set_xlabel('Current Total Score')
    ax3.set_ylabel('Weighted Normalized Score')
    ax3.set_title(f'Top {n_top} Layouts\nOverlap: {overlap} ({overlap_pct:.1f}%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax4.axis('off')
    stats_text = [
        f"Layouts analyzed: {len(current_scores):,}",
        f"Current range: {current_scores.min():.5f} to {current_scores.max():.5f}",
        f"Weighted range: {weighted_scores.min():.3f} to {weighted_scores.max():.3f}",
        f"Item range: [{item_min:.3f}, {item_max:.3f}]",
        f"Pair range: [{pair_min:.3f}, {pair_max:.3f}]",
        f"Weights: {item_weight:.1f} item + {pair_weight:.1f} pair",
        f"Top {n_top} overlap: {overlap_pct:.1f}%",
        f"Pearson correlation: {correlation:.4f}",
        f"Spearman correlation: {rank_correlation:.4f}"
    ]
    ax4.text(0.1, 0.9, '\n'.join(stats_text), fontsize=11, verticalalignment='top')
    
    if save_path is None:
        save_path = f'scoring_comparison_w{item_weight:.1f}_{pair_weight:.1f}.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Scoring comparison saved as {save_path}")
    plt.close()
    
    return {
        'pearson_correlation': correlation,
        'spearman_correlation': rank_correlation,
        'top_n_overlap_percent': overlap_pct,
        'n_layouts': len(current_scores)
    }

#-----------------------------------------------------------------------------
# Main function
#-----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Analyze layout optimization results (streamlined version)')
    parser.add_argument('--results-dir', type=str, default='output/layouts', help='Directory containing result files')
    parser.add_argument('--max-files', type=int, default=None, help='Maximum number of files to process')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--median-mad-only', action='store_true', help='Only generate median-MAD plots')
    parser.add_argument('--scoring-comparison', action='store_true', help='Generate scoring comparison plots')
    
    args = parser.parse_args()
    
    print(f"Streamlined analyze_results.py starting...")
    print(f"Results directory: {args.results_dir}")
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
                plot_median_mad_generic(args.results_dir, score_type, False, args.max_files)
                plot_median_mad_generic(args.results_dir, score_type, True, args.max_files)
            return
        
        # Scoring comparison analysis
        if args.scoring_comparison:
            print("Running scoring comparison analysis...")
            # Test multiple weight combinations
            for weights in [(0.5, 0.5), (0.3, 0.7), (0.2, 0.8), (0.7, 0.3)]:
                plot_scoring_comparison(args.results_dir, args.max_files, weights=weights)
            return
        
        # Full analysis
        print("Loading results for full analysis...")
        df = load_results(args.results_dir, args.max_files, args.debug)
        
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
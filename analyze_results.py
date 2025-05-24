#!/usr/bin/env python3
"""
Analyze layout optimization results and create scatter plots of scores.

python3 analyze_results.py --results-dir output/layouts
python3 analyze_results.py --max-files 1000
python3 analyze_results.py --debug
python3 analyze_results.py --results-dir output/layouts --median-mad-only

"""
import os
import sys
import pandas as pd
import glob
import matplotlib.pyplot as plt
import csv
import re
import argparse
import numpy as np

try:
    from optimize_layout import visualize_keyboard_layout
    visualization_available = True
except ImportError:
    print("Warning: Could not import visualization functions")
    visualization_available = False

def parse_result_csv(filepath):
    """Parse a layout results CSV file and extract key metrics."""
    try:
        # First check if file exists
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None
            
        # Check file size
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            print(f"File is empty: {filepath}")
            return None
            
        # Now try to read the file
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"No content in file: {filepath}")
            return None
            
        # Extract configuration info
        config_info = {}
        header_end = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:  # Empty line marks end of header
                header_end = i
                break
                
            parts = line.split(',')
            if len(parts) >= 2:
                key, value = parts[0].strip('"'), parts[1].strip('"')
                config_info[key] = value
        
        # Find the data header and parse data rows
        data_section = lines[header_end+1:]
        header_row = None
        
        for i, line in enumerate(data_section):
            if 'Total score' in line:
                header_row = i
                break
        
        if header_row is None:
            print(f"Could not find 'Total score' column in {filepath}")
            return None
            
        results = []
        
        # Process ALL data rows after the header, not just the first one
        for row_idx in range(header_row + 1, len(data_section)):
            data_row = data_section[row_idx].strip()
            if not data_row:  # Skip empty lines
                continue
                
            # Debug print
            if 'debug' in globals() and debug and row_idx < header_row + 3:
                print(f"Processing data row {row_idx}: {data_row[:50]}...")
            
            # Parse CSV row properly
            reader = csv.reader([data_row])
            try:
                row_data = next(reader)
            except:
                print(f"Could not parse row {row_idx} in {filepath}")
                continue
            
            # Debug print
            if 'debug' in globals() and debug and row_idx < header_row + 3:
                print(f"Parsed row data: {row_data[:4]}...")
            
            # Simply look for the score columns by position
            # Based on the example format:
            # "Items","Positions","Optimized Items","Optimized Positions","Rank","Total score","Item score","Item-pair score"
            if len(row_data) >= 8:  # Full format with optimized columns
                items_idx = 0
                positions_idx = 1
                opt_items_idx = 2
                opt_positions_idx = 3
                rank_idx = 4
                total_score_idx = 5
                item_score_idx = 6
                item_pair_score_idx = 7
            elif len(row_data) >= 6:  # Shorter format
                items_idx = 0
                positions_idx = 1
                opt_items_idx = None
                opt_positions_idx = None
                rank_idx = 2
                total_score_idx = 3
                item_score_idx = 4
                item_pair_score_idx = 5
            else:
                if 'debug' in globals() and debug:
                    print(f"Not enough columns in {filepath} row {row_idx}, found {len(row_data)}")
                continue  # Skip this row
                
            # Extract data
            items = row_data[items_idx].strip('"') if items_idx < len(row_data) else ""
            positions = row_data[positions_idx].strip('"') if positions_idx < len(row_data) else ""
            
            # Extract optimized items/positions if available
            opt_items = ""
            opt_positions = ""
            if opt_items_idx is not None and opt_items_idx < len(row_data):
                opt_items = row_data[opt_items_idx].strip('"')
            if opt_positions_idx is not None and opt_positions_idx < len(row_data):
                opt_positions = row_data[opt_positions_idx].strip('"')
            
            try:
                rank = int(row_data[rank_idx].strip('"')) if rank_idx < len(row_data) else 1
            except ValueError:
                if 'debug' in globals() and debug:
                    print(f"Could not parse rank from {row_data[rank_idx]}")
                rank = 1
            
            try:
                total_score = float(row_data[total_score_idx].strip('"'))
            except (ValueError, IndexError) as e:
                if 'debug' in globals() and debug:
                    print(f"Error parsing total score: {e}")
                continue  # Skip this row
            
            try:
                item_score = float(row_data[item_score_idx].strip('"'))
            except (ValueError, IndexError) as e:
                if 'debug' in globals() and debug:
                    print(f"Error parsing item score: {e}")
                item_score = 0.0
            
            try:
                item_pair_score = float(row_data[item_pair_score_idx].strip('"'))
            except (ValueError, IndexError) as e:
                if 'debug' in globals() and debug:
                    print(f"Error parsing item-pair score: {e}")
                item_pair_score = 0.0
                
            # Clean up special characters in positions string
            clean_positions = positions
            for special, replacement in [
                ('[semicolon]', ';'),
                ('[comma]', ','),
                ('[period]', '.'),
                ('[slash]', '/')
            ]:
                clean_positions = clean_positions.replace(special, replacement)
            
            # Also clean optimized positions if available
            clean_opt_positions = opt_positions
            for special, replacement in [
                ('[semicolon]', ';'),
                ('[comma]', ','),
                ('[period]', '.'),
                ('[slash]', '/')
            ]:
                clean_opt_positions = clean_opt_positions.replace(special, replacement)
            
            # Get items to assign and positions to assign from config info
            items_to_assign = config_info.get('Items to assign', '')
            positions_to_assign = config_info.get('Available positions', '')
            items_assigned = config_info.get('Assigned items', '')
            positions_assigned = config_info.get('Assigned positions', '')
                        
            # Create result dictionary
            result = {
                'config_id': os.path.basename(filepath).replace('layout_results_', '').replace('.csv', ''),
                'items': items,
                'positions': clean_positions,
                'opt_items': opt_items,
                'opt_positions': clean_opt_positions,
                'total_score': total_score,
                'item_score': item_score,
                'item_pair_score': item_pair_score,
                'items_to_assign': items_to_assign,
                'positions_to_assign': positions_to_assign,
                'items_assigned': items_assigned,
                'positions_assigned': positions_assigned,
                'rank': rank
            }
            
            results.append(result)
        
        if 'debug' in globals() and debug:
            print(f"Successfully parsed {len(results)} rows from {filepath}")
                
        return results
            
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def load_results(results_dir, max_files=None):
    """Load layout result files and return a dataframe."""
    all_results = []
    
    # Find CSV files
    files = glob.glob(f"{results_dir}/layout_results_*.csv")
    if max_files:
        files = files[:max_files]
    
    if not files:
        print(f"No CSV files found matching pattern: {results_dir}/layout_results_*.csv")
        return pd.DataFrame()  # Return empty dataframe
    
    print(f"Found {len(files)} files to process")
    
    # Process each file
    successful_files = 0
    for i, filepath in enumerate(files):
        if i % 10 == 0:  # Print progress every 10 files
            print(f"Processing file {i+1}/{len(files)}: {os.path.basename(filepath)}")
        results = parse_result_csv(filepath)
        if results:
            all_results.extend(results)
            successful_files += 1
    
    print(f"Successfully parsed {successful_files}/{len(files)} files, yielding {len(all_results)} results")
    
    if not all_results:
        print("No valid results found in any file!")
        return pd.DataFrame()  # Return empty dataframe
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Print column names to help debugging
    print(f"DataFrame columns: {', '.join(df.columns.tolist())}")
    
    if not df.empty:
        # Verify that total_score = item_score * item_pair_score
        df['calculated_total'] = df['item_score'] * df['item_pair_score']
        
        # Check if there's a significant difference between calculated and reported total
        tol = 1e-8  # Tolerance for floating point comparison
        mismatch_count = ((df['total_score'] - df['calculated_total']).abs() > tol).sum()
        if mismatch_count > 0:
            print(f"Warning: {mismatch_count} rows have total scores that don't match the product of item and item-pair scores")
            # Calculate difference statistics for mismatches
            mismatch_df = df[((df['total_score'] - df['calculated_total']).abs() > tol)]
            print(f"  Average difference: {(mismatch_df['total_score'] - mismatch_df['calculated_total']).abs().mean()}")
            print(f"  Max difference: {(mismatch_df['total_score'] - mismatch_df['calculated_total']).abs().max()}")
    
    return df

def plot_scores_scatter(df, save_path=None):
    """Create a scatter plot of all three score types."""
    if df.empty:
        print("No results to plot!")
        return
    
    # Determine which score columns to use
    total_col = 'total_score'
    item_col = 'item_score'
    item_pair_col = 'item_pair_score'
    title_prefix = ''
    if save_path is None:
        save_path = 'raw_scores_scatter.png'
    
    # Sort by the total score
    df_sorted = df.sort_values(total_col)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with color-coded points
    plt.scatter(range(len(df_sorted)), df_sorted[total_col], 
                marker='.', s=30, alpha=0.6, label='Total Score', edgecolors='none')
    plt.scatter(range(len(df_sorted)), df_sorted[item_col], 
                marker='.', s=30, alpha=0.6, label='Item Score', edgecolors='none')
    plt.scatter(range(len(df_sorted)), df_sorted[item_pair_col], 
                marker='.', s=30, alpha=0.6, label='Item-pair Score', edgecolors='none')
    
    # Add labels and title
    plt.xlabel(f'Layout Index (sorted by total score)')
    plt.ylabel(f'{title_prefix}Score')
    plt.title(f'{title_prefix}Layout Optimization Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate the score ranges
    score_min = min(df_sorted[total_col].min(), 
                    df_sorted[item_col].min(), 
                    df_sorted[item_pair_col].min())
    score_max = max(df_sorted[total_col].max(), 
                    df_sorted[item_col].max(), 
                    df_sorted[item_pair_col].max())
    
    # Set y-axis limits with padding
    score_range = score_max - score_min
    plt.ylim(max(0, score_min - score_range * 0.05), score_max + score_range * 0.05)
    
    # Add score range information
    info_text = [
        f"Total Score: {df_sorted[total_col].min():.6f} to {df_sorted[total_col].max():.6f}",
        f"Item Score: {df_sorted[item_col].min():.6f} to {df_sorted[item_col].max():.6f}",
        f"Item-pair Score: {df_sorted[item_pair_col].min():.6f} to {df_sorted[item_pair_col].max():.6f}"
    ]
    plt.figtext(0.02, 0.02, '\n'.join(info_text), fontsize=9)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Generated {title_prefix.lower()}scores scatter plot (saved as {save_path})")

def plot_scores_by_item_score(df, save_path=None):
    """Create a scatter plot of scores, sorted by 1-key scores."""
    if df.empty:
        print("No results to plot!")
        return
    
    # Determine which score columns to use
    total_col = 'total_score'
    item_col = 'item_score'
    item_pair_col = 'item_pair_score'
    title_prefix = ''
    if save_path is None:
        save_path = 'raw_scores_by_1key.png'
    
    # Sort by the item score
    df_sorted = df.sort_values(item_col)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with color-coded points
    plt.scatter(range(len(df_sorted)), df_sorted[total_col], 
                marker='.', s=30, alpha=0.6, label='Total Score', edgecolors='none')
    plt.scatter(range(len(df_sorted)), df_sorted[item_col], 
                marker='.', s=30, alpha=0.6, label='Item Score (1-key)', edgecolors='none')
    plt.scatter(range(len(df_sorted)), df_sorted[item_pair_col], 
                marker='.', s=30, alpha=0.6, label='Item-pair Score (2-key)', edgecolors='none')
    
    # Add labels and title
    plt.xlabel(f'Layout Index (sorted by 1-key score)')
    plt.ylabel(f'{title_prefix}Score')
    plt.title(f'{title_prefix}Layout Scores Sorted by 1-Key Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate the score ranges
    score_min = min(df_sorted[total_col].min(), 
                    df_sorted[item_col].min(), 
                    df_sorted[item_pair_col].min())
    score_max = max(df_sorted[total_col].max(), 
                    df_sorted[item_col].max(), 
                    df_sorted[item_pair_col].max())
    
    # Set y-axis limits with padding
    score_range = score_max - score_min
    plt.ylim(max(0, score_min - score_range * 0.05), score_max + score_range * 0.05)
    
    # Add score range information
    info_text = [
        f"1-key Score Range: {df_sorted[item_col].min():.6f} to {df_sorted[item_col].max():.6f}",
        f"2-key Score Range: {df_sorted[item_pair_col].min():.6f} to {df_sorted[item_pair_col].max():.6f}",
        f"Total Score Range: {df_sorted[total_col].min():.6f} to {df_sorted[total_col].max():.6f}"
    ]
    plt.figtext(0.02, 0.02, '\n'.join(info_text), fontsize=9)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Generated scores sorted by 1-key plot (saved as {save_path})")

def plot_scores_by_item_pair_score(df, save_path=None):
    """Create a scatter plot of scores, sorted by 2-key scores."""
    if df.empty:
        print("No results to plot!")
        return
    
    # Determine which score columns to use
    total_col = 'total_score'
    item_col = 'item_score'
    item_pair_col = 'item_pair_score'
    title_prefix = ''
    if save_path is None:
        save_path = 'raw_scores_by_2key.png'
    
    # Sort by the item-pair score
    df_sorted = df.sort_values(item_pair_col)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with color-coded points
    plt.scatter(range(len(df_sorted)), df_sorted[total_col], 
                marker='.', s=30, alpha=0.6, label='Total Score', edgecolors='none')
    plt.scatter(range(len(df_sorted)), df_sorted[item_col], 
                marker='.', s=30, alpha=0.6, label='Item Score (1-key)', edgecolors='none')
    plt.scatter(range(len(df_sorted)), df_sorted[item_pair_col], 
                marker='.', s=30, alpha=0.6, label='Item-pair Score (2-key)', edgecolors='none')
    
    # Add labels and title
    plt.xlabel(f'Layout Index (sorted by 2-key score)')
    plt.ylabel(f'{title_prefix}Score')
    plt.title(f'{title_prefix}Layout Scores Sorted by 2-Key Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate the score ranges
    score_min = min(df_sorted[total_col].min(), 
                    df_sorted[item_col].min(), 
                    df_sorted[item_pair_col].min())
    score_max = max(df_sorted[total_col].max(), 
                    df_sorted[item_col].max(), 
                    df_sorted[item_pair_col].max())
    
    # Set y-axis limits with padding
    score_range = score_max - score_min
    plt.ylim(max(0, score_min - score_range * 0.05), score_max + score_range * 0.05)
    
    # Add score range information
    info_text = [
        f"1-key Score Range: {df_sorted[item_col].min():.6f} to {df_sorted[item_col].max():.6f}",
        f"2-key Score Range: {df_sorted[item_pair_col].min():.6f} to {df_sorted[item_pair_col].max():.6f}",
        f"Total Score Range: {df_sorted[total_col].min():.6f} to {df_sorted[total_col].max():.6f}"
    ]
    plt.figtext(0.02, 0.02, '\n'.join(info_text), fontsize=9)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Generated scores sorted by 2-key plot (saved as {save_path})")

def plot_1key_vs_2key_scores(df, save_path=None, with_product=False):
    """Create a scatter plot of 1-key scores vs 2-key scores with optional product indicators."""
    if df.empty:
        print("No results to plot!")
        return
    
    # Determine which score columns to use
    item_col = 'item_score'
    item_pair_col = 'item_pair_score'
    title_prefix = ''
    if save_path is None:
        save_path = 'raw_1key_vs_2key.png'
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot for 1-key vs 2-key scores
    plt.scatter(df[item_col], df[item_pair_col], 
                marker='.', s=30, alpha=0.6, edgecolors='none', label='Layout Scores')
    
    # Add product dots if requested
    if with_product:
        # Calculate product of 1-key and 2-key scores for each layout
        df['score_product'] = df[item_col] * df[item_pair_col]
        
        # For each layout, we'll place the product on the X axis
        # and make the size of the dot proportional to the product value
        # Create an array of zeros the same length as our data
        product_x = df[item_col].values
        product_y = df[item_pair_col].values
        
        # Plot the products at the same position as their source datapoints
        # but make them red and larger
        plt.scatter(product_x, product_y, 
                    marker='o', s=df['score_product']*10000, color='red', alpha=0.3, 
                    edgecolors='red', linewidth=1, label='Score Products')
                    
        # Add a contour plot showing product lines
        x_range = np.linspace(df[item_col].min()*0.9, df[item_col].max()*1.1, 100)
        y_range = np.linspace(df[item_pair_col].min()*0.9, df[item_pair_col].max()*1.1, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = X * Y  # Product values
        
        # Plot contour lines for constant product values
        contour = plt.contour(X, Y, Z, 6, colors='gray', alpha=0.4, linestyles='--')
        plt.clabel(contour, inline=True, fontsize=8, fmt='%0.6f')
        
        title = f'{title_prefix}1-Key vs 2-Key Scores with Products'
        plt.legend(loc='upper left')
    else:
        title = f'{title_prefix}1-Key vs 2-Key Scores'
    
    # Add labels and title
    plt.xlabel(f'{title_prefix}Item Score (1-key)')
    plt.ylabel(f'{title_prefix}Item-pair Score (2-key)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Calculate correlation coefficient
    corr = df[item_col].corr(df[item_pair_col])
    
    # Add correlation info
    plt.figtext(0.02, 0.02, f'Correlation: {corr:.4f}', fontsize=9)
    
    # Save the plot
    if with_product:
        save_path = save_path.replace('.png', '_with_product.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Generated 1-key vs 2-key plot (saved as {save_path})")

def plot_score_distributions(df, save_path=None):
    """Plot the distributions of 1-key and 2-key scores together."""
    if df.empty:
        print("No results to plot!")
        return
    
    # Determine which score columns to use
    item_col = 'item_score'
    item_pair_col = 'item_pair_score'
    title_prefix = ''
    if save_path is None:
        save_path = 'raw_score_distributions.png'
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot histograms
    ax.hist(df[item_col], bins=30, alpha=0.5, label='1-key Scores')
    ax.hist(df[item_pair_col], bins=30, alpha=0.5, label='2-key Scores')
    
    # Add labels and title
    ax.set_xlabel(f'{title_prefix}Score Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{title_prefix}Distribution of 1-key and 2-key Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add distribution statistics
    item_stats = [
        f"1-key Mean: {df[item_col].mean():.6f}",
        f"1-key Median: {df[item_col].median():.6f}",
        f"1-key Std Dev: {df[item_col].std():.6f}"
    ]
    
    pair_stats = [
        f"2-key Mean: {df[item_pair_col].mean():.6f}",
        f"2-key Median: {df[item_pair_col].median():.6f}",
        f"2-key Std Dev: {df[item_pair_col].std():.6f}"
    ]
    
    stats_text = item_stats + pair_stats
    plt.figtext(0.02, 0.02, '\n'.join(stats_text), fontsize=9)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Generated score distributions plot (saved as {save_path})")

def analyze_results(df):
    """Print basic analysis of results."""
    if df.empty:
        print("No results to analyze!")
        return
    
    print(f"\nAnalyzed {len(df)} layout results")
    
    # Score statistics
    print("\nScore Statistics:")
    print(f"  Total Score: {df['total_score'].min():.6f} to {df['total_score'].max():.6f} (mean: {df['total_score'].mean():.6f})")
    print(f"  Item Score: {df['item_score'].min():.6f} to {df['item_score'].max():.6f} (mean: {df['item_score'].mean():.6f})")
    print(f"  Item-pair Score: {df['item_pair_score'].min():.6f} to {df['item_pair_score'].max():.6f} (mean: {df['item_pair_score'].mean():.6f})")
    
    # Calculate correlations
    corr_item_total = df['item_score'].corr(df['total_score'])
    corr_pair_total = df['item_pair_score'].corr(df['total_score'])
    corr_item_pair = df['item_score'].corr(df['item_pair_score'])
    
    print("\nCorrelations:")
    print(f"  Item Score to Total Score: {corr_item_total:.4f}")
    print(f"  Item-pair Score to Total Score: {corr_pair_total:.4f}")
    print(f"  Item Score to Item-pair Score: {corr_item_pair:.4f}")
    
    # Find best layouts by each score type
    best_total_idx = df['total_score'].idxmax()
    best_item_idx = df['item_score'].idxmax()
    best_pair_idx = df['item_pair_score'].idxmax()
    
    print("\nBest Layout by Total Score:")
    print(f"  Config ID: {df.loc[best_total_idx, 'config_id']}")
    print(f"  Total Score: {df.loc[best_total_idx, 'total_score']:.6f}")
    print(f"  Item Score: {df.loc[best_total_idx, 'item_score']:.6f}")
    print(f"  Item-pair Score: {df.loc[best_total_idx, 'item_pair_score']:.6f}")
    print(f"  Items: {df.loc[best_total_idx, 'items']}")
    print(f"  Positions: {df.loc[best_total_idx, 'positions']}")
    
    print("\nBest Layout by Item Score:")
    print(f"  Config ID: {df.loc[best_item_idx, 'config_id']}")
    print(f"  Total Score: {df.loc[best_item_idx, 'total_score']:.6f}")
    print(f"  Item Score: {df.loc[best_item_idx, 'item_score']:.6f}")
    print(f"  Item-pair Score: {df.loc[best_item_idx, 'item_pair_score']:.6f}")
    print(f"  Items: {df.loc[best_item_idx, 'items']}")
    print(f"  Positions: {df.loc[best_item_idx, 'positions']}")
    
    print("\nBest Layout by Item-pair Score:")
    print(f"  Config ID: {df.loc[best_pair_idx, 'config_id']}")
    print(f"  Total Score: {df.loc[best_pair_idx, 'total_score']:.6f}")
    print(f"  Item Score: {df.loc[best_pair_idx, 'item_score']:.6f}")
    print(f"  Item-pair Score: {df.loc[best_pair_idx, 'item_pair_score']:.6f}")
    print(f"  Items: {df.loc[best_pair_idx, 'items']}")
    print(f"  Positions: {df.loc[best_pair_idx, 'positions']}")

def parse_result_csv_scores_only(filepath):
    """Parse a layout results CSV file and extract only the total scores (memory efficient)."""
    try:
        # First check if file exists
        if not os.path.exists(filepath):
            return None
            
        # Check file size
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            return None
            
        # Now try to read the file
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            return None
            
        # Skip header section (configuration info)
        header_end = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:  # Empty line marks end of header
                header_end = i
                break
        
        # Find the data header
        data_section = lines[header_end+1:]
        header_row = None
        
        for i, line in enumerate(data_section):
            if 'Total score' in line:
                header_row = i
                break
        
        if header_row is None:
            return None
            
        total_scores = []
        
        # Process ALL data rows after the header
        for row_idx in range(header_row + 1, len(data_section)):
            data_row = data_section[row_idx].strip()
            if not data_row:  # Skip empty lines
                continue
                
            # Parse CSV row
            reader = csv.reader([data_row])
            try:
                row_data = next(reader)
            except:
                continue
            
            # Extract just the total score (assuming it's column 5 in 8-column format)
            total_score_idx = 5 if len(row_data) >= 8 else 3
            
            try:
                total_score = float(row_data[total_score_idx].strip('"'))
                total_scores.append(total_score)
            except (ValueError, IndexError):
                continue
                
        return total_scores
            
    except Exception as e:
        return None

def load_results_grouped_by_file(results_dir, max_files=None):
    """Load layout result files and return a dictionary grouped by file."""
    results_by_file = {}
    
    # Find CSV files
    files = glob.glob(f"{results_dir}/layout_results_*.csv")
    if max_files:
        files = files[:max_files]
    
    if not files:
        print(f"No CSV files found matching pattern: {results_dir}/layout_results_*.csv")
        return {}
    
    print(f"Found {len(files)} files to process")
    
    # Process each file
    successful_files = 0
    for i, filepath in enumerate(files):
        if i % 1000 == 0:  # Print progress every 1000 files
            print(f"Processing file {i+1}/{len(files)}: {os.path.basename(filepath)}")
        
        results = parse_result_csv(filepath)
        if results:
            file_key = os.path.basename(filepath).replace('layout_results_', '').replace('.csv', '')
            results_by_file[file_key] = results
            successful_files += 1
    
    print(f"Successfully parsed {successful_files}/{len(files)} files")
    
    return results_by_file

def calculate_mad(values):
    """Calculate Median Absolute Deviation."""
    median_val = np.median(values)
    mad = np.median(np.abs(values - median_val))
    return mad

def plot_median_mad_memory_efficient(results_dir, max_files=None, save_path=None):
    """Create median-MAD plot without loading all data into memory."""
    if save_path is None:
        save_path = 'median_mad_by_file_efficient.png'
    
    # Find CSV files
    files = glob.glob(f"{results_dir}/layout_results_*.csv")
    if max_files:
        files = files[:max_files]
    
    if not files:
        print(f"No CSV files found matching pattern: {results_dir}/layout_results_*.csv")
        return
    
    print(f"Found {len(files)} files to process")
    
    file_stats = []
    
    # Process files one at a time
    for i, filepath in enumerate(files):
        if i % 5000 == 0:
            print(f"Processing file {i+1}/{len(files)}: {os.path.basename(filepath)}")
        
        # Get just the scores for this file
        total_scores = parse_result_csv_scores_only(filepath)
        
        if total_scores and len(total_scores) > 0:
            # Calculate median and MAD immediately
            scores_array = np.array(total_scores)
            median_score = np.median(scores_array)
            mad_score = calculate_mad(scores_array)
            
            file_key = os.path.basename(filepath).replace('layout_results_', '').replace('.csv', '')
            
            file_stats.append({
                'file_key': file_key,
                'median': median_score,
                'mad': mad_score,
                'count': len(total_scores),
                'min_score': float(np.min(scores_array)),
                'max_score': float(np.max(scores_array))
            })
            
            # Debug info for first few files
            #if i < 5:
            #    print(f"DEBUG: File {file_key}: {len(total_scores)} scores, median={median_score:.6f}, MAD={mad_score:.6f}")
    
    if not file_stats:
        print("No file statistics calculated!")
        return
    
    print(f"Successfully processed {len(file_stats)} files")
    
    # Sort by file key for consistent ordering
    file_stats.sort(key=lambda x: x['file_key'])
    
    # Extract data for plotting
    medians = [stat['median'] for stat in file_stats]
    mads = [stat['mad'] for stat in file_stats]
    counts = [stat['count'] for stat in file_stats]
    file_indices = range(len(file_stats))
    
    print(f"Plotting {len(file_stats)} files...")
    print(f"DEBUG: Layouts per file - min: {min(counts)}, max: {max(counts)}, avg: {np.mean(counts):.1f}")
    print(f"DEBUG: Median range: {min(medians):.6f} to {max(medians):.6f}")
    print(f"DEBUG: MAD range: {min(mads):.6f} to {max(mads):.6f}")
    print(f"DEBUG: MAD as % of median: {np.mean([m/med*100 for m, med in zip(mads, medians) if med > 0]):.2f}%")
    
    # Create the plot
    plt.figure(figsize=(20, 8))
    
    # First plot error bars in black
    plt.errorbar(file_indices, medians, yerr=mads, fmt='none', 
                 capsize=1, elinewidth=1, color='black', alpha=0.6, zorder=1)
    
    # Then plot the median points in red on top
    plt.scatter(file_indices, medians, s=4, color='red', alpha=0.8, 
               edgecolors='none', zorder=2)
    
    # Add labels and title
    plt.xlabel('File Index (each file contains ~1000 layouts)')
    plt.ylabel('Median Total Score')
    plt.title(f'Median Total Scores with MAD Error Bars\n({len(file_stats)} files, avg {np.mean(counts):.0f} layouts each)')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show reasonable tick marks
    if len(file_stats) > 50:
        tick_step = max(1, len(file_stats) // 50)
        plt.xticks(range(0, len(file_stats), tick_step))
    
    # Add statistics text
    overall_median = np.median(medians)
    overall_mad = calculate_mad(np.array(medians))
    min_median = min(medians)
    max_median = max(medians)
    avg_mad = np.mean(mads)
    
    stats_text = [
        f"Files processed: {len(file_stats)}",
        f"Total layouts: {sum(counts):,}",
        f"Layouts per file: ~{np.mean(counts):.0f}",
        f"Median of medians: {overall_median:.6f}",
        f"MAD of medians: {overall_mad:.6f}",
        f"Average within-file MAD: {avg_mad:.6f}",
        f"Range: {min_median:.6f} to {max_median:.6f}"
    ]
    
    plt.figtext(0.02, 0.02, '\n'.join(stats_text), fontsize=10)
    
    # Add a legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='Median Score'),
        Line2D([0], [0], color='black', linewidth=2, label='MAD Error Bars')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Generated median-MAD plot (saved as {save_path})")
    
    # Save summary statistics (much smaller file)
    summary_df = pd.DataFrame(file_stats)
    summary_df.to_csv('file_median_mad_summary_efficient.csv', index=False)
    print("File statistics saved to file_median_mad_summary_efficient.csv")
    
    return file_stats

def main():
    parser = argparse.ArgumentParser(description='Analyze layout optimization results.')
    parser.add_argument('--results-dir', type=str, default='output/layouts',
                        help='Directory containing result files')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process (default: all)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug output')
    parser.add_argument('--median-mad-only', action='store_true', default=False,
                        help='Only generate the median-MAD plot (faster for large datasets)')
    args = parser.parse_args()
    
    print(f"Arguments parsed successfully!")
    print(f"median_mad_only = {args.median_mad_only}")
    print(f"results_dir = {args.results_dir}")

    # Enable more verbose output if debug flag is set
    global debug
    debug = args.debug
    
    try:
        print(f"Loading and analyzing results from {args.results_dir}")
        
        # Check if directory exists
        if not os.path.exists(args.results_dir):
            print(f"ERROR: Directory '{args.results_dir}' does not exist!")
            return
        
        # Check if directory contains any CSV files
        csv_files = glob.glob(f"{args.results_dir}/layout_results_*.csv")
        if not csv_files:
            print(f"ERROR: No layout_results_*.csv files found in '{args.results_dir}'!")
            return
        
        print(f"Found {len(csv_files)} CSV files to process")
        

        # If only median-MAD plot is requested, use memory-efficient approach  
        if args.median_mad_only:
            print("Processing files one at a time (memory-efficient approach)...")
            plot_median_mad_memory_efficient(args.results_dir, max_files=args.max_files)
            return
                
        if debug:
            # Print first few files for debugging
            print("First few files:")
            for f in csv_files[:5]:
                print(f"  {f}")
        
        # Load the results with error handling
        try:
            df = load_results(args.results_dir, max_files=args.max_files)
        except Exception as e:
            print(f"ERROR: Failed to load results: {str(e)}")
            import traceback
            traceback.print_exc()
            return
        
        if df is None or df.empty:
            print("No valid results to analyze! Check the format of your CSV files.")
            return
        
        print(f"Successfully loaded {len(df)} layout results")
        
        if debug:
            # Print DataFrame info for debugging
            print("\nDataFrame columns:")
            print(df.columns.tolist())
            print("\nFirst few rows:")
            print(df.head(3))

        try:
            # Generate original plots
            plot_scores_scatter(df, save_path=f'scores_scatter.png')
            
            # Generate new plots
            plot_scores_by_item_score(df, save_path=f'scores_by_1key.png')
            plot_scores_by_item_pair_score(df, save_path=f'scores_by_2key.png')
            plot_1key_vs_2key_scores(df, save_path=f'1key_vs_2key.png', with_product=False)
            plot_score_distributions(df, save_path=f'score_distributions.png')
            
            # Analyze results
            analyze_results(df)
            
            # Save results to Excel with all necessary columns
            columns_to_export = [
                'config_id', 
                'total_score', 
                'item_score', 
                'item_pair_score',
                'items',                # Complete letter sequence
                'positions',            # Complete position sequence
                'items_to_assign',      # Items that needed to be assigned
                'positions_to_assign',  # Available positions
                'items_assigned',       # Pre-assigned items
                'positions_assigned',   # Pre-assigned positions
                'opt_items',            # Optimized items (if available)
                'opt_positions',        # Optimized positions (if available)
                'rank'                  # Rank of layout
            ]
            
            # Select columns that exist in the DataFrame
            valid_columns = [col for col in columns_to_export if col in df.columns]
            df_export = df[valid_columns]
            
            # Sort by total score (descending)
            df_export.sort_values('total_score', ascending=False, inplace=True)
            
            # Save to Excel
            try:
                df_export.to_excel("layout_scores_summary.xlsx", index=False)
                print("\nResults saved to layout_scores_summary.xlsx")
            except Exception as e:
                print(f"Error saving Excel file: {e}")
                print("Saving to CSV instead...")
                df_export.to_csv("layout_scores_summary.csv", index=False)
                print("Results saved to layout_scores_summary.csv")
            
        except Exception as e:
            print(f"ERROR during analysis or plotting: {str(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
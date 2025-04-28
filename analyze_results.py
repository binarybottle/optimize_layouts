#!/usr/bin/env python3
"""
Analyze layout optimization results and create scatter plots of scores.
Now shows weighted scores instead of normalized scores.
"""
import os
import sys
import pandas as pd
import glob
import matplotlib.pyplot as plt
import csv
import re
import argparse

try:
    from optimize_layout import visualize_keyboard_layout
    visualization_available = True
except ImportError:
    print("Warning: Could not import visualization functions")
    visualization_available = False

def parse_result_csv(filepath):
    """Parse a layout results CSV file and extract key metrics."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        if not lines:
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
            return None
            
        results = []
        if header_row + 1 < len(data_section):  # Make sure there's data after the header
            data_row = data_section[header_row + 1]
            
            # Parse CSV row properly
            reader = csv.reader([data_row])
            row_data = next(reader)
            
            # Simply look for the score columns by position
            # Based on the example format:
            # "Items","Positions","Optimized Items","Optimized Positions","Rank","Total score","Item score (unweighted)","Item-pair score (unweighted)"
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
                return None  # Not enough columns
                
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
                rank = 1
            
            try:
                total_score = float(row_data[total_score_idx].strip('"'))
            except (ValueError, IndexError):
                total_score = 0.0
            
            try:
                item_score = float(row_data[item_score_idx].strip('"'))
            except (ValueError, IndexError):
                item_score = 0.0
            
            try:
                item_pair_score = float(row_data[item_pair_score_idx].strip('"'))
            except (ValueError, IndexError):
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
            
            # Get weight values from config
            try:
                item_weight = float(config_info.get('Item weight', '0.222'))
            except ValueError:
                item_weight = 0.222
                
            try:
                item_pair_weight = float(config_info.get('Item-pair weight', '0.099'))
            except ValueError:
                item_pair_weight = 0.099
            
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
                'item_weight': item_weight,
                'item_pair_weight': item_pair_weight,
                'rank': rank
            }
            
            results.append(result)
                
        return results
            
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def load_results(results_dir, max_files=None):
    """Load layout result files and return a dataframe."""
    all_results = []
    
    # Find CSV files
    files = glob.glob(f"{results_dir}/layout_results_*.csv")
    if max_files:
        files = files[:max_files]
    
    print(f"Found {len(files)} files to process")
    
    # Process each file
    for filepath in files:
        results = parse_result_csv(filepath)
        if results:
            all_results.extend(results)
    
    print(f"Successfully parsed {len(all_results)} results")
    
    # Create DataFrame
    df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    
    # Add weighted scores instead of normalized scores
    if not df.empty:
        # Calculate weighted scores
        df['weighted_item_score'] = df['item_score'] * df['item_weight']
        df['weighted_item_pair_score'] = df['item_pair_score'] * df['item_pair_weight']
        
        # Verify that total_score = weighted_item_score + weighted_item_pair_score
        df['calculated_total'] = df['weighted_item_score'] + df['weighted_item_pair_score']
        
        # Check if there's a significant difference between calculated and reported total
        tol = 1e-8  # Tolerance for floating point comparison
        mismatch_count = ((df['total_score'] - df['calculated_total']).abs() > tol).sum()
        if mismatch_count > 0:
            print(f"Warning: {mismatch_count} rows have total scores that don't match the weighted sum")
            # Calculate difference statistics for mismatches
            mismatch_df = df[((df['total_score'] - df['calculated_total']).abs() > tol)]
            print(f"  Average difference: {(mismatch_df['total_score'] - mismatch_df['calculated_total']).abs().mean()}")
            print(f"  Max difference: {(mismatch_df['total_score'] - mismatch_df['calculated_total']).abs().max()}")
    
    return df

def plot_scores_scatter(df, weighted=True, save_path=None):
    """Create a scatter plot of all three score types (now with weighted instead of normalized)."""
    if df.empty:
        print("No results to plot!")
        return
    
    # Determine which score columns to use
    if weighted:
        total_col = 'total_score'
        item_col = 'weighted_item_score'
        item_pair_col = 'weighted_item_pair_score'
        title_prefix = 'Weighted '
        if save_path is None:
            save_path = 'weighted_scores_scatter.png'
    else:
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
    
    # Calculate average weights for info text
    avg_item_weight = df['item_weight'].mean()
    avg_item_pair_weight = df['item_pair_weight'].mean()
    
    # Add score range information
    if weighted:
        info_text = [
            f"Average weights - Item: {avg_item_weight:.3f}, Item-pair: {avg_item_pair_weight:.3f}",
            f"Total Score: {df_sorted[total_col].min():.6f} to {df_sorted[total_col].max():.6f}",
            f"Weighted Item Score: {df_sorted[item_col].min():.6f} to {df_sorted[item_col].max():.6f}",
            f"Weighted Item-pair Score: {df_sorted[item_pair_col].min():.6f} to {df_sorted[item_pair_col].max():.6f}"
        ]
    else:
        info_text = [
            f"Average weights - Item: {avg_item_weight:.3f}, Item-pair: {avg_item_pair_weight:.3f}",
            f"Total Score: {df_sorted[total_col].min():.6f} to {df_sorted[total_col].max():.6f}",
            f"Item Score (unweighted): {df_sorted[item_col].min():.6f} to {df_sorted[item_col].max():.6f}",
            f"Item-pair Score (unweighted): {df_sorted[item_pair_col].min():.6f} to {df_sorted[item_pair_col].max():.6f}"
        ]
    plt.figtext(0.02, 0.02, '\n'.join(info_text), fontsize=9)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Generated {title_prefix.lower()}scores scatter plot (saved as {save_path})")

def analyze_results(df):
    """Print basic analysis of results."""
    if df.empty:
        print("No results to analyze!")
        return
    
    print(f"\nAnalyzed {len(df)} layout results")
    
    # Score statistics
    print("\nScore Statistics:")
    print(f"  Total Score: {df['total_score'].min():.6f} to {df['total_score'].max():.6f} (mean: {df['total_score'].mean():.6f})")
    print(f"  Item Score (unweighted): {df['item_score'].min():.6f} to {df['item_score'].max():.6f} (mean: {df['item_score'].mean():.6f})")
    print(f"  Item-pair Score (unweighted): {df['item_pair_score'].min():.6f} to {df['item_pair_score'].max():.6f} (mean: {df['item_pair_score'].mean():.6f})")
    
    if 'weighted_item_score' in df.columns:
        print(f"  Weighted Item Score: {df['weighted_item_score'].min():.6f} to {df['weighted_item_score'].max():.6f} (mean: {df['weighted_item_score'].mean():.6f})")
        print(f"  Weighted Item-pair Score: {df['weighted_item_pair_score'].min():.6f} to {df['weighted_item_pair_score'].max():.6f} (mean: {df['weighted_item_pair_score'].mean():.6f})")
        print(f"  Average weights - Item: {df['item_weight'].mean():.3f}, Item-pair: {df['item_pair_weight'].mean():.3f}")
    
    # Find best layouts by each score type
    best_total_idx = df['total_score'].idxmax()
    best_item_idx = df['item_score'].idxmax()
    best_pair_idx = df['item_pair_score'].idxmax()
    
    print("\nBest Layout by Total Score:")
    print(f"  Config ID: {df.loc[best_total_idx, 'config_id']}")
    print(f"  Total Score: {df.loc[best_total_idx, 'total_score']:.6f}")
    print(f"  Item Score (unweighted): {df.loc[best_total_idx, 'item_score']:.6f}")
    print(f"  Item-pair Score (unweighted): {df.loc[best_total_idx, 'item_pair_score']:.6f}")
    if 'weighted_item_score' in df.columns:
        print(f"  Weighted Item Score: {df.loc[best_total_idx, 'weighted_item_score']:.6f}")
        print(f"  Weighted Item-pair Score: {df.loc[best_total_idx, 'weighted_item_pair_score']:.6f}")
    print(f"  Items: {df.loc[best_total_idx, 'items']}")
    print(f"  Positions: {df.loc[best_total_idx, 'positions']}")
    
    print("\nBest Layout by Item Score:")
    print(f"  Config ID: {df.loc[best_item_idx, 'config_id']}")
    print(f"  Total Score: {df.loc[best_item_idx, 'total_score']:.6f}")
    print(f"  Item Score (unweighted): {df.loc[best_item_idx, 'item_score']:.6f}")
    print(f"  Item-pair Score (unweighted): {df.loc[best_item_idx, 'item_pair_score']:.6f}")
    if 'weighted_item_score' in df.columns:
        print(f"  Weighted Item Score: {df.loc[best_item_idx, 'weighted_item_score']:.6f}")
        print(f"  Weighted Item-pair Score: {df.loc[best_item_idx, 'weighted_item_pair_score']:.6f}")
    print(f"  Items: {df.loc[best_item_idx, 'items']}")
    print(f"  Positions: {df.loc[best_item_idx, 'positions']}")
    
    print("\nBest Layout by Item-pair Score:")
    print(f"  Config ID: {df.loc[best_pair_idx, 'config_id']}")
    print(f"  Total Score: {df.loc[best_pair_idx, 'total_score']:.6f}")
    print(f"  Item Score (unweighted): {df.loc[best_pair_idx, 'item_score']:.6f}")
    print(f"  Item-pair Score (unweighted): {df.loc[best_pair_idx, 'item_pair_score']:.6f}")
    if 'weighted_item_score' in df.columns:
        print(f"  Weighted Item Score: {df.loc[best_pair_idx, 'weighted_item_score']:.6f}")
        print(f"  Weighted Item-pair Score: {df.loc[best_pair_idx, 'weighted_item_pair_score']:.6f}")
    print(f"  Items: {df.loc[best_pair_idx, 'items']}")
    print(f"  Positions: {df.loc[best_pair_idx, 'positions']}")

def main():
    parser = argparse.ArgumentParser(description='Analyze layout optimization results.')
    parser.add_argument('--results-dir', type=str, default='output/layouts',
                      help='Directory containing result files')
    parser.add_argument('--max-files', type=int, default=None,
                      help='Maximum number of files to process (default: all)')
    args = parser.parse_args()
    
    print(f"Loading and analyzing results from {args.results_dir}")
    df = load_results(args.results_dir, max_files=args.max_files)
    
    if not df.empty:
        # Generate both plots
        plot_scores_scatter(df, weighted=False, save_path='raw_scores_scatter.png')
        plot_scores_scatter(df, weighted=True, save_path='weighted_scores_scatter.png')
        
        # Analyze results
        analyze_results(df)
        
        # Save results to Excel with all necessary columns
        columns_to_export = [
            'config_id', 
            'total_score', 
            'item_score', 
            'item_pair_score',
            'weighted_item_score',
            'weighted_item_pair_score',
            'item_weight',
            'item_pair_weight',
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
        df_export.to_excel("layout_scores_summary.xlsx", index=False)
        print("\nResults saved to layout_scores_summary.xlsx")
    else:
        print("No results to analyze!")

if __name__ == "__main__":
    main()
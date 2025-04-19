#!/usr/bin/env python3
"""
Analyze layout optimization results and create plots of component scores.
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
                rank_idx = 4
                total_score_idx = 5
                item_score_idx = 6
                item_pair_score_idx = 7
            elif len(row_data) >= 6:  # Shorter format
                items_idx = 0
                positions_idx = 1
                rank_idx = 2
                total_score_idx = 3
                item_score_idx = 4
                item_pair_score_idx = 5
            else:
                return None  # Not enough columns
                
            # Extract data
            items = row_data[items_idx].strip('"') if items_idx < len(row_data) else ""
            positions = row_data[positions_idx].strip('"') if positions_idx < len(row_data) else ""
            
            try:
                rank = int(row_data[rank_idx].strip('"')) if rank_idx < len(row_data) else 1
            except ValueError:
                rank = 1
            
            try:
                # Parse the total score directly from the string
                raw_total = row_data[total_score_idx].strip('"')
                total_score = float(raw_total)
            except (ValueError, IndexError):
                total_score = 0.0
            
            try:
                raw_item = row_data[item_score_idx].strip('"')
                item_score = float(raw_item)
            except (ValueError, IndexError):
                item_score = 0.0
            
            try:
                raw_item_pair = row_data[item_pair_score_idx].strip('"')
                item_pair_score = float(raw_item_pair)
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
            
            # Create result dictionary
            result = {
                'items': items,
                'positions': clean_positions,
                'raw_positions': positions,
                'rank': rank,
                'raw_total_score': raw_total,  # Keep the raw string
                'parsed_total_score': total_score,  # Keep the parsed float
                'total_score': total_score,  # This is what gets used in analysis
                'item_score': item_score,
                'item_pair_score': item_pair_score,
                'file_id': os.path.basename(filepath),
                'items_assigned': config_info.get('Assigned items', ''),
                'positions_assigned': config_info.get('Assigned positions', '')
            }
            
            # Add config ID
            config_id = os.path.basename(filepath).replace('layout_results_', '').replace('.csv', '')
            result['config_id'] = config_id
            
            # Add configuration info
            for k, v in config_info.items():
                if k not in result:
                    result[k] = v
            
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
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()

def plot_component_scores(df, save_path='component_scores.png'):
    """Plot item score and item-pair score."""
    if df.empty:
        print("No results to plot!")
        return
    
    # Sort by item score
    df_sorted = df.sort_values('item_score')
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot component scores
    plt.plot(df_sorted['item_score'].values, marker='s', linestyle='-', 
             linewidth=1.5, markersize=4, alpha=0.8, label='Item Score (unweighted)')
    plt.plot(df_sorted['item_pair_score'].values, marker='^', linestyle='-', 
             linewidth=1.5, markersize=4, alpha=0.8, label='Item-pair Score (unweighted)')
    
    # Add labels and title
    plt.xlabel('Layout Index (sorted by item score)')
    plt.ylabel('Score')
    plt.title('Layout Optimization Component Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits
    score_min = min(df_sorted['item_score'].min(), df_sorted['item_pair_score'].min())
    score_max = max(df_sorted['item_score'].max(), df_sorted['item_pair_score'].max())
    score_range = score_max - score_min
    plt.ylim(max(0, score_min - score_range * 0.05), score_max + score_range * 0.05)
    
    # Add info text
    info_text = [
        f"Item Score: {df_sorted['item_score'].min():.6f} to {df_sorted['item_score'].max():.6f}",
        f"Item-pair Score: {df_sorted['item_pair_score'].min():.6f} to {df_sorted['item_pair_score'].max():.6f}"
    ]
    plt.figtext(0.02, 0.02, '\n'.join(info_text), fontsize=9)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Generated component scores plot (saved as {save_path})")

def analyze_results(df):
    """Print basic analysis of results."""
    if df.empty:
        print("No results to analyze!")
        return
    
    print(f"\nAnalyzed {len(df)} layout results")
    
    # Component score statistics
    print("\nItem Score Statistics:")
    print(f"  Mean: {df['item_score'].mean():.6f}")
    print(f"  Min: {df['item_score'].min():.6f}")
    print(f"  Max: {df['item_score'].max():.6f}")
    
    print("\nItem-pair Score Statistics:")
    print(f"  Mean: {df['item_pair_score'].mean():.6f}")
    print(f"  Min: {df['item_pair_score'].min():.6f}")
    print(f"  Max: {df['item_pair_score'].max():.6f}")
    
    # Find best layouts by component scores
    best_item_idx = df['item_score'].idxmax()
    best_pair_idx = df['item_pair_score'].idxmax()
    
    print("\nBest Layout by Item Score:")
    print(f"  Config ID: {df.loc[best_item_idx, 'config_id']}")
    print(f"  Item Score: {df.loc[best_item_idx, 'item_score']:.6f}")
    print(f"  Item-pair Score: {df.loc[best_item_idx, 'item_pair_score']:.6f}")
    
    print("\nBest Layout by Item-pair Score:")
    print(f"  Config ID: {df.loc[best_pair_idx, 'config_id']}")
    print(f"  Item Score: {df.loc[best_pair_idx, 'item_score']:.6f}")
    print(f"  Item-pair Score: {df.loc[best_pair_idx, 'item_pair_score']:.6f}")

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
        # Show raw vs parsed total scores for a sample
        print("\nSample of raw vs parsed total scores:")
        sample_df = df.sample(min(5, len(df)))
        for _, row in sample_df.iterrows():
            print(f"  Raw: {row['raw_total_score']}, Parsed: {row['parsed_total_score']}")
        
        # Generate plots and analysis
        plot_component_scores(df)
        analyze_results(df)
        
        # Save results to Excel
        df.to_excel("component_scores_summary.xlsx", index=False)
        print("\nResults saved to component_scores_summary.xlsx")
    else:
        print("No results to analyze!")

if __name__ == "__main__":
    main()
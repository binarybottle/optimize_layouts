#!/usr/bin/env python3
"""
Analyze the results of the layout optimization runs.
Includes visual display of top layouts, if specified.

Usage:
>> python analyze_results.py --top 5 --config config.yaml
"""
import os
import sys
import pandas as pd
import glob
import matplotlib.pyplot as plt
import csv
import re
import argparse
import yaml

# Import the original visualization functions
sys.path.append('.')  # Ensure current directory is in path
try:
    from optimize_layout import visualize_keyboard_layout, load_config
    visualization_available = True
except ImportError:
    print("Warning: Could not import functions from optimize_layout.py")
    visualization_available = False

def parse_result_csv(filepath):
    """Parse a layout results CSV file and extract key metrics."""
    try:
        # First try to read the header information to understand the structure
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Check if this is an empty file
        if not lines:
            print(f"Empty file: {filepath}")
            return None
            
        # Extract configuration info
        config_info = {}
        header_end = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:  # Empty line marks end of header
                header_end = i
                break
                
            # Try to split by comma first
            parts = line.split(',')
            if len(parts) >= 2:
                key, value = parts[0].strip('"'), parts[1].strip('"')
                config_info[key] = value
        
        # Parse the actual data section (after the header)
        data_section = lines[header_end+1:]  # Skip the empty line
        
        # Find the header row
        header_row = None
        for i, line in enumerate(data_section):
            if 'Items' in line or 'Rank' in line or 'Score' in line:
                header_row = i
                break
        
        if header_row is None:
            print(f"Could not find data header in {filepath}")
            return None
            
        # Get all data rows (multiple layouts might be in one file)
        data_rows = data_section[header_row + 1:]
        results = []
        
        for data_row in data_rows:
            if not data_row.strip():
                continue
                
            # Parse the data row (handle different formats)
            try:
                # Try with csv module first (handles quoting properly)
                reader = csv.reader([data_row])
                row_data = next(reader)
                
                # Extract data based on position - older files may have different columns
                if len(row_data) >= 6:  # Newer format with 6+ columns
                    items = row_data[0]
                    positions = row_data[1]
                    rank = int(row_data[2]) if row_data[2].isdigit() else 1
                    total_score = float(row_data[3])
                    item_score = float(row_data[4])
                    item_pair_score = float(row_data[5])
                elif len(row_data) >= 3:  # Older format with 3+ columns
                    items = row_data[0]
                    positions = row_data[1]
                    total_score = float(row_data[2])
                    # Estimate these if not available
                    item_score = total_score * 0.5  # Assuming equal weights
                    item_pair_score = total_score * 0.5
                    rank = 1
                else:
                    continue  # Not enough data
                    
                # Clean up special characters in positions string
                clean_positions = positions
                for special, replacement in [
                    ('[semicolon]', ';'),
                    ('[comma]', ','),
                    ('[period]', '.'),
                    ('[slash]', '/')
                ]:
                    clean_positions = clean_positions.replace(special, replacement)
                
                # Extract file identifier - this will be used for exact matching
                file_id = os.path.basename(filepath)
                
                # Get assigned items/positions from the config_info if available
                assigned_items = config_info.get('Assigned items', '')
                assigned_positions = config_info.get('Assigned positions', '')
                
                result = {
                    'items': items,
                    'positions': clean_positions,  # Use cleaned positions
                    'raw_positions': positions,    # Keep original for reference
                    'rank': rank,
                    'total_score': total_score,
                    'item_score': item_score,
                    'item_pair_score': item_pair_score,
                    'file_id': file_id,
                    'items_assigned': assigned_items,
                    'positions_assigned': assigned_positions
                }

                # Extract config ID from filepath - use the full basename to avoid duplicates
                config_id = os.path.basename(filepath).replace('layout_results_', '').replace('.csv', '')
                result['config_id'] = config_id
                
                # Add any configuration info we found
                for k, v in config_info.items():
                    if k not in result:
                        result[k] = v
                
                results.append(result)
                
            except Exception as e:
                print(f"Error parsing data row in {filepath}: {e}")
                continue
                
        return results
            
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def load_all_results(results_dir):
    """Load all result files and return a dataframe."""
    all_results = []
    processed_files = set()  # Track processed files to avoid duplicates
    
    # Use a single pattern that makes sense for your directory structure
    pattern = f"{results_dir}/**/layout_results_*.csv"
    
    # Find all CSV result files
    all_files = glob.glob(pattern, recursive=True)
    
    # Remove duplicates by normalizing paths
    unique_files = set(os.path.abspath(f) for f in all_files)
    
    print(f"Found {len(unique_files)} unique result files")
    
    # Process each file
    for filepath in unique_files:
        if filepath in processed_files:
            continue
            
        results = parse_result_csv(filepath)
        if results:
            all_results.extend(results)
            processed_files.add(filepath)
    
    print(f"Successfully parsed {len(processed_files)} result files")
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()

def create_visualization_config(row, base_config=None):
    """Create a config dictionary for visualization based on a result row and base config."""
    # Start with the base config if available
    if base_config:
        config = base_config.copy()
    else:
        # Create a minimal config for visualization
        config = {
            'optimization': {
                'items_to_assign': row['items'],
                'positions_to_assign': row['positions'].upper(),
                'items_to_constrain': row.get('Items to constrain', ''),
                'positions_to_constrain': row.get('Constraint positions', ''),
                'items_assigned': row.get('items_assigned', ''),
                'positions_assigned': row.get('positions_assigned', ''),
                'scoring': {
                    'item_weight': float(row.get('Item weight', 0.5)),
                    'item_pair_weight': float(row.get('Item-pair weight', 0.5)),
                    'missing_item_pair_norm_score': 1.0,
                    'missing_position_pair_norm_score': 1.0
                },
                'nlayouts': 10
            },
            'visualization': {
                'print_keyboard': True
            }
        }
    
    # Ensure the minimum required fields are set for visualization
    if 'optimization' not in config:
        config['optimization'] = {}
    
    # Override with result-specific data
    config['optimization']['items_to_assign'] = row['items']
    config['optimization']['positions_to_assign'] = row['positions'].upper()
    
    # Use row values if present, otherwise keep what's in the base config
    if 'items_assigned' in row and row['items_assigned']:
        config['optimization']['items_assigned'] = row['items_assigned']
    if 'positions_assigned' in row and row['positions_assigned']:
        config['optimization']['positions_assigned'] = row['positions_assigned']
    
    return config

def create_visualization_mapping(row):
    """Create a proper mapping dictionary from a result row."""
    mapping = {}
    items = row['items']
    positions = row['positions'].upper()  # Convert to uppercase for visualization
    
    # Create mapping of each letter to its position
    for i, letter in enumerate(items):
        if i < len(positions):
            mapping[letter] = positions[i]
    
    # Include any assigned items if they're present in the data
    if 'items_assigned' in row and 'positions_assigned' in row:
        items_assigned = row['items_assigned']
        positions_assigned = row['positions_assigned']
        
        for i, letter in enumerate(items_assigned):
            if i < len(positions_assigned):
                mapping[letter] = positions_assigned[i]
    
    return mapping

def analyze_results(df, display_top=3, base_config=None):
    """Analyze results and generate insights."""
    if df.empty:
        print("No results to analyze!")
        return
    
    print(f"Analyzed {len(df)} layout results from {df['config_id'].nunique()} configurations")
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"Mean total score: {df['total_score'].mean():.6f}")
    print(f"Best total score: {df['total_score'].max():.6f}")
    print(f"Worst total score: {df['total_score'].min():.6f}")
    
    # Only group if we have multiple configurations
    if df['config_id'].nunique() > 1:
        print("\nPerformance by Configuration:")
        # For each configuration, get the best score
        config_stats = df.groupby('config_id')['total_score'].agg(['max']).rename(columns={'max': 'best_score'})
        config_stats['rank'] = config_stats['best_score'].rank(ascending=False).astype(int)
        config_stats = config_stats.sort_values('best_score', ascending=False)
        
        print("Top 10 Configurations by Score:")
        print(config_stats.head(10))
        
        print("\nBottom 10 Configurations by Score:")
        print(config_stats.tail(10))
    
    # Find the best layout overall
    best_idx = df['total_score'].idxmax()
    best_layout = df.iloc[best_idx]
    
    print("\nBest Layout Overall:")
    print(f"Config ID: {best_layout.get('config_id', 'N/A')}")
    print(f"Total Score: {best_layout['total_score']:.6f}")
    print(f"Items: {best_layout['items']}")
    print(f"Positions: {best_layout['positions']}")
    print(f"Raw Positions: {best_layout.get('raw_positions', 'N/A')}")
    print(f"Source file: {best_layout.get('file_id', 'unknown')}")
    
    # Show the assigned positions from the filename
    if best_layout.get('items_assigned') and best_layout.get('positions_assigned'):
        print(f"Pre-assigned: {best_layout['items_assigned']} in positions {best_layout['positions_assigned']}")
    
    # Try to identify the letters in better format
    try:
        items = best_layout['items']
        positions = best_layout['positions']
        
        if len(items) == len(positions):
            print("\nLetter to Position Mapping:")
            for i, letter in enumerate(items):
                print(f"  {letter} -> {positions[i]}")
    except:
        pass
    
    # Display top layouts on CLI if visualization is available
    if visualization_available:
        print("\nDisplaying top layouts:")
        
        # Get the top results
        top_df = df.sort_values('total_score', ascending=False).head(display_top)
        
        for i, (idx, row) in enumerate(top_df.iterrows(), 1):
            print(f"\n#{i}: Score: {row['total_score']:.6f}")
            print(f"Config ID: {row['config_id']}")
            
            # Create a custom config for this specific layout
            config = create_visualization_config(row, base_config)
            
            # Create the mapping
            mapping = create_visualization_mapping(row)
            
            # Display configuration details
            print(f"File: {row['file_id']}")
            print(f"Items to assign: {row['items']}")
            print(f"Positions to assign: {row['positions'].upper()}")
            if row.get('items_assigned'):
                print(f"Pre-assigned: {row['items_assigned']} in positions {row['positions_assigned']}")
            
            # Use the visualization function for this specific layout
            try:
                visualize_keyboard_layout(
                    mapping=mapping,
                    title=f"Layout #{i} (Config: {row['config_id']})",
                    config=config
                )
            except Exception as e:
                print(f"Error visualizing layout: {e}")
                print("Letter mapping:")
                for letter, pos in mapping.items():
                    print(f"  {letter} -> {pos}")
    else:
        print("\nVisualization not available - printing text results only:")
        top_df = df.sort_values('total_score', ascending=False).head(display_top)
        for i, (_, row) in enumerate(top_df.iterrows(), 1):
            print(f"\n#{i}: Score: {row['total_score']:.6f}")
            print(f"Items: {row['items']}")
            print(f"Positions: {row['positions']}")
    
    # Save to Excel for further analysis
    df.to_excel("optimization_results_summary.xlsx", index=False)
    print("\nResults summary saved to optimization_results_summary.xlsx")
    
    # Generate visualizations
    try:
        # Score distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['total_score'], bins=20, alpha=0.7)
        plt.title('Distribution of Layout Scores')
        plt.xlabel('Total Score')
        plt.ylabel('Frequency')
        plt.savefig('score_distribution.png')
        print("Generated score distribution chart")
        
        # If we have multiple configs, plot them
        if df['config_id'].nunique() > 1:
            plt.figure(figsize=(14, 6))
            # Get the best score for each config
            best_scores = df.groupby('config_id')['total_score'].max().sort_values(ascending=False)
            
            # Limit to top 30 for readability
            if len(best_scores) > 30:
                best_scores = best_scores.head(30)
                
            best_scores.plot(kind='bar')
            plt.title('Best Score by Configuration (Top 30)')
            plt.xlabel('Configuration')
            plt.ylabel('Best Score')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig('config_performance.png')
            print("Generated configuration performance chart")
            
    except Exception as e:
        print(f"Error generating visualizations: {e}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze layout optimization results.')
    parser.add_argument('--top', type=int, default=3, 
                        help='Number of top layouts to display (default: 3)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Directory containing result files (default: from config)')
    args = parser.parse_args()
    
    # Load the configuration file
    try:
        base_config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
        
        # Get results directory from config if not specified
        if args.results_dir is None:
            results_dir = base_config['paths']['output']['layout_results_folder']
        else:
            results_dir = args.results_dir
            
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default settings")
        base_config = None
        results_dir = 'output/layouts' if args.results_dir is None else args.results_dir
    
    print(f"Using results directory: {results_dir}")
    print("Loading and analyzing optimization results...")
    
    results_df = load_all_results(results_dir)
    analyze_results(results_df, display_top=args.top, base_config=base_config)
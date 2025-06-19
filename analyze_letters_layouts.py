#!/usr/bin/env python3
"""
Analyze Keyboard Layout Target Matches

This script processes keyboard layout optimization results to count how many
target item-position matches occur in each layout. It's designed to analyze
specific combinations like vowels (e,a,o,i) placed on home row positions.

Features:
- Processes multiple CSV files containing layout results
- Counts target item-position matches for each individual layout
- Handles special character replacements (semicolon, comma, etc.)
- Generates summary statistics and distributions
- Exports results with one row per layout

Target Analysis:
- Target items: e, a, o, i (common vowels)
- Target positions: U, I, O, J, K, L, ;, M (right-hand home/adjacent keys)

Input:
- Directory containing CSV files with layout results
- Expected format: "Items","Positions" columns with layout data

Output:
- layout_match_counts.csv (one row per layout with match counts)
- Console summary with distribution statistics

Usage:
    python analyze_letters_layouts.py
    # (will prompt for directory path)
"""
import pandas as pd
import os
import glob
from typing import Dict
import re

def parse_csv_file(filepath: str) -> pd.DataFrame:
    """
    Parse a CSV file with metadata at the top and data table below.
    
    Returns:
        data_df (DataFrame): The actual data table
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Find where the data table starts (look for the header row)
    data_start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('"Items","Positions"'):
            data_start_idx = i
            break
    
    if data_start_idx is None:
        raise ValueError(f"Could not find data table header in {filepath}")
    
    # Parse data table
    data_lines = lines[data_start_idx:]
    data_content = ''.join(data_lines)
    
    # Use pandas to read the CSV data
    from io import StringIO
    data_df = pd.read_csv(StringIO(data_content))
    
    return data_df

def count_target_matches_in_layout(items_str: str, positions_str: str, target_items: set, target_positions: set) -> int:
    """
    Count number of target item-position matches in a single layout.
    
    Args:
        items_str: String of items (each character is an item)
        positions_str: String of positions (each character is a position)
        target_items: Set of items to look for
        target_positions: Set of positions to look for
    
    Returns:
        Count of target matches in this layout
    """
    match_count = 0
    
    # Handle special character replacements in positions
    positions_str = positions_str.replace('[semicolon]', ';')
    positions_str = positions_str.replace('[comma]', ',')
    positions_str = positions_str.replace('[slash]', '/')
    positions_str = positions_str.replace('[period]', '.')
    
    # Count matches in this layout
    min_len = min(len(items_str), len(positions_str))
    for i in range(min_len):
        item = items_str[i]
        position = positions_str[i]
        
        if item in target_items and position in target_positions:
            match_count += 1
    
    return match_count

def process_all_files(directory_path: str, file_pattern: str = "*.csv") -> pd.DataFrame:
    """
    Process all CSV files and generate one row per layout with match counts.
    
    Args:
        directory_path: Path to directory containing CSV files
        file_pattern: Pattern to match CSV files (default: "*.csv")
    
    Returns:
        DataFrame with one row per layout showing match counts
    """
    # Define target items and positions
    target_items = {'e', 'a', 'o', 'i'}
    target_positions = {'U', 'I', 'O', 'J', 'K', 'L', ';', 'M'}
    #target_positions = {'W', 'E', 'R', 'A', 'S', 'D', 'F', 'V'}


    # Find all CSV files
    file_pattern_full = os.path.join(directory_path, file_pattern)
    csv_files = glob.glob(file_pattern_full)
    
    print(f"Found {len(csv_files)} CSV files to process...")
    print(f"Expected total layouts: {len(csv_files) * 1000:,}")
    
    results = []
    
    for i, filepath in enumerate(csv_files):
        if i % 1000 == 0:  # Progress update every 1000 files
            print(f"Processing file {i+1:,}/{len(csv_files):,}: {os.path.basename(filepath)}")
        
        try:
            data_df = parse_csv_file(filepath)
            filename = os.path.basename(filepath)
            
            # Process each layout (row) in this file
            for row_idx, row in data_df.iterrows():
                items_str = str(row.get('Items', ''))
                positions_str = str(row.get('Positions', ''))
                
                # Count matches for this specific layout
                match_count = count_target_matches_in_layout(items_str, positions_str, target_items, target_positions)
                
                # Create result row for this layout
                result_row = {
                    'filename': filename,
                    'layout_number': row_idx + 1,  # 1-indexed layout number within file
                    'target_matches_count': match_count,
                    'items': items_str,
                    'positions': positions_str,
                    'rank': row.get('Rank', ''),
                    'total_score': row.get('Total score', ''),
                    'optimized_items': row.get('Optimized Items', ''),
                    'optimized_positions': row.get('Optimized Positions', '')
                }
                
                results.append(result_row)
        
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        # Sort by match count (descending), then by filename and layout number
        results_df = results_df.sort_values(['target_matches_count', 'filename', 'layout_number'], 
                                          ascending=[False, True, True])
        results_df = results_df.reset_index(drop=True)
    
    return results_df

def generate_summary_report(results_df: pd.DataFrame) -> None:
    """Generate a summary report of the findings."""
    if results_df.empty:
        print("No layouts processed.")
        return
    
    print(f"\n=== SUMMARY REPORT ===")
    print(f"Total layouts processed: {len(results_df):,}")
    print(f"Total files processed: {results_df['filename'].nunique():,}")
    
    # Count layouts with matches
    layouts_with_matches = results_df[results_df['target_matches_count'] > 0]
    print(f"Layouts with target matches: {len(layouts_with_matches):,}")
    print(f"Layouts with no matches: {len(results_df) - len(layouts_with_matches):,}")
    
    if len(layouts_with_matches) > 0:
        print(f"Total target matches across all layouts: {results_df['target_matches_count'].sum():,}")
        print(f"Average matches per layout (layouts with matches): {layouts_with_matches['target_matches_count'].mean():.2f}")
        print(f"Max matches in a single layout: {results_df['target_matches_count'].max()}")
        
        print(f"\n=== TOP 10 LAYOUTS WITH MOST MATCHES ===")
        top_layouts = results_df.head(10)
        for _, row in top_layouts.iterrows():
            print(f"{row['filename']} (layout {row['layout_number']}): {row['target_matches_count']} matches, rank {row['rank']}")
        
        print(f"\n=== MATCH COUNT DISTRIBUTION ===")
        match_counts = results_df['target_matches_count'].value_counts().sort_index()
        for count, num_layouts in match_counts.head(20).items():
            if count >= 0:  # Show all counts including 0
                print(f"{count} matches: {num_layouts:,} layouts")

# Main execution
if __name__ == "__main__":
    # Set the directory path where your CSV files are located
    directory_path = input("Enter the directory path containing your CSV files: ").strip()
    
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        exit(1)
    
    # Process all files
    print("Starting layout processing...")
    print("Target items: e, a, o, i")
    print("Target positions: U, I, O, P, J, K, L, ;, M, ., /")
    print("Output: One row per layout with match count")
    print()
    
    results = process_all_files(directory_path)
    
    # Generate summary report
    generate_summary_report(results)
    
    # Save results to CSV
    if not results.empty:
        output_filename = "layout_match_counts.csv"
        results.to_csv(output_filename, index=False)
        print(f"\nResults saved to: {output_filename}")
        print(f"Output contains {len(results):,} rows (one per layout)")
        
        # Display first few results
        print(f"\n=== TOP 10 LAYOUTS BY MATCH COUNT ===")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(results.head(10).to_string(index=False))
    else:
        print("No layouts were processed successfully.")
    
    print(f"\nProcessing complete!")
#!/usr/bin/env python3
"""
Global Pareto Front Selection for Layout MOO Results

This script processes all MOO results files in output/layouts/ directory,
finds the global Pareto front across all solutions, and outputs a single
CSV file with the globally optimal solutions.

Leverages parsing functions from analyze_results.py for robust file handling.

# Basic usage
python select_global_moo_solutions.py

# Test with limited files first
python select_global_moo_solutions.py --max-files 100 --verbose

# Custom pattern if needed
python select_global_moo_solutions.py --file-pattern "moo_results_config_*.csv"

# Different objectives if needed
python select_global_moo_solutions.py --objectives "engram7_load" "engram7_strength"
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import time
import glob

# Import parsing functions from analyze_results.py
try:
    from analyze_results import (
        process_files_batch, 
        parse_result_csv,
        _get_column_indices,
        _extract_scores_only_new,
        _extract_full_results_new
    )
    print("Successfully imported parsing functions from analyze_results.py")
except ImportError as e:
    print(f"Warning: Could not import from analyze_results.py: {e}")
    print("Make sure analyze_results.py is in the same directory")
    # Fallback to basic implementation
    def process_files_batch(results_dir, file_pattern="moo_results_config_*.csv", max_files=None, progress_step=1000):
        pattern_path = f"{results_dir}/{file_pattern}"
        files = glob.glob(pattern_path)
        if max_files:
            files = files[:max_files]
        return files
    
    def parse_result_csv(filepath, scores_only=False, debug=False):
        # Basic fallback parser - simplified version
        try:
            df = pd.read_csv(filepath, skiprows=4)  # Skip metadata
            return df.to_dict('records') if not scores_only else None
        except:
            return None


def is_dominated(solution_a: pd.Series, solution_b: pd.Series, 
                objectives: List[str], maximize: List[bool]) -> bool:
    """
    Check if solution_a is dominated by solution_b.
    
    Args:
        solution_a: First solution
        solution_b: Second solution  
        objectives: List of objective column names
        maximize: List of booleans indicating if each objective should be maximized
        
    Returns:
        True if solution_a is dominated by solution_b
    """
    better_in_at_least_one = False
    worse_in_any = False
    
    for obj, is_max in zip(objectives, maximize):
        a_val = solution_a[obj]
        b_val = solution_b[obj]
        
        if pd.isna(a_val) or pd.isna(b_val):
            continue
            
        if is_max:
            if b_val > a_val:
                better_in_at_least_one = True
            elif b_val < a_val:
                worse_in_any = True
        else:
            if b_val < a_val:
                better_in_at_least_one = True
            elif b_val > a_val:
                worse_in_any = True
    
    return better_in_at_least_one and not worse_in_any


def efficient_pareto_filter(solutions: pd.DataFrame, 
                          objectives: List[str],
                          maximize: List[bool]) -> pd.DataFrame:
    """
    Efficiently find the Pareto front from a set of solutions.
    
    Args:
        solutions: DataFrame with all solutions
        objectives: List of objective column names to consider
        maximize: List of booleans indicating if each objective should be maximized
        
    Returns:
        DataFrame with only Pareto optimal solutions
    """
    if len(solutions) == 0:
        return solutions
    
    # Remove any rows with NaN in objective columns
    solutions_clean = solutions.dropna(subset=objectives).copy()
    if len(solutions_clean) == 0:
        print("Warning: No solutions remaining after removing NaN values")
        return pd.DataFrame()
    
    # Sort by first objective to improve efficiency
    primary_obj = objectives[0]
    ascending = not maximize[0]
    solutions_sorted = solutions_clean.sort_values(primary_obj, ascending=ascending).reset_index(drop=True)
    
    pareto_indices = []
    
    for i, candidate in solutions_sorted.iterrows():
        is_dominated_flag = False
        
        # Check if candidate is dominated by any solution already in Pareto front
        for j in pareto_indices:
            pareto_solution = solutions_sorted.iloc[j]
            if is_dominated(candidate, pareto_solution, objectives, maximize):
                is_dominated_flag = True
                break
        
        if not is_dominated_flag:
            # Remove any solutions in current Pareto front that are dominated by candidate
            pareto_indices = [j for j in pareto_indices 
                            if not is_dominated(solutions_sorted.iloc[j], candidate, objectives, maximize)]
            pareto_indices.append(i)
    
    return solutions_sorted.iloc[pareto_indices].reset_index(drop=True)


def divide_and_conquer_pareto(solutions: pd.DataFrame,
                            objectives: List[str],
                            maximize: List[bool],
                            chunk_size: int = 10000) -> pd.DataFrame:
    """
    Handle large solution sets using divide and conquer approach.
    
    Args:
        solutions: DataFrame with all solutions
        objectives: List of objective column names
        maximize: List of booleans for maximization
        chunk_size: Maximum chunk size for processing
        
    Returns:
        DataFrame with Pareto optimal solutions
    """
    if len(solutions) <= chunk_size:
        return efficient_pareto_filter(solutions, objectives, maximize)
    
    # Divide
    mid = len(solutions) // 2
    left_pareto = divide_and_conquer_pareto(
        solutions.iloc[:mid], objectives, maximize, chunk_size)
    right_pareto = divide_and_conquer_pareto(
        solutions.iloc[mid:], objectives, maximize, chunk_size)
    
    # Conquer: merge the two Pareto fronts
    combined = pd.concat([left_pareto, right_pareto], ignore_index=True)
    return efficient_pareto_filter(combined, objectives, maximize)


def load_all_solutions(input_dir: str, file_pattern: str = "moo_results_config_*.csv", 
                      max_files: int = None, verbose: bool = False) -> pd.DataFrame:
    """
    Load all solutions from MOO results files using analyze_results.py functions.
    
    Args:
        input_dir: Directory containing CSV files
        file_pattern: Pattern to match files
        max_files: Maximum number of files to process
        verbose: Print verbose output
        
    Returns:
        DataFrame with all solutions
    """
    # Find all CSV files
    csv_files = process_files_batch(input_dir, file_pattern, max_files)
    
    if not csv_files:
        print(f"Error: No CSV files found matching pattern: {file_pattern}")
        return pd.DataFrame()
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Load all solutions
    all_solutions = []
    successful_files = 0
    total_solutions = 0
    
    start_time = time.time()
    
    for i, csv_file in enumerate(csv_files):
        if verbose and (i % 100 == 0):
            print(f"Processing {csv_file} ({i+1}/{len(csv_files)})")
        
        # Parse the file using analyze_results.py function
        results = parse_result_csv(str(csv_file), scores_only=False, debug=verbose)
        
        if results and len(results) > 0:
            # Convert to DataFrame if it's a list of dicts
            if isinstance(results, list):
                df_file = pd.DataFrame(results)
            else:
                df_file = results
                
            # Add source file information
            df_file['source_file'] = os.path.basename(csv_file)
            
            all_solutions.append(df_file)
            successful_files += 1
            total_solutions += len(df_file)
            
            if verbose:
                print(f"  Loaded {len(df_file)} solutions")
        else:
            if verbose:
                print(f"  Failed to parse {csv_file}")
    
    if not all_solutions:
        print("Error: No valid solutions found in any file")
        return pd.DataFrame()
    
    # Combine all solutions
    print(f"Combining {total_solutions:,} solutions from {successful_files} files...")
    combined_solutions = pd.concat(all_solutions, ignore_index=True)
    
    print(f"Successfully loaded {len(combined_solutions):,} solutions from {successful_files}/{len(csv_files)} files")
    print(f"Loading time: {time.time() - start_time:.2f} seconds")
    
    return combined_solutions


def main():
    parser = argparse.ArgumentParser(description='Select global Pareto optimal layouts')
    parser.add_argument('--input-dir', default='output/layouts/', 
                       help='Directory containing MOO results CSV files')
    parser.add_argument('--output-file', default='output/global_moo_solutions.csv',
                       help='Output CSV file for global Pareto solutions')
    parser.add_argument('--objectives', nargs='+', 
                       default=['Complete Item', 'Complete Pair'],
                       help='Objective columns to use for Pareto filtering')
    parser.add_argument('--maximize', nargs='+', type=lambda x: x.lower() == 'true',
                       default=[True, True],
                       help='Whether to maximize each objective (true/false for each)')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Chunk size for divide-and-conquer processing')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--file-pattern', default='moo_results_config_*.csv',
                       help='File pattern to match')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist")
        return 1
    
    print(f"Global Pareto Selection Starting...")
    print(f"Input directory: {args.input_dir}")
    print(f"File pattern: {args.file_pattern}")
    print(f"Output file: {args.output_file}")
    print(f"Objectives: {args.objectives}")
    print(f"Maximize objectives: {args.maximize}")
    
    # Load all solutions
    start_time = time.time()
    combined_solutions = load_all_solutions(
        args.input_dir, args.file_pattern, args.max_files, args.verbose)
    
    if combined_solutions.empty:
        print("Error: No solutions loaded")
        return 1
    
    # Map objective names to actual column names (handle variations)
    objective_mapping = {
        'Complete Item': ['Complete Item', 'item_score'],
        'Complete Pair': ['Complete Pair', 'item_pair_score'],
        'Opt Item Score': ['Opt Item Score', 'opt_item_score'],
        'Opt Item-Pair Score': ['Opt Item-Pair Score', 'opt_item_pair_score']
    }
    
    # Find actual column names
    actual_objectives = []
    for obj in args.objectives:
        found = False
        if obj in combined_solutions.columns:
            actual_objectives.append(obj)
            found = True
        elif obj in objective_mapping:
            for variant in objective_mapping[obj]:
                if variant in combined_solutions.columns:
                    actual_objectives.append(variant)
                    found = True
                    break
        
        if not found:
            print(f"Error: Objective '{obj}' not found in data")
            print(f"Available columns: {list(combined_solutions.columns)}")
            return 1
    
    print(f"Using objective columns: {actual_objectives}")
    
    # Find global Pareto front
    print(f"Finding global Pareto front...")
    print(f"Processing {len(combined_solutions):,} solutions with {args.chunk_size:,} chunk size...")
    
    pareto_start = time.time()
    global_pareto = divide_and_conquer_pareto(
        combined_solutions, actual_objectives, args.maximize, args.chunk_size)
    pareto_time = time.time() - pareto_start
    
    # Statistics
    reduction_factor = len(combined_solutions) / len(global_pareto) if len(global_pareto) > 0 else float('inf')
    print(f"\n=== Results ===")
    print(f"Original solutions: {len(combined_solutions):,}")
    print(f"Global Pareto solutions: {len(global_pareto):,}")
    print(f"Reduction factor: {reduction_factor:.1f}x")
    print(f"Pareto filtering time: {pareto_time:.2f} seconds")
    
    if len(global_pareto) == 0:
        print("Error: No Pareto optimal solutions found")
        return 1
    
    # Analyze source distribution
    if 'source_file' in global_pareto.columns:
        source_counts = global_pareto['source_file'].value_counts()
        print(f"\nGlobal Pareto solutions come from {len(source_counts)} different source files")
        print(f"Max solutions from single file: {source_counts.max()}")
        
        if args.verbose:
            print("\nTop contributing files:")
            for file, count in source_counts.head(10).items():
                print(f"  {file}: {count} solutions")
    
    # Objective statistics
    print(f"\nObjective ranges in global Pareto set:")
    for obj in actual_objectives:
        values = global_pareto[obj]
        print(f"  {obj}: {values.min():.6f} to {values.max():.6f}")
    
    # Reset index for clean output (no arbitrary ranking of Pareto solutions)
    global_pareto = global_pareto.reset_index(drop=True)
    
    # Clean up columns - remove empty and unnecessary columns
    columns_to_remove = [
        'items_to_assign', 'positions_to_assign', 'items_assigned', 'positions_assigned',
        'opt_items', 'opt_positions'  # These are often empty in the new format
    ]
    
    # Remove specified columns if they exist
    for col in columns_to_remove:
        if col in global_pareto.columns:
            global_pareto = global_pareto.drop(columns=[col])
    
    # Remove completely empty columns
    empty_cols = []
    for col in global_pareto.columns:
        if global_pareto[col].isna().all() or (global_pareto[col] == '').all():
            empty_cols.append(col)
    
    if empty_cols:
        global_pareto = global_pareto.drop(columns=empty_cols)
        print(f"Removed empty columns: {empty_cols}")
    
    # Reorder columns to put important ones first
    priority_cols = ['config_id', 'items', 'positions'] + actual_objectives + ['source_file']
    remaining_cols = [col for col in global_pareto.columns if col not in priority_cols]
    ordered_cols = [col for col in priority_cols if col in global_pareto.columns] + remaining_cols
    global_pareto = global_pareto[ordered_cols]
    
    # Save results
    output_path = Path(args.output_file)
    
    # Create header with metadata (matching original format)
    print(f"Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('"Global Pareto Optimal Solutions"\n')
        f.write(f'"Total original solutions","{len(combined_solutions)}"\n')
        f.write(f'"Global Pareto solutions","{len(global_pareto)}"\n')
        f.write(f'"Objectives used","{", ".join(actual_objectives)}"\n')
        f.write(f'"Maximizing objectives","{", ".join(map(str, args.maximize))}"\n')
        f.write(f'"Processing time (seconds)","{time.time() - start_time:.2f}"\n')
        f.write('\n')
        
        # Write the data
        global_pareto.to_csv(f, index=False)
    
    print(f"\nGlobal Pareto solutions saved to: {output_path}")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    
    # Show sample of Pareto solutions
    print(f"\nSample of global Pareto solutions:")
    sample_5 = global_pareto.head(5)
    display_cols = actual_objectives.copy()
    if 'positions' in global_pareto.columns:
        display_cols.append('positions')
    if 'items' in global_pareto.columns:
        display_cols.append('items')
    
    available_cols = [col for col in display_cols if col in global_pareto.columns]
    print(sample_5[available_cols].to_string(index=False))
    
    return 0


if __name__ == '__main__':
    exit(main())
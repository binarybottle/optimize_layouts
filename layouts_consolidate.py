#!/usr/bin/env python3
"""
Consolidate global Pareto front solutions across multiple MOO result files.

This script processes all MOO results files in output/layouts/ directory,
finds the global Pareto front across all solutions, and outputs a single
CSV file with the globally optimal solutions.

Output Format:
- items: letters in assignment order (e.g., "etaoinsrhldcum") 
- positions: QWERTY positions where those letters go (e.g., "KJ;ASDVRLFUEIM")
- layout_qwerty: layout string in 32-key QWERTY order (QWERTYUIOPASDFGHJKL;ZXCVBNM,./[')
  with spaces for unassigned positions, such as: "  cr  du  oinl  teha   s  m     "

Usage:
    # Basic usage
    python layouts_consolidate.py

    # Test with limited files first  
    python layouts_consolidate.py --max-files 100 --verbose

    # Custom pattern and objectives
    python layouts_consolidate.py --file-pattern "moo_results_config_*.csv" --objectives "engram_key_preference,engram_row_separation,engram_same_row,engram_same_finger"

    # Keyboard layout optimization study commands:
    poetry run python3 layouts_consolidate.py \
        --objectives engram_key_preference engram_avg4_score \
        --maximize true true
    poetry run python3 layouts_consolidate.py \
        --objectives engram_key_preference engram_row_separation engram_same_row engram_same_finger \
        --maximize true true true true

"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import time
import glob

def safe_string_conversion(value, preserve_spaces: bool = False) -> str:
    """Safely convert value to string, preserving apostrophes and avoiding NaN issues."""
    if value == "'":
        return "'"
    
    str_value = str(value)
    if not preserve_spaces:
        str_value = str_value.strip()
    
    if str_value.upper() in ['NAN', 'NA', 'NULL']:
        raise ValueError(f"Detected problematic value conversion: {value} -> {str_value}")
    
    return str_value

def validate_layout_data(items: str, positions: str) -> bool:
    """Validate layout data for problematic values."""
    try:
        # Convert safely
        safe_items = safe_string_conversion(items, preserve_spaces=True)
        safe_positions = safe_string_conversion(positions, preserve_spaces=True)
        
        # Check lengths match
        if len(safe_items) != len(safe_positions):
            return False
        
        # Check for problematic values in mapping
        for item, pos in zip(safe_items, safe_positions):
            safe_string_conversion(item)
            safe_string_conversion(pos)
        
        return True
    except (ValueError, TypeError):
        return False

def parse_individual_moo_file(filepath: str, debug: bool = False) -> pd.DataFrame:
    """
    Parse a single MOO results CSV file with metadata preservation and enhanced validation.
    
    Args:
        filepath: Path to CSV file
        debug: Enable debug output
        
    Returns:
        DataFrame with parsed results and metadata, or None if parsing fails
    """
    try:
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            if debug:
                print(f"File does not exist or is empty: {filepath}")
            return None
        
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required_columns = ['rank', 'items', 'positions']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            if debug:
                print(f"Missing required columns in {filepath}: {missing}")
                print(f"Available columns: {list(df.columns)}")
            return None
        
        # Enhanced validation: Filter out rows with invalid layout data
        valid_rows = []
        invalid_count = 0
        
        for idx, row in df.iterrows():
            items = row.get('items', '')
            positions = row.get('positions', '')
            
            # Skip rows with missing or empty layout data
            if pd.isna(items) or pd.isna(positions) or not items or not positions:
                invalid_count += 1
                if debug:
                    print(f"Skipping row {idx} with missing layout data: items='{items}' positions='{positions}'")
                continue
            
            # Validate layout data structure and content
            if validate_layout_data(items, positions):
                valid_rows.append(idx)
            else:
                invalid_count += 1
                if debug:
                    print(f"Skipping row {idx} with invalid layout data: items='{items}' positions='{positions}'")
        
        if not valid_rows:
            if debug:
                print(f"No valid layout data found in {filepath}")
            return None
        
        if invalid_count > 0:
            if debug:
                print(f"Filtered out {invalid_count} rows with invalid layout data from {filepath}")
            df = df.loc[valid_rows].copy()
        
        # Extract config_id from filename
        filename = os.path.basename(filepath)
        config_id = filename.replace('moo_results_config_', '').replace('moo_results_', '').replace('.csv', '')
        if '_' in config_id:
            config_id = config_id.split('_')[0]  # Take first part before timestamp
        
        # Add metadata columns with safe string conversion
        df['config_id'] = safe_string_conversion(config_id)
        df['source_file'] = safe_string_conversion(filename)
        df['source_rank'] = df['rank']  # Preserve original rank from source file
        
        # Remove the original 'rank' column to avoid confusion with global rankings
        df = df.drop('rank', axis=1)
        
        # Additional validation: ensure items and positions are properly formatted
        def clean_layout_data(row):
            try:
                items = safe_string_conversion(row['items'], preserve_spaces=True)
                positions = safe_string_conversion(row['positions'], preserve_spaces=True)
                row['items'] = items
                row['positions'] = positions
                return row
            except ValueError as e:
                if debug:
                    print(f"Warning: Could not clean layout data in row: {e}")
                return row
        
        df = df.apply(clean_layout_data, axis=1)
        
        if debug:
            print(f"Successfully parsed {filepath}: {len(df)} valid solutions")
            
        return df
        
    except Exception as e:
        if debug:
            print(f"Error parsing {filepath}: {e}")
        return None
    
def convert_items_positions_to_qwerty_layout(items, positions):
    """
    Convert items->positions mapping to QWERTY-ordered layout string.
    
    Args:
        items: Letters in assignment order (e.g., "etaoinsrhldcum")
        positions: QWERTY positions where those letters go (e.g., "KJ;ASDVRLFUEIM")
        
    Returns:
        Layout string in QWERTY key order (e.g., "  cr  du  oinl  teha   s  m     ")
    """
    QWERTY_ORDER = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"
    
    # Create mapping from position to letter
    pos_to_letter = dict(zip(positions, items))
    
    # Build layout string in QWERTY order
    layout_chars = []
    for qwerty_pos in QWERTY_ORDER:
        if qwerty_pos in pos_to_letter:
            layout_chars.append(pos_to_letter[qwerty_pos])
        else:
            layout_chars.append(' ')  # Use space for unassigned positions
    
    return ''.join(layout_chars)

def load_all_solutions(input_dir: str, file_pattern: str = "moo_results_config_*.csv", 
                       max_files: int = None, verbose: bool = False) -> pd.DataFrame:
    """
    Load all solutions from MOO results files in directory.
    
    Args:
        input_dir: Directory containing CSV files
        file_pattern: Pattern to match files
        max_files: Maximum number of files to process
        verbose: Print verbose output
        
    Returns:
        DataFrame with all solutions and source metadata
    """
    # Find all matching files
    pattern_path = f"{input_dir}/{file_pattern}"
    csv_files = glob.glob(pattern_path)
    
    if not csv_files:
        print(f"Error: No CSV files found matching pattern: {pattern_path}")
        return pd.DataFrame()
    
    # Apply max_files limit for testing
    if max_files:
        csv_files = csv_files[:max_files]
        print(f"Limited to first {max_files} files for testing")
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process all files
    all_dataframes = []
    successful_files = 0
    total_solutions = 0
    failed_files = []
    
    start_time = time.time()
    
    for i, csv_file in enumerate(csv_files):
        # Progress reporting
        if verbose or (i > 0 and i % 100 == 0):
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            print(f"Processing file {i+1}/{len(csv_files)} ({rate:.1f} files/sec): {os.path.basename(csv_file)}")
        
        # Parse the file
        df_file = parse_individual_moo_file(csv_file, debug=verbose)
        
        if df_file is not None and len(df_file) > 0:
            all_dataframes.append(df_file)
            successful_files += 1
            total_solutions += len(df_file)
            
            if verbose:
                print(f"  → Loaded {len(df_file)} solutions")
        else:
            failed_files.append(csv_file)
            if verbose:
                print(f"  → Failed to parse")
    
    # Report results
    loading_time = time.time() - start_time
    print(f"\nLoading Summary:")
    print(f"  Successful files: {successful_files}/{len(csv_files)}")
    print(f"  Failed files: {len(failed_files)}")
    print(f"  Total solutions: {total_solutions:,}")
    print(f"  Loading time: {loading_time:.2f} seconds")
    
    if failed_files and verbose:
        print(f"  Failed files: {[os.path.basename(f) for f in failed_files[:5]]}")
        if len(failed_files) > 5:
            print(f"    ... and {len(failed_files) - 5} more")
    
    if not all_dataframes:
        print("Error: No valid solutions found in any file")
        return pd.DataFrame()
    
    # Combine all DataFrames
    print("Combining all solutions...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"Successfully combined {len(combined_df):,} solutions from {successful_files} files")
    
    return combined_df

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

def fast_pareto_filter_numpy(solutions: pd.DataFrame, 
                             objectives: List[str],
                             maximize: List[bool]) -> pd.DataFrame:
    """
    Fast Pareto filtering using numpy operations.
    """
    if len(solutions) == 0:
        return solutions
    
    # Remove NaN values
    solutions_clean = solutions.dropna(subset=objectives).copy()
    if len(solutions_clean) == 0:
        return pd.DataFrame()
    
    print(f"Fast Pareto filtering {len(solutions_clean):,} solutions...")
    
    # Convert objectives to numpy array for fast operations
    obj_matrix = solutions_clean[objectives].values
    
    # Flip signs for minimization objectives
    for i, is_max in enumerate(maximize):
        if not is_max:
            obj_matrix[:, i] = -obj_matrix[:, i]
    
    # Use efficient numpy-based Pareto filtering
    n_solutions = len(obj_matrix)
    is_pareto = np.ones(n_solutions, dtype=bool)

    # For each solution, check if ANY other solution dominates it
    for i in range(n_solutions):
        if not is_pareto[i]:
            continue
        current = obj_matrix[i]
        # Check if current solution is dominated by any other solution
        dominated_by = np.all(obj_matrix >= current, axis=1) & np.any(obj_matrix > current, axis=1)
        if np.any(dominated_by & is_pareto):
            is_pareto[i] = False

        # Progress reporting
        if i % 5000 == 0 and i > 0:
            remaining = np.sum(is_pareto)
            print(f"  Processed {i:,}/{n_solutions:,}, Pareto candidates: {remaining:,}")
    
    # Return Pareto optimal solutions
    pareto_solutions = solutions_clean[is_pareto].reset_index(drop=True)
    print(f"Found {len(pareto_solutions):,} Pareto optimal solutions")
    
    return pareto_solutions

def hierarchical_pareto_filter(solutions: pd.DataFrame,
                              objectives: List[str], 
                              maximize: List[bool],
                              sample_size: int = 50000) -> pd.DataFrame:
    """
    Hierarchical approach for large datasets: filter in stages to reduce computational burden.
    """
    if len(solutions) <= sample_size:
        return fast_pareto_filter_numpy(solutions, objectives, maximize)
    
    print(f"Using hierarchical filtering for {len(solutions):,} solutions...")
    
    # Stage 1: Random sampling to get initial Pareto front
    print("Stage 1: Random sampling...")
    sample_solutions = solutions.sample(n=sample_size, random_state=42)
    initial_pareto = fast_pareto_filter_numpy(sample_solutions, objectives, maximize)
    
    print(f"Initial Pareto front: {len(initial_pareto):,} solutions")
    
    # Stage 2: Process remaining solutions in chunks
    print("Stage 2: Processing remaining solutions...")
    remaining_solutions = solutions.drop(sample_solutions.index)
    
    chunk_size = 25000
    current_pareto = initial_pareto
    
    for i in range(0, len(remaining_solutions), chunk_size):
        chunk = remaining_solutions.iloc[i:i+chunk_size]
        chunk_num = i // chunk_size + 1
        total_chunks = (len(remaining_solutions) + chunk_size - 1) // chunk_size
        
        print(f"  Processing chunk {chunk_num}/{total_chunks}, size: {len(chunk):,}")
        
        # Combine chunk with current Pareto front
        combined = pd.concat([chunk, current_pareto], ignore_index=True)
        
        # Find Pareto front of combined solutions
        current_pareto = fast_pareto_filter_numpy(combined, objectives, maximize)
        
        print(f"  Current Pareto front: {len(current_pareto):,} solutions")
    
    return current_pareto

def smart_objective_filtering(solutions: pd.DataFrame,
                            objectives: List[str],
                            maximize: List[bool],
                            percentile_threshold: float = 0.1) -> pd.DataFrame:
    """
    Pre-filter solutions by keeping only top percentile in each objective.
    """
    print(f"Smart pre-filtering {len(solutions):,} solutions...")
    
    # Keep solutions that are in top percentile for ANY objective
    keep_mask = pd.Series([False] * len(solutions))
    
    for obj, is_max in zip(objectives, maximize):
        if is_max:
            threshold = solutions[obj].quantile(1 - percentile_threshold)
            obj_mask = solutions[obj] >= threshold
        else:
            threshold = solutions[obj].quantile(percentile_threshold)
            obj_mask = solutions[obj] <= threshold
        
        keep_mask |= obj_mask
        print(f"  {obj}: keeping solutions {'≥' if is_max else '≤'} {threshold:.6f}")
    
    filtered_solutions = solutions[keep_mask].reset_index(drop=True)
    
    reduction = len(solutions) / len(filtered_solutions) if len(filtered_solutions) > 0 else float('inf')
    print(f"Pre-filtering: {len(solutions):,} → {len(filtered_solutions):,} solutions ({reduction:.1f}x reduction)")
    
    return filtered_solutions

def optimized_pareto_selection(solutions: pd.DataFrame,
                              objectives: List[str],
                              maximize: List[bool],
                              chunk_size: int = 50000) -> pd.DataFrame:
    """
    Optimized Pareto filtering with smart pre-filtering and hierarchical processing.
    """
    print(f"Starting optimized Pareto filtering for {len(solutions):,} solutions...")
    
    if len(solutions) == 0:
        return solutions
    
    # Step 1: Smart pre-filtering for very large datasets
    #if len(solutions) > 100000:
    #    solutions = smart_objective_filtering(solutions, objectives, maximize, percentile_threshold=0.2)
    
    # Step 2: Use hierarchical approach for large datasets
    if len(solutions) > chunk_size:
        return hierarchical_pareto_filter(solutions, objectives, maximize, sample_size=chunk_size)
    else:
        return fast_pareto_filter_numpy(solutions, objectives, maximize)

def save_pareto_results(pareto_solutions: pd.DataFrame, output_path: str, 
                       processing_stats: Dict) -> None:
    """Save Pareto results with metadata and standardized column ordering."""
    if len(pareto_solutions) == 0:
        print("Warning: No Pareto solutions to save")
        return
    
    # Add layout_qwerty column if items and positions exist
    if 'items' in pareto_solutions.columns and 'positions' in pareto_solutions.columns:
        print("Adding layout_qwerty column...")
        pareto_solutions['layout_qwerty'] = pareto_solutions.apply(
            lambda row: convert_items_positions_to_qwerty_layout(row['items'], row['positions']), 
            axis=1
        )
    
    # Standardized column ordering: layout metadata first, then objectives, then source info
    standard_cols = ['config_id', 'items', 'positions', 'layout_qwerty']
    
    # Identify objective columns (exclude known metadata columns)
    metadata_cols = ['config_id', 'source_rank', 'items', 'positions', 'layout_qwerty', 
                     'source_file', 'layout', 'combined_score']
    objective_cols = [col for col in pareto_solutions.columns if col not in metadata_cols]
    
    # Final standardized column ordering
    ordered_cols = []
    
    # Add standard columns that exist
    for col in standard_cols:
        if col in pareto_solutions.columns:
            ordered_cols.append(col)
    
    # Add source_rank after standard columns
    if 'source_rank' in pareto_solutions.columns:
        ordered_cols.append('source_rank')
    
    # Add objective columns (sorted for consistency)
    ordered_cols.extend(sorted(objective_cols))
    
    # Add remaining metadata columns
    remaining_cols = [col for col in pareto_solutions.columns if col not in ordered_cols]
    ordered_cols.extend(remaining_cols)
    
    # Reorder the DataFrame
    pareto_ordered = pareto_solutions[ordered_cols]
    
    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine file paths
    csv_path = output_path
    info_path = output_path.replace('.csv', '_info.txt')
    
    # Save metadata to separate info file
    print(f"Saving metadata to {info_path}...")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("Global Pareto Optimal Solutions - Processing Information\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Generation Details:\n")
        f.write(f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Processing time: {processing_stats['processing_time']:.2f} seconds\n\n")
        
        f.write("Input Statistics:\n")
        f.write(f"  Total original solutions: {processing_stats['total_solutions']:,}\n")
        f.write(f"  Source files processed: {processing_stats['source_files']}\n\n")
        
        f.write("Output Statistics:\n")
        f.write(f"  Global Pareto solutions: {len(pareto_ordered):,}\n")
        f.write(f"  Reduction factor: {processing_stats['reduction_factor']:.1f}x\n\n")
        
        f.write("Optimization Configuration:\n")
        f.write(f"  Objectives used: {', '.join(processing_stats['objectives'])}\n")
        f.write(f"  Maximizing objectives: {', '.join(map(str, processing_stats['maximize_flags']))}\n\n")
        
        f.write("Data Format Description:\n")
        f.write("  items: Letters in assignment order (arbitrary order)\n")
        f.write("  positions: QWERTY positions where those letters go\n")
        f.write("  layout_qwerty: Layout string in QWERTY key order\n")
        f.write("                 (what letter is at each QWERTY position)\n\n")
        
        f.write("Column Information:\n")
        for i, col in enumerate(ordered_cols, 1):
            col_type = "Layout" if col in standard_cols else \
                      "Objective" if col in objective_cols else "Metadata"
            f.write(f"  {i:2d}. {col} ({col_type})\n")
        
        f.write(f"\nTotal columns: {len(ordered_cols)}\n")
    
    # Save clean CSV data
    print(f"Saving {len(pareto_ordered):,} Pareto solutions to {csv_path}...")
    pareto_ordered.to_csv(csv_path, index=False)
    
    print(f"Successfully saved:")
    print(f"  Data: {csv_path}")
    print(f"  Metadata: {info_path}")


def main():
    parser = argparse.ArgumentParser(description='Select global Pareto optimal layouts')
    parser.add_argument('--input-dir', default='output/layouts/', 
                       help='Directory containing MOO results CSV files')
    parser.add_argument('--output-file', default='output/layouts_consolidate_moo_solutions.csv',
                       help='Output CSV file for global Pareto solutions')
    parser.add_argument('--objectives', nargs='+', 
                       default=['engram_key_preference','row_separation','engram_same_row','engram_same_finger'],
                       help='Objective columns to use for Pareto filtering')
    parser.add_argument('--maximize', nargs='+', type=lambda x: x.lower() == 'true',
                       default=[True, True, True, True],
                       help='Whether to maximize each objective (true/false for each)')
    parser.add_argument('--chunk-size', type=int, default=50000,
                       help='Chunk size for hierarchical processing')
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
    print(f"Chunk size: {args.chunk_size:,}")
    
    # Validate maximize argument length
    if len(args.maximize) != len(args.objectives):
        print(f"Error: Number of maximize flags ({len(args.maximize)}) must match number of objectives ({len(args.objectives)})")
        return 1
    
    start_time = time.time()
    
    # Load all solutions
    print(f"\n=== Loading Solutions ===")
    combined_solutions = load_all_solutions(
        args.input_dir, args.file_pattern, args.max_files, args.verbose)
    
    if combined_solutions.empty:
        print("Error: No solutions loaded")
        return 1
    
    loading_time = time.time() - start_time
    print(f"Loading completed in {loading_time:.2f} seconds")
    
    # Validate objectives exist in data
    print(f"\n=== Validating Objectives ===")
    print(f"Available columns: {list(combined_solutions.columns)}")
    
    actual_objectives = []
    for obj in args.objectives:
        if obj in combined_solutions.columns:
            actual_objectives.append(obj)
        else:
            print(f"Warning: Objective '{obj}' not found in data")
    
    if not actual_objectives:
        print("Error: No valid objectives found in data")
        return 1
    
    # Adjust maximize flags to match actual objectives
    actual_maximize = []
    for i, obj in enumerate(args.objectives):
        if obj in actual_objectives:
            if i < len(args.maximize):
                actual_maximize.append(args.maximize[i])
            else:
                actual_maximize.append(True)  # Default to maximize
    
    print(f"Using objectives: {actual_objectives}")
    print(f"Maximize flags: {actual_maximize}")
    
    # Find global Pareto front
    print(f"\n=== Finding Global Pareto Front ===")
    pareto_start = time.time()
    
    global_pareto = optimized_pareto_selection(
        combined_solutions, actual_objectives, actual_maximize, args.chunk_size)
    
    pareto_time = time.time() - pareto_start
    total_time = time.time() - start_time
    
    # Calculate statistics
    reduction_factor = len(combined_solutions) / len(global_pareto) if len(global_pareto) > 0 else float('inf')
    
    print(f"\n=== Results Summary ===")
    print(f"Original solutions: {len(combined_solutions):,}")
    print(f"Global Pareto solutions: {len(global_pareto):,}")
    print(f"Reduction factor: {reduction_factor:.1f}x")
    print(f"Pareto filtering time: {pareto_time:.2f} seconds")
    print(f"Total processing time: {total_time:.2f} seconds")
    
    if len(global_pareto) == 0:
        print("Error: No Pareto optimal solutions found")
        return 1
    
    # Analyze source distribution
    if 'source_file' in global_pareto.columns:
        source_counts = global_pareto['source_file'].value_counts()
        print(f"\nPareto solutions come from {len(source_counts)} different source files")
        print(f"Max solutions from single file: {source_counts.max()}")
        
        if args.verbose and len(source_counts) <= 20:
            print("Solutions per source file:")
            for file, count in source_counts.items():
                print(f"  {file}: {count}")
        elif len(source_counts) > 20:
            print("Top 10 contributing files:")
            for file, count in source_counts.head(10).items():
                print(f"  {file}: {count}")
    
    # Objective statistics
    print(f"\nObjective ranges in global Pareto set:")
    for obj in actual_objectives:
        values = global_pareto[obj].dropna()
        if len(values) > 0:
            print(f"  {obj}: {values.min():.6f} to {values.max():.6f}")
    
    # Prepare processing stats for output
    processing_stats = {
        "total_solutions": len(combined_solutions),
        "source_files": combined_solutions['source_file'].nunique() if 'source_file' in combined_solutions.columns else 0,
        "reduction_factor": reduction_factor,
        "objectives": actual_objectives,
        "maximize_flags": actual_maximize,
        "processing_time": total_time
    }
    
    # Save results
    print(f"\n=== Saving Results ===")
    save_pareto_results(global_pareto, args.output_file, processing_stats)
    
    # Show sample of results
    print(f"\nSample of global Pareto solutions:")
    sample_cols = ['items', 'positions'] + actual_objectives[:3]  # Limit to first 3 objectives
    available_sample_cols = [col for col in sample_cols if col in global_pareto.columns]
    
    sample_df = global_pareto.head(5)[available_sample_cols]
    print(sample_df.to_string(index=False))
    
    print(f"\n✓ Global Pareto selection complete!")
    print(f"Output saved to: {args.output_file}")
    print(f"Metadata saved to: {args.output_file.replace('.csv', '_info.txt')}")
    
    return 0

if __name__ == '__main__':
    exit(main())
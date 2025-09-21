#!/usr/bin/env python3
"""
Filter MOO keyboard layout results using statistical and composite scoring approaches.

This script provides three focused filtering strategies to reduce large sets of MOO solutions
to manageable, high-quality subsets for further analysis.

FILTERING METHODS:

1. intersection: 
   Finds layouts that perform in the top percentage for ALL objectives simultaneously.
   This is a very selective filter that only keeps layouts excelling across every dimension.
   For example, with --top-percent 10, only layouts in the top 10% of ALL objectives survive.
   Use this when you want layouts that are consistently good at everything.

2. score:
   Computes specified scores (e.g., engram_3key_order, comfort_total) for all layouts using 
   score_layouts.py, then filters to the top percentage by that metric.
   Specify the score type with --score-type (e.g., engram_3key_order, comfort_total).
   Use this when you want to prioritize layouts optimized for a specific metric.

3. intersection_score:
   Two-stage filtering combining intersection with any scoring metric:
   Stage 1: Intersection filtering at a broader percentage (--intersection-percent)
   Stage 2: Computes specified scores for intersection survivors, then filters by 
            top percentage of that score (--score-percent)
   Specify the score type with --score-type (e.g., engram_3key_order, comfort_total).
   Use this for efficient multi-dimensional filtering with a final score-based refinement.

Usage:
    # Intersection filtering (layouts excellent in ALL objectives)
    python layouts_filter.py --input moo_analysis_results.csv --method intersection --top-percent 10
    
    # Score filtering (by any scoring method)
    python layouts_filter.py --input moo_analysis_results.csv --method score --score-type engram_3key_order --top-percent 10
    
    # Combined intersection + any score filtering  
    python layouts_filter.py --input moo_analysis_results.csv --method intersection_score --intersection-percent 15 --score-type engram_3key_order --score-percent 10
    
    # Generate report and visualization
    python layouts_filter.py --input moo_analysis_results.csv --method intersection --top-percent 15 --report --plot

    # Add multiple scores to score-based filtering
    python layouts_filter.py --input moo_analysis_results.csv --method score --score-type engram_3key_order --top-percent 15 --include-scores-in-output "comfort_total,dvorak_effort,qwerty_distance"

    # Example used in study
    poetry run python3 layouts_filter.py --input output/analyze_phase1/layouts_consolidate_plot_filter1/moo_analysis_results.csv --method intersection --top-percent 75 --save-removed --include-scores-in-output "engram_3key_order" --verbose

"""

import pandas as pd
import numpy as np
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Set, Tuple, Dict
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class MOOLayoutFilter:
    def __init__(self, input_file: str, verbose: bool = False, score_layouts_path: str = None):
        self.input_file = input_file
        self.verbose = verbose
        self.score_layouts_path = score_layouts_path or self._find_score_layouts_script()
        self.df = self._load_data()
        self.objective_columns = self._detect_objectives()
        self.filtering_stats = {}  # Track filtering statistics
        self.computed_score_col = None  # Will be set when scores are added
        
    def _find_score_layouts_script(self) -> str:
        """Find the score_layouts.py script in common locations."""
        possible_paths = [
            'score_layouts.py',  # Same directory
            '../score_layouts.py',  # Parent directory
            '../keyboard_layout_scorers/score_layouts.py',  # Common structure
            '../../keyboard_layout_scorers/score_layouts.py',  # Another level up
            './keyboard_layout_scorers/score_layouts.py',  # Subdirectory
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                if self.verbose:
                    print(f"Found score_layouts.py at: {path}")
                return path
        
        # If not found, return a default and let the user specify
        return 'score_layouts.py'
    
    def _load_data(self) -> pd.DataFrame:
        """Load MOO results data."""
        df = pd.read_csv(self.input_file)
        if self.verbose:
            print(f"Loaded {len(df)} layouts from {self.input_file}")
        return df
    
    def _detect_objectives(self) -> List[str]:
        """Detect objective columns (numeric columns that are optimization targets)."""
        # Common objective patterns in MOO keyboard optimization
        potential_objectives = []
        for col in self.df.columns:
            if any(pattern in col.lower() for pattern in [
                'engram_', 'comfort_', 'dvorak', 'distance', 'time', 'efficiency', 'speed'
            ]):
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    potential_objectives.append(col)
        
        # Filter to columns that actually vary (not constant)
        objectives = []
        for col in potential_objectives:
            if self.df[col].nunique() > 1:
                objectives.append(col)
        
        if self.verbose:
            print(f"Detected objective columns: {objectives}")
        
        return objectives
    
    def filter_intersection(self, top_percent: float = 10.0) -> pd.DataFrame:
        """Filter to keep layouts that are in the top percentile for ALL objectives (intersection)."""
        if not self.objective_columns:
            raise ValueError("No objective columns detected")
        
        # Find layouts that are in the top percentile for each objective
        top_layout_sets = []
        thresholds = {}
        per_objective_layouts = {}
        
        for col in self.objective_columns:
            threshold = np.percentile(self.df[col], 100 - top_percent)
            thresholds[col] = threshold
            top_layouts = set(self.df[self.df[col] >= threshold].index)
            top_layout_sets.append(top_layouts)
            per_objective_layouts[col] = top_layouts
            
            if self.verbose:
                print(f"{col}: threshold {threshold:.4f}, {len(top_layouts)} layouts qualify")
        
        # Find intersection (layouts that are in top percentile for ALL objectives)
        if top_layout_sets:
            intersection_layouts = set.intersection(*top_layout_sets)
            union_layouts = set.union(*top_layout_sets)
        else:
            intersection_layouts = set()
            union_layouts = set()
        
        filtered_df = self.df.loc[list(intersection_layouts)].copy()
        
        # Calculate detailed filtering breakdown
        total_layouts = len(self.df)
        filtered_by_thresholds = total_layouts - len(union_layouts)  # Failed individual thresholds
        filtered_by_intersection = len(union_layouts) - len(intersection_layouts)  # Failed intersection requirement
        
        # Store comprehensive filtering statistics
        self.filtering_stats['intersection'] = {
            'method': 'intersection',
            'top_percent': top_percent,
            'thresholds': thresholds,
            'per_objective_counts': {col: len(top_set) for col, top_set in zip(self.objective_columns, top_layout_sets)},
            'per_objective_layouts': per_objective_layouts,
            'union_count': len(union_layouts),
            'intersection_count': len(intersection_layouts),
            'original_count': total_layouts,
            'filtered_by_thresholds': filtered_by_thresholds,
            'filtered_by_intersection': filtered_by_intersection,
            'threshold_retention_rate': len(union_layouts) / total_layouts * 100,
            'intersection_retention_rate': len(intersection_layouts) / total_layouts * 100
        }
        
        if self.verbose:
            print(f"Filtering breakdown:")
            print(f"  Original layouts: {total_layouts}")
            print(f"  Passed individual thresholds (union): {len(union_layouts)} ({len(union_layouts)/total_layouts*100:.1f}%)")
            print(f"  Passed intersection requirement: {len(intersection_layouts)} ({len(intersection_layouts)/total_layouts*100:.1f}%)")
            print(f"  Filtered by thresholds: {filtered_by_thresholds} ({filtered_by_thresholds/total_layouts*100:.1f}%)")
            print(f"  Filtered by intersection: {filtered_by_intersection} ({filtered_by_intersection/total_layouts*100:.1f}%)")
        
        # Warning for overly restrictive filtering
        if len(intersection_layouts) < 10:
            print(f"\n⚠️  WARNING: Only {len(intersection_layouts)} layouts survived intersection filtering!")
            print(f"    With {len(self.objective_columns)} objectives and {top_percent}% threshold,")
            expected_rate = (top_percent/100) ** len(self.objective_columns) * 100
            print(f"    mathematical expectation: ~{expected_rate:.2f}% retention rate")
            print(f"    actual retention: {len(intersection_layouts)/total_layouts*100:.2f}%")
            if len(intersection_layouts) == 0:
                print(f"    This means NO layouts are simultaneously in top {top_percent}% of ALL objectives")
            else:
                print(f"    This suggests layouts with identical/near-identical objective values")
            print(f"    Consider using a higher --top-percent (e.g., 70-90%) or 'score' method instead.")
            
        if len(intersection_layouts) <= 1:
            print(f"\n⚠️  ERROR: Intersection filtering kept ≤1 layout - filtering too restrictive!")
            print(f"    Intersection filtering requires layouts to excel in ALL objectives simultaneously.")
            print(f"    Suggested fixes:")
            print(f"    1. Use 'score' method: --method score --score-type engram_3key_order --top-percent 10")
            print(f"    2. Use much higher threshold: --method intersection --top-percent 80")
            print(f"    3. Use 'intersection_score': broader intersection + score refinement")
        
        return filtered_df
    
    def add_scores(self, score_type: str, use_poetry: bool = True) -> pd.DataFrame:
        """Add specified scores using score_layouts.py."""
        if 'items' not in self.df.columns:
            raise ValueError("Need 'items' column for scoring")
        
        # Create temporary layouts for scoring
        temp_layouts = []
        for idx, row in self.df.iterrows():
            layout_name = f"layout_{idx}"
            items = str(row['items'])
            temp_layouts.append(f"{layout_name}:{items}")
        
        if self.verbose:
            print(f"Computing {score_type} scores for {len(temp_layouts)} layouts...")
            print(f"Using score_layouts.py at: {self.score_layouts_path}")
        
        # Create temporary files for layouts and CSV output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as layouts_file:
            layouts_file_path = layouts_file.name
            for layout in temp_layouts:
                layouts_file.write(layout + '\n')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_csv_path = temp_file.name
        
        try:
            # Determine working directory for score_layouts.py
            score_layouts_path = Path(self.score_layouts_path)
            if score_layouts_path.is_absolute():
                work_dir = score_layouts_path.parent
                script_name = score_layouts_path.name
            else:
                work_dir = Path.cwd() / score_layouts_path.parent
                script_name = score_layouts_path.name
            
            # Try file input first, fall back to chunking if not supported
            if use_poetry:
                cmd = [
                    'poetry', 'run', 'python3', script_name,
                    '--compare-file', layouts_file_path,
                    '--scorers', score_type,
                    '--csv', temp_csv_path,
                    '--quiet'
                ]
            else:
                cmd = [
                    'python', script_name,
                    '--compare-file', layouts_file_path,
                    '--scorers', score_type,
                    '--csv', temp_csv_path,
                    '--quiet'
                ]
            
            # Add fallback score table paths if needed
            potential_score_tables = [
                work_dir / 'tables' / 'scores_2key_detailed.csv',
                Path.cwd() / 'tables' / 'scores_2key_detailed.csv',
                Path.cwd() / '..' / 'keyboard_layout_scorers' / 'tables' / 'scores_2key_detailed.csv'
            ]
            
            score_table_found = None
            for score_table_path in potential_score_tables:
                if score_table_path.exists():
                    score_table_found = score_table_path
                    break
            
            if score_table_found:
                cmd.extend(['--score-table', str(score_table_found)])
                if self.verbose:
                    print(f"Using score table: {score_table_found}")
            
            # Add potential frequency file paths
            potential_freq_files = [
                work_dir / 'input' / 'english-letter-pair-frequencies-google-ngrams.csv',
                Path.cwd() / 'input' / 'english-letter-pair-frequencies-google-ngrams.csv',
                Path.cwd() / '..' / 'keyboard_layout_scorers' / 'input' / 'english-letter-pair-frequencies-google-ngrams.csv'
            ]
            
            freq_file_found = None
            for freq_file_path in potential_freq_files:
                if freq_file_path.exists():
                    freq_file_found = freq_file_path
                    break
            
            if freq_file_found:
                cmd.extend(['--frequency-file', str(freq_file_found)])
                if self.verbose:
                    print(f"Using frequency file: {freq_file_found}")
            
            if self.verbose:
                print(f"Working directory: {work_dir}")
                print("Running command:", ' '.join(cmd[:8]) + "..." if len(cmd) > 8 else ' '.join(cmd))
            
            # Run the command with the correct working directory
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
            
            if result.returncode != 0:
                # If file input failed, try chunking approach
                if "--compare-file" in cmd:
                    if self.verbose:
                        print("File input not supported, trying chunking approach...")
                    return self._add_scores_chunked(temp_layouts, temp_csv_path, score_type, use_poetry)
                
                print(f"Command failed with return code {result.returncode}")
                print(f"Working directory: {work_dir}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                
                # Try without poetry if that was the issue
                if use_poetry:
                    if self.verbose:
                        print("Retrying without poetry...")
                    return self.add_scores(score_type, use_poetry=False)
                else:
                    # Try with fallback approach as a last resort
                    if self.verbose:
                        print("Trying with fallback approach...")
                    return self._add_scores_fallback(temp_layouts, temp_csv_path, score_type)
            
            # Load the scores
            if not Path(temp_csv_path).exists():
                raise RuntimeError(f"Output CSV file not created: {temp_csv_path}")
            
            scores_df = pd.read_csv(temp_csv_path)
            
            if self.verbose:
                print(f"Successfully loaded scores from {temp_csv_path}")
                print(f"Scores dataframe shape: {scores_df.shape}")
                print(f"Columns: {list(scores_df.columns)}")
            
            # Add scores to original dataframe
            enhanced_df = self.df.copy()
            
            if score_type in scores_df.columns:
                enhanced_df[score_type] = scores_df[score_type].values
                self.computed_score_col = score_type
                
                if self.verbose:
                    score_range = (scores_df[score_type].min(), scores_df[score_type].max())
                    print(f"Added {score_type} scores (range: {score_range[0]:.4f} - {score_range[1]:.4f})")
            else:
                raise ValueError(f"{score_type} column not found in scoring results. Available columns: {list(scores_df.columns)}")
            
            return enhanced_df
            
        finally:
            # Clean up temp files
            Path(temp_csv_path).unlink(missing_ok=True)
            Path(layouts_file_path).unlink(missing_ok=True)
    
    def _add_multiple_scores_chunked(self, df: pd.DataFrame, score_types: List[str], temp_layouts: List[str], temp_csv_path: str, use_poetry: bool = True) -> pd.DataFrame:
        """Add multiple scores using chunking approach to avoid argument list length limits."""
        if self.verbose:
            print("Using chunking approach for multiple scores...")
        
        # Determine working directory for score_layouts.py
        score_layouts_path = Path(self.score_layouts_path)
        if score_layouts_path.is_absolute():
            work_dir = score_layouts_path.parent
            script_name = score_layouts_path.name
        else:
            work_dir = Path.cwd() / score_layouts_path.parent
            script_name = score_layouts_path.name
        
        # Chunk layouts into smaller batches
        chunk_size = 50  # Start conservative
        all_results = []
        scorers_str = ','.join(score_types)
        
        for i in range(0, len(temp_layouts), chunk_size):
            chunk = temp_layouts[i:i+chunk_size]
            
            # Create temporary CSV for this chunk
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as chunk_file:
                chunk_csv_path = chunk_file.name
            
            try:
                # Build command for this chunk
                if use_poetry:
                    cmd = [
                        'poetry', 'run', 'python3', script_name,
                        '--compare'] + chunk + [
                        '--scorers', scorers_str,
                        '--csv', chunk_csv_path,
                        '--quiet'
                    ]
                else:
                    cmd = [
                        'python', script_name,
                        '--compare'] + chunk + [
                        '--scorers', scorers_str,
                        '--csv', chunk_csv_path,
                        '--quiet'
                    ]
                
                # Add fallback paths for dependencies
                potential_score_tables = [
                    work_dir / 'tables' / 'scores_2key_detailed.csv',
                    Path.cwd() / 'tables' / 'scores_2key_detailed.csv',
                    Path.cwd() / '..' / 'keyboard_layout_scorers' / 'tables' / 'scores_2key_detailed.csv'
                ]
                
                score_table_found = None
                for score_table_path in potential_score_tables:
                    if score_table_path.exists():
                        score_table_found = score_table_path
                        break
                
                if score_table_found:
                    cmd.extend(['--score-table', str(score_table_found)])
                
                potential_freq_files = [
                    work_dir / 'input' / 'english-letter-pair-frequencies-google-ngrams.csv',
                    Path.cwd() / 'input' / 'english-letter-pair-frequencies-google-ngrams.csv',
                    Path.cwd() / '..' / 'keyboard_layout_scorers' / 'input' / 'english-letter-pair-frequencies-google-ngrams.csv'
                ]
                
                freq_file_found = None
                for freq_file_path in potential_freq_files:
                    if freq_file_path.exists():
                        freq_file_found = freq_file_path
                        break
                
                if freq_file_found:
                    cmd.extend(['--frequency-file', str(freq_file_found)])
                
                if self.verbose:
                    print(f"Processing chunk {i//chunk_size + 1}/{(len(temp_layouts) + chunk_size - 1)//chunk_size} ({len(chunk)} layouts)")
                
                # Run the command
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
                
                if result.returncode != 0:
                    # If still getting argument list error, reduce chunk size
                    if "Argument list too long" in result.stderr:
                        if chunk_size > 10:
                            chunk_size = max(10, chunk_size // 2)
                            if self.verbose:
                                print(f"Reducing chunk size to {chunk_size}")
                            # Retry this chunk with smaller size
                            continue
                        else:
                            raise RuntimeError(f"Cannot reduce chunk size further: {result.stderr}")
                    else:
                        raise RuntimeError(f"Chunk processing failed: {result.stderr}")
                
                # Load results for this chunk
                if Path(chunk_csv_path).exists():
                    chunk_df = pd.read_csv(chunk_csv_path)
                    all_results.append(chunk_df)
                    
            finally:
                # Clean up chunk CSV
                Path(chunk_csv_path).unlink(missing_ok=True)
        
        # Combine all chunk results
        if all_results:
            combined_scores = pd.concat(all_results, ignore_index=True)
            combined_scores.to_csv(temp_csv_path, index=False)
            
            if self.verbose:
                print(f"Successfully combined {len(all_results)} chunks for multiple scores")
        else:
            print("Warning: No successful chunks processed for multiple scores")
            return df
        
        # Continue with normal processing
        scores_df = pd.read_csv(temp_csv_path)
        enhanced_df = df.copy()
        
        # Add each score type
        added_scores = []
        for score_type in score_types:
            if score_type in scores_df.columns:
                enhanced_df[score_type] = scores_df[score_type].values
                added_scores.append(score_type)
                if self.verbose:
                    score_range = (scores_df[score_type].min(), scores_df[score_type].max())
                    print(f"  Added {score_type}: {score_range[0]:.4f} - {score_range[1]:.4f}")
            else:
                print(f"Warning: {score_type} not found in scoring results")
        
        if added_scores:
            print(f"Successfully added {len(added_scores)} additional scores: {', '.join(added_scores)}")
        else:
            print("Warning: No additional scores were successfully added")
            
        return enhanced_df
    
    def _add_scores_chunked(self, temp_layouts: List[str], temp_csv_path: str, score_type: str, use_poetry: bool = True) -> pd.DataFrame:
        """Add scores using chunking approach to avoid argument list length limits."""
        if self.verbose:
            print("Using chunking approach to avoid argument list limits...")
        
        # Determine working directory for score_layouts.py
        score_layouts_path = Path(self.score_layouts_path)
        if score_layouts_path.is_absolute():
            work_dir = score_layouts_path.parent
            script_name = score_layouts_path.name
        else:
            work_dir = Path.cwd() / score_layouts_path.parent
            script_name = score_layouts_path.name
        
        # Chunk layouts into smaller batches (adjust chunk size as needed)
        chunk_size = 50  # Start conservative
        all_results = []
        
        for i in range(0, len(temp_layouts), chunk_size):
            chunk = temp_layouts[i:i+chunk_size]
            
            # Create temporary CSV for this chunk
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as chunk_file:
                chunk_csv_path = chunk_file.name
            
            try:
                # Build command for this chunk
                if use_poetry:
                    cmd = [
                        'poetry', 'run', 'python3', script_name,
                        '--compare'] + chunk + [
                        '--scorers', score_type,
                        '--csv', chunk_csv_path,
                        '--quiet'
                    ]
                else:
                    cmd = [
                        'python', script_name,
                        '--compare'] + chunk + [
                        '--scorers', score_type,
                        '--csv', chunk_csv_path,
                        '--quiet'
                    ]
                
                # Add fallback paths for dependencies
                potential_score_tables = [
                    work_dir / 'tables' / 'scores_2key_detailed.csv',
                    Path.cwd() / 'tables' / 'scores_2key_detailed.csv',
                    Path.cwd() / '..' / 'keyboard_layout_scorers' / 'tables' / 'scores_2key_detailed.csv'
                ]
                
                score_table_found = None
                for score_table_path in potential_score_tables:
                    if score_table_path.exists():
                        score_table_found = score_table_path
                        break
                
                if score_table_found:
                    cmd.extend(['--score-table', str(score_table_found)])
                
                potential_freq_files = [
                    work_dir / 'input' / 'english-letter-pair-frequencies-google-ngrams.csv',
                    Path.cwd() / 'input' / 'english-letter-pair-frequencies-google-ngrams.csv',
                    Path.cwd() / '..' / 'keyboard_layout_scorers' / 'input' / 'english-letter-pair-frequencies-google-ngrams.csv'
                ]
                
                freq_file_found = None
                for freq_file_path in potential_freq_files:
                    if freq_file_path.exists():
                        freq_file_found = freq_file_path
                        break
                
                if freq_file_found:
                    cmd.extend(['--frequency-file', str(freq_file_found)])
                
                if self.verbose:
                    print(f"Processing chunk {i//chunk_size + 1}/{(len(temp_layouts) + chunk_size - 1)//chunk_size} ({len(chunk)} layouts)")
                
                # Run the command
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
                
                if result.returncode != 0:
                    # If still getting argument list error, reduce chunk size
                    if "Argument list too long" in result.stderr:
                        if chunk_size > 10:
                            chunk_size = max(10, chunk_size // 2)
                            if self.verbose:
                                print(f"Reducing chunk size to {chunk_size}")
                            # Retry this chunk with smaller size
                            continue
                        else:
                            raise RuntimeError(f"Cannot reduce chunk size further: {result.stderr}")
                    else:
                        raise RuntimeError(f"Chunk processing failed: {result.stderr}")
                
                # Load results for this chunk
                if Path(chunk_csv_path).exists():
                    chunk_df = pd.read_csv(chunk_csv_path)
                    all_results.append(chunk_df)
                    
            finally:
                # Clean up chunk CSV
                Path(chunk_csv_path).unlink(missing_ok=True)
        
        # Combine all chunk results
        if all_results:
            combined_scores = pd.concat(all_results, ignore_index=True)
            combined_scores.to_csv(temp_csv_path, index=False)
            
            if self.verbose:
                print(f"Successfully combined {len(all_results)} chunks")
        else:
            raise RuntimeError("No successful chunks processed")
        
        # Continue with normal processing
        scores_df = pd.read_csv(temp_csv_path)
        enhanced_df = self.df.copy()
        
        if score_type in scores_df.columns:
            enhanced_df[score_type] = scores_df[score_type].values
            self.computed_score_col = score_type
            
            if self.verbose:
                score_range = (scores_df[score_type].min(), scores_df[score_type].max())
                print(f"Added {score_type} scores (range: {score_range[0]:.4f} - {score_range[1]:.4f})")
        else:
            raise ValueError(f"{score_type} column not found in scoring results. Available columns: {list(scores_df.columns)}")
        
        return enhanced_df
    
    def add_multiple_scores(self, df: pd.DataFrame, score_types: List[str], use_poetry: bool = True) -> pd.DataFrame:
        """Add multiple score types to a dataframe."""
        if not score_types:
            return df
            
        if 'items' not in df.columns:
            raise ValueError("Need 'items' column for scoring")
        
        print(f"Computing additional scores: {', '.join(score_types)}")
        
        # Create temporary layouts for scoring
        temp_layouts = []
        for idx, row in df.iterrows():
            layout_name = f"layout_{idx}"
            items = str(row['items'])
            temp_layouts.append(f"{layout_name}:{items}")
        
        # Create temporary files for layouts and CSV output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as layouts_file:
            layouts_file_path = layouts_file.name
            for layout in temp_layouts:
                layouts_file.write(layout + '\n')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_csv_path = temp_file.name
        
        try:
            # Determine working directory for score_layouts.py
            score_layouts_path = Path(self.score_layouts_path)
            if score_layouts_path.is_absolute():
                work_dir = score_layouts_path.parent
                script_name = score_layouts_path.name
            else:
                work_dir = Path.cwd() / score_layouts_path.parent
                script_name = score_layouts_path.name
            
            # Build command with all score types, try file input first
            scorers_str = ','.join(score_types)
            
            if use_poetry:
                cmd = [
                    'poetry', 'run', 'python3', script_name,
                    '--compare-file', layouts_file_path,
                    '--scorers', scorers_str,
                    '--csv', temp_csv_path,
                    '--quiet'
                ]
            else:
                cmd = [
                    'python', script_name,
                    '--compare-file', layouts_file_path,
                    '--scorers', scorers_str,
                    '--csv', temp_csv_path,
                    '--quiet'
                ]
            
            # Add fallback paths for dependencies
            potential_score_tables = [
                work_dir / 'tables' / 'scores_2key_detailed.csv',
                Path.cwd() / 'tables' / 'scores_2key_detailed.csv',
                Path.cwd() / '..' / 'keyboard_layout_scorers' / 'tables' / 'scores_2key_detailed.csv'
            ]
            
            score_table_found = None
            for score_table_path in potential_score_tables:
                if score_table_path.exists():
                    score_table_found = score_table_path
                    break
            
            if score_table_found:
                cmd.extend(['--score-table', str(score_table_found)])
            
            potential_freq_files = [
                work_dir / 'input' / 'english-letter-pair-frequencies-google-ngrams.csv',
                Path.cwd() / 'input' / 'english-letter-pair-frequencies-google-ngrams.csv',
                Path.cwd() / '..' / 'keyboard_layout_scorers' / 'input' / 'english-letter-pair-frequencies-google-ngrams.csv'
            ]
            
            freq_file_found = None
            for freq_file_path in potential_freq_files:
                if freq_file_path.exists():
                    freq_file_found = freq_file_path
                    break
            
            if freq_file_found:
                cmd.extend(['--frequency-file', str(freq_file_found)])
            
            if self.verbose:
                print(f"Working directory: {work_dir}")
                print("Running scoring command for additional metrics...")
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
            
            if result.returncode != 0:
                # If file input failed, try chunking approach
                if "--compare-file" in cmd:
                    if self.verbose:
                        print("File input not supported, trying chunking approach...")
                    return self._add_multiple_scores_chunked(df, score_types, temp_layouts, temp_csv_path, use_poetry)
                
                if use_poetry and "poetry" in result.stderr.lower():
                    if self.verbose:
                        print("Retrying without poetry...")
                    return self.add_multiple_scores(df, score_types, use_poetry=False)
                else:
                    print(f"Warning: Failed to compute additional scores: {result.stderr}")
                    return df
            
            # Load and merge the scores
            if not Path(temp_csv_path).exists():
                print(f"Warning: Score output file not created")
                return df
            
            scores_df = pd.read_csv(temp_csv_path)
            enhanced_df = df.copy()
            
            # Add each score type
            added_scores = []
            for score_type in score_types:
                if score_type in scores_df.columns:
                    enhanced_df[score_type] = scores_df[score_type].values
                    added_scores.append(score_type)
                    if self.verbose:
                        score_range = (scores_df[score_type].min(), scores_df[score_type].max())
                        print(f"  Added {score_type}: {score_range[0]:.4f} - {score_range[1]:.4f}")
                else:
                    print(f"Warning: {score_type} not found in scoring results")
            
            if added_scores:
                print(f"Successfully added {len(added_scores)} additional scores: {', '.join(added_scores)}")
            else:
                print("Warning: No additional scores were successfully added")
                
            return enhanced_df
            
        finally:
            # Clean up temp files
            Path(temp_csv_path).unlink(missing_ok=True)
            Path(layouts_file_path).unlink(missing_ok=True)
    
    def _add_scores_fallback(self, temp_layouts: List[str], temp_csv_path: str, score_type: str) -> pd.DataFrame:
        """Fallback method to add scores with absolute paths."""
        try:
            # Try to find required files in common locations
            current_dir = Path.cwd()
            parent_dir = current_dir.parent
            
            # Look for the keyboard_layout_scorers directory
            scorer_dirs = [
                current_dir / 'keyboard_layout_scorers',
                parent_dir / 'keyboard_layout_scorers',
                current_dir / '..' / 'keyboard_layout_scorers',
            ]
            
            scorer_dir = None
            for d in scorer_dirs:
                if d.exists() and (d / 'score_layouts.py').exists():
                    scorer_dir = d.resolve()
                    break
            
            if not scorer_dir:
                raise RuntimeError("Could not locate keyboard_layout_scorers directory")
            
            # Build command with absolute paths
            cmd = [
                'python', str(scorer_dir / 'score_layouts.py'),
                '--compare'] + temp_layouts + [
                '--scorers', score_type,
                '--csv', temp_csv_path,
                '--quiet'
            ]
            
            # Add absolute paths for required files
            score_table = scorer_dir / 'tables' / 'scores_2key_detailed.csv'
            if score_table.exists():
                cmd.extend(['--score-table', str(score_table)])
            
            freq_file = scorer_dir / 'input' / 'english-letter-pair-frequencies-google-ngrams.csv'
            if freq_file.exists():
                cmd.extend(['--frequency-file', str(freq_file)])
            
            if self.verbose:
                print(f"Fallback: Using scorer directory: {scorer_dir}")
                print("Fallback command:", ' '.join(cmd[:8]) + "..." if len(cmd) > 8 else ' '.join(cmd))
            
            # Run with the scorer directory as working directory
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=scorer_dir)
            
            if result.returncode != 0:
                raise RuntimeError(f"Fallback failed: {result.stderr}\nSTDOUT: {result.stdout}")
            
            # Load the scores
            scores_df = pd.read_csv(temp_csv_path)
            enhanced_df = self.df.copy()
            
            if score_type in scores_df.columns:
                enhanced_df[score_type] = scores_df[score_type].values
                self.computed_score_col = score_type
                
                if self.verbose:
                    score_range = (scores_df[score_type].min(), scores_df[score_type].max())
                    print(f"Fallback success: Added {score_type} scores (range: {score_range[0]:.4f} - {score_range[1]:.4f})")
            else:
                raise ValueError(f"{score_type} column not found in scoring results. Available columns: {list(scores_df.columns)}")
            
            return enhanced_df
            
        except Exception as e:
            raise RuntimeError(f"All scoring attempts failed. Last error: {e}")
    
    def filter_by_score(self, df: pd.DataFrame, score_type: str, top_percent: float = 10.0) -> pd.DataFrame:
        """Filter by top percentile of specified scores."""
        if score_type not in df.columns:
            raise ValueError(f"Score column '{score_type}' not available in dataframe. Available columns: {list(df.columns)}")
        
        threshold = np.percentile(df[score_type], 100 - top_percent)
        filtered_df = df[df[score_type] >= threshold].copy()
        
        # Store filtering statistics
        self.filtering_stats['score'] = {
            'method': 'score',
            'score_type': score_type,
            'top_percent': top_percent,
            'threshold': threshold,
            'filtered_count': len(filtered_df),
            'input_count': len(df),
            'retention_rate': len(filtered_df) / len(df) * 100
        }
        
        if self.verbose:
            print(f"Score filtering ({score_type}): {len(df)} -> {len(filtered_df)} layouts ({len(filtered_df)/len(df)*100:.1f}%)")
            print(f"Threshold: {threshold:.4f}")
        
        return filtered_df
    
    def filter_intersection_score(self, intersection_percent: float = 15.0, 
                                 score_type: str = 'engram_3key_order',
                                 score_percent: float = 10.0) -> pd.DataFrame:
        """Combined intersection + score filtering."""
        
        if self.verbose:
            print(f"=== Combined Intersection + {score_type} Score Filtering ===")
        
        # Stage 1: Intersection filtering
        stage1_df = self.filter_intersection(intersection_percent)
        
        # Stage 2: Add scores to intersection results only
        intersection_enhanced = self.add_scores_subset(stage1_df, score_type)
        
        # Stage 3: Apply score filtering to intersection results
        final_df = self.filter_by_score(intersection_enhanced, score_type, score_percent)
        
        # Update combined statistics
        self.filtering_stats['combined'] = {
            'method': 'intersection_score',
            'intersection_percent': intersection_percent,
            'score_type': score_type,
            'score_percent': score_percent,
            'stage1_count': len(stage1_df),
            'final_count': len(final_df),
            'original_count': len(self.df),
            'overall_retention_rate': len(final_df) / len(self.df) * 100
        }
        
        if self.verbose:
            print(f"Combined filtering: {len(self.df)} -> {len(stage1_df)} -> {len(final_df)} layouts")
            print(f"Overall retention: {len(final_df)/len(self.df)*100:.1f}%")
        
        return final_df
    
    def add_scores_subset(self, subset_df: pd.DataFrame, score_type: str) -> pd.DataFrame:
        """Add scores to a subset of layouts."""
        if 'items' not in subset_df.columns:
            raise ValueError("Need 'items' column for scoring")
        
        # Create temporary layouts for scoring (only for subset)
        temp_layouts = []
        for idx, row in subset_df.iterrows():
            layout_name = f"layout_{idx}"
            items = str(row['items'])
            temp_layouts.append(f"{layout_name}:{items}")
        
        if self.verbose:
            print(f"Computing {score_type} scores for {len(temp_layouts)} filtered layouts...")
            print(f"Using score_layouts.py at: {self.score_layouts_path}")
        
        # Create temporary CSV output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_csv_path = temp_file.name
        
        try:
            # Use the same logic as add_scores but for subset
            return self._compute_scores(temp_layouts, temp_csv_path, subset_df, score_type)
            
        finally:
            # Clean up temp file
            Path(temp_csv_path).unlink(missing_ok=True)
    
    def _compute_scores(self, temp_layouts: List[str], temp_csv_path: str, 
                       target_df: pd.DataFrame, score_type: str, use_poetry: bool = True) -> pd.DataFrame:
        """Helper method to compute scores for given layouts."""
        
        # Determine working directory for score_layouts.py
        score_layouts_path = Path(self.score_layouts_path)
        if score_layouts_path.is_absolute():
            work_dir = score_layouts_path.parent
            script_name = score_layouts_path.name
        else:
            work_dir = Path.cwd() / score_layouts_path.parent
            script_name = score_layouts_path.name
        
        # Build command based on whether to use poetry
        if use_poetry:
            cmd = [
                'poetry', 'run', 'python3', script_name,
                '--compare'] + temp_layouts + [
                '--scorers', score_type,
                '--csv', temp_csv_path,
                '--quiet'
            ]
        else:
            cmd = [
                'python', script_name,
                '--compare'] + temp_layouts + [
                '--scorers', score_type,
                '--csv', temp_csv_path,
                '--quiet'
            ]
        
        # Add fallback score table paths if needed
        potential_score_tables = [
            work_dir / 'tables' / 'scores_2key_detailed.csv',
            Path.cwd() / 'tables' / 'scores_2key_detailed.csv',
            Path.cwd() / '..' / 'keyboard_layout_scorers' / 'tables' / 'scores_2key_detailed.csv'
        ]
        
        score_table_found = None
        for score_table_path in potential_score_tables:
            if score_table_path.exists():
                score_table_found = score_table_path
                break
        
        if score_table_found:
            cmd.extend(['--score-table', str(score_table_found)])
            if self.verbose:
                print(f"Using score table: {score_table_found}")
        
        # Add potential frequency file paths
        potential_freq_files = [
            work_dir / 'input' / 'english-letter-pair-frequencies-google-ngrams.csv',
            Path.cwd() / 'input' / 'english-letter-pair-frequencies-google-ngrams.csv',
            Path.cwd() / '..' / 'keyboard_layout_scorers' / 'input' / 'english-letter-pair-frequencies-google-ngrams.csv'
        ]
        
        freq_file_found = None
        for freq_file_path in potential_freq_files:
            if freq_file_path.exists():
                freq_file_found = freq_file_path
                break
        
        if freq_file_found:
            cmd.extend(['--frequency-file', str(freq_file_found)])
            if self.verbose:
                print(f"Using frequency file: {freq_file_found}")
        
        if self.verbose:
            print(f"Working directory: {work_dir}")
            print("Running command:", ' '.join(cmd[:8]) + "..." if len(cmd) > 8 else ' '.join(cmd))
        
        # Run the command with the correct working directory
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
        
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            print(f"Working directory: {work_dir}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            
            # Try without poetry if that was the issue
            if use_poetry:
                if self.verbose:
                    print("Retrying without poetry...")
                return self._compute_scores(temp_layouts, temp_csv_path, target_df, score_type, use_poetry=False)
            else:
                raise RuntimeError(f"score_layouts.py failed: {result.stderr}\nSTDOUT: {result.stdout}")
        
        # Load the scores
        if not Path(temp_csv_path).exists():
            raise RuntimeError(f"Output CSV file not created: {temp_csv_path}")
        
        scores_df = pd.read_csv(temp_csv_path)
        
        if self.verbose:
            print(f"Successfully loaded scores from {temp_csv_path}")
            print(f"Scores dataframe shape: {scores_df.shape}")
            print(f"Columns: {list(scores_df.columns)}")
        
        # Add scores to target dataframe
        enhanced_df = target_df.copy()
        
        if score_type in scores_df.columns:
            enhanced_df[score_type] = scores_df[score_type].values
            self.computed_score_col = score_type
            
            if self.verbose:
                score_range = (scores_df[score_type].min(), scores_df[score_type].max())
                print(f"Added {score_type} scores (range: {score_range[0]:.4f} - {score_range[1]:.4f})")
        else:
            raise ValueError(f"{score_type} column not found in scoring results. Available columns: {list(scores_df.columns)}")
        
        return enhanced_df
    
    def generate_statistical_report(self, filtered_df: pd.DataFrame, output_dir: str = '.') -> str:
        """Generate a statistical report comparing filtered vs original datasets."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = Path(output_dir) / f'filtering_report_{timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write("MOO Layout Filtering Statistical Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {self.input_file}\n")
            f.write(f"Filtering method: {getattr(self, 'filtering_stats', {}).get('intersection', {}).get('method', 'unknown')}\n\n")
            
            # Dataset sizes
            f.write("Dataset Sizes:\n")
            f.write(f"Original: {len(self.df)} layouts\n")
            f.write(f"Filtered: {len(filtered_df)} layouts\n")
            f.write(f"Retention rate: {len(filtered_df)/len(self.df)*100:.2f}%\n\n")
            
            # Filtering breakdown if available
            if 'intersection' in self.filtering_stats:
                stats = self.filtering_stats['intersection']
                f.write("Filtering Breakdown:\n")
                f.write("-" * 25 + "\n")
                f.write(f"Original layouts: {stats['original_count']}\n")
                f.write(f"Passed individual thresholds (union): {stats['union_count']} ({stats['threshold_retention_rate']:.1f}%)\n")
                f.write(f"Passed intersection requirement: {stats['intersection_count']} ({stats['intersection_retention_rate']:.1f}%)\n\n")
                
                f.write(f"Filtered by individual thresholds: {stats['filtered_by_thresholds']} ({stats['filtered_by_thresholds']/stats['original_count']*100:.1f}%)\n")
                f.write(f"Filtered by intersection requirement: {stats['filtered_by_intersection']} ({stats['filtered_by_intersection']/stats['original_count']*100:.1f}%)\n\n")
                
                f.write("Per-objective threshold analysis:\n")
                for col in self.objective_columns:
                    if col in stats['per_objective_counts']:
                        count = stats['per_objective_counts'][col]
                        threshold = stats['thresholds'][col]
                        f.write(f"  {col}: {count} layouts above {threshold:.4f} ({count/stats['original_count']*100:.1f}%)\n")
                f.write("\n")
            
            # Objective statistics comparison
            f.write("Objective Statistics Comparison:\n")
            f.write("-" * 40 + "\n")
            
            for col in self.objective_columns:
                if col in filtered_df.columns:
                    orig_stats = self.df[col].describe()
                    filt_stats = filtered_df[col].describe()
                    
                    f.write(f"\n{col}:\n")
                    f.write(f"  Original - Mean: {orig_stats['mean']:.4f}, Std: {orig_stats['std']:.4f}\n")
                    f.write(f"  Original - Min: {orig_stats['min']:.4f}, Max: {orig_stats['max']:.4f}\n")
                    f.write(f"  Filtered - Mean: {filt_stats['mean']:.4f}, Std: {filt_stats['std']:.4f}\n")
                    f.write(f"  Filtered - Min: {filt_stats['min']:.4f}, Max: {filt_stats['max']:.4f}\n")
                    
                    improvement = (filt_stats['mean'] - orig_stats['mean']) / orig_stats['mean'] * 100
                    f.write(f"  Mean improvement: {improvement:+.2f}%\n")
                    
                    # Show max value loss
                    max_loss = orig_stats['max'] - filt_stats['max']
                    max_loss_pct = max_loss / orig_stats['max'] * 100
                    f.write(f"  Maximum value lost: {max_loss:.4f} ({max_loss_pct:.1f}%)\n")
            
            # Filtering statistics
            if hasattr(self, 'filtering_stats') and self.filtering_stats:
                f.write(f"\nDetailed Filtering Statistics:\n")
                f.write("-" * 35 + "\n")
                for key, stats in self.filtering_stats.items():
                    f.write(f"{key}: {stats}\n")
        
        return str(report_path)
    
    def plot_filtering_results(self, filtered_df: pd.DataFrame, output_dir: str = '.') -> str:
        """Generate visualization showing filtering results."""
        
        if not self.objective_columns:
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = Path(output_dir) / f'filtering_plot_{timestamp}.png'
        
        # Create subplots: histograms + box plots
        n_objectives = len(self.objective_columns)
        fig, axes = plt.subplots(3, n_objectives, figsize=(4*n_objectives, 12))
        
        if n_objectives == 1:
            axes = axes.reshape(3, 1)
        
        for i, col in enumerate(self.objective_columns):
            if col in filtered_df.columns:
                # Top subplot: full range histograms
                axes[0, i].hist(self.df[col], bins=50, alpha=0.7, label='Original', color='blue')
                axes[0, i].hist(filtered_df[col], bins=30, alpha=0.7, label='Filtered', color='red')
                axes[0, i].set_title(f'{col}\nFull Distribution')
                axes[0, i].set_xlabel(col)
                axes[0, i].set_ylabel('Frequency')
                if i == 0:
                    axes[0, i].legend()
                
                # Middle subplot: zoomed histograms with better clarity
                filt_min, filt_max = filtered_df[col].min(), filtered_df[col].max()
                data_range = self.df[col].max() - self.df[col].min()
                
                # Expand zoom to show context around filtered data
                context_range = data_range * 0.05  # Show 5% of total range as context
                zoom_min = max(self.df[col].min(), filt_min - context_range)
                zoom_max = min(self.df[col].max(), filt_max + context_range)
                
                # Filter original data to zoom range
                orig_zoom = self.df[(self.df[col] >= zoom_min) & (self.df[col] <= zoom_max)][col]
                
                # Create histograms with explicit bin edges for better comparison
                bin_edges = np.linspace(zoom_min, zoom_max, 25)
                
                axes[1, i].hist(orig_zoom, bins=bin_edges, alpha=0.6, 
                              label=f'Original in range (n={len(orig_zoom)})', color='blue')
                axes[1, i].hist(filtered_df[col], bins=bin_edges, alpha=0.8, 
                              label=f'Retained (n={len(filtered_df)})', color='red')
                
                # Add threshold line and annotations
                if 'intersection' in self.filtering_stats:
                    threshold = self.filtering_stats['intersection']['thresholds'][col]
                    axes[1, i].axvline(x=threshold, color='green', linestyle='--', 
                                     linewidth=2, alpha=0.8, label=f'Threshold: {threshold:.4f}')
                    
                    # Add text annotation showing which side is "better"
                    axes[1, i].text(0.02, 0.98, 'LOWER\n(worse)', transform=axes[1, i].transAxes, 
                                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                    axes[1, i].text(0.98, 0.98, 'HIGHER\n(better)', transform=axes[1, i].transAxes, 
                                   verticalalignment='top', horizontalalignment='right', 
                                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
                
                axes[1, i].set_title(f'{col}\nZoomed View (High-Performance Region)')
                axes[1, i].set_xlabel(col)
                axes[1, i].set_ylabel('Frequency')
                axes[1, i].set_xlim(zoom_min, zoom_max)
                if i == 0:
                    axes[1, i].legend(loc='upper left')
                
                # Bottom subplot: box plots
                box_data = [self.df[col], filtered_df[col]]
                box_labels = ['Original', 'Filtered']
                box_plot = axes[2, i].boxplot(box_data, tick_labels=box_labels, patch_artist=True)
                
                # Color the box plots to match the histograms
                box_plot['boxes'][0].set_facecolor('lightblue')
                box_plot['boxes'][1].set_facecolor('lightcoral')
                
                axes[2, i].set_title(f'{col}\nBox Plot Comparison')
                axes[2, i].set_ylabel(col)
                
                # Update box plot labels for clarity
                axes[2, i].set_xticklabels(['Original', 'Retained'])
                
                # Add summary statistics as text
                orig_mean = self.df[col].mean()
                filt_mean = filtered_df[col].mean()
                improvement = (filt_mean - orig_mean) / orig_mean * 100
                
                axes[2, i].text(0.5, 0.02, f'Mean improvement: {improvement:+.1f}%', 
                               transform=axes[2, i].transAxes, ha='center',
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def plot_filtered_objective_scatter(self, filtered_df: pd.DataFrame, output_dir: str = '.') -> str:
        """Generate scatter plot showing removed vs retained layouts."""
        
        if not self.objective_columns:
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = Path(output_dir) / f'filtered_objective_scores_scatter_{timestamp}.png'
        
        # Create indices for original vs filtered
        filtered_indices = set(filtered_df.index)
        original_indices = set(self.df.index)
        removed_indices = original_indices - filtered_indices
        
        # Sort by global rank or combined score for consistent x-axis
        sort_col = 'global_rank' if 'global_rank' in self.df.columns else 'combined_score'
        if sort_col in self.df.columns:
            df_sorted = self.df.sort_values(sort_col, ascending=False if sort_col == 'combined_score' else True)
        else:
            df_sorted = self.df
        
        plt.figure(figsize=(14, 8))
        
        colors = self.get_colors(len(self.objective_columns))
        x_positions = range(len(df_sorted))
        
        # Plot each objective
        for i, obj_col in enumerate(self.objective_columns):
            # Separate retained and removed points
            retained_mask = df_sorted.index.isin(filtered_indices)
            removed_mask = df_sorted.index.isin(removed_indices)
            
            # Plot removed points first (so they appear behind retained points)
            if removed_mask.any():
                plt.scatter(np.array(x_positions)[removed_mask], df_sorted[obj_col][removed_mask], 
                           marker='.', s=1, alpha=0.3, color='lightgray', label=f'{obj_col} (removed)' if i == 0 else "")
            
            # Plot retained points
            if retained_mask.any():
                plt.scatter(np.array(x_positions)[retained_mask], df_sorted[obj_col][retained_mask], 
                           marker='.', s=2, alpha=0.8, color=colors[i], label=f'{obj_col} (retained)')
        
        plt.xlabel('Solution Index (sorted by ranking)')
        plt.ylabel('Objective Score')
        plt.title(f'Retained Multi-Objective Scores\n'
                 f'Original: {len(self.df):,} solutions, Retained: {len(filtered_df):,} solutions '
                 f'({len(filtered_df)/len(self.df)*100:.1f}%)')
        
        # Create legend
        try:
            legend = plt.legend(markerscale=10, frameon=True, ncol=2)
            if legend:
                legend.get_frame().set_alpha(0.9)
        except Exception as e:
            print(f"Warning: Could not create legend: {e}")
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def get_colors(self, n_colors):
        """Get high-contrast colors for plotting."""
        base_colors = [
            '#000000',  # Black
            '#17becf',   # Cyan
            '#9467bd',  # Purple
            '#d62728',  # Red
            '#2ca02c',  # Green
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange
            '#bcbd22',  # Olive
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
        ]
        
        if n_colors <= len(base_colors):
            return base_colors[:n_colors]
        else:
            return [base_colors[i % len(base_colors)] for i in range(n_colors)]

def main():
    parser = argparse.ArgumentParser(description='Filter MOO keyboard layout results')
    parser.add_argument('--input', required=True, help='Input CSV file (MOO results)')
    parser.add_argument('--output', help='Output CSV file (filtered results)')
    parser.add_argument('--save-removed', action='store_true', 
                       help='Save removed layouts to a separate CSV file')
    parser.add_argument('--method', choices=['intersection', 'score', 'intersection_score'],
                       default='intersection_score', help='Filtering method')
    
    # Path configuration
    parser.add_argument('--score-layouts-path', help='Path to score_layouts.py script')
    parser.add_argument('--no-poetry', action='store_true', help='Use direct python instead of poetry run')
    
    # General method options
    parser.add_argument('--top-percent', type=float, default=25.0,
                       help='Top percentage to keep (for intersection and score methods)')
    
    # Intersection method options
    parser.add_argument('--intersection-percent', type=float, default=30.0,
                       help='Top percentage for intersection filtering (for intersection_score method)')
    
    # Score method options
    parser.add_argument('--score-type', default='engram_3key_order',
                       help='Score type to compute and filter by (e.g., engram_3key_order, comfort_total)')
    parser.add_argument('--score-percent', type=float, default=15.0,
                       help='Top percentage for score filtering (for intersection_score method)')
    
    # Additional scoring options
    parser.add_argument('--include-scores-in-output', 
                       help='Comma-separated list of additional scores to include in output (e.g., "engram_3key_order,comfort_total")')
    
    # Reporting and visualization options
    parser.add_argument('--report', action='store_true',
                       help='Generate statistical report')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization showing retained vs removed data')
    parser.add_argument('--output-dir', default='output',
                       help='Output directory for reports and plots')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # Create output directory if it doesn't exist
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize filter
        filter_tool = MOOLayoutFilter(args.input, args.verbose, args.score_layouts_path)
        original_df = filter_tool.df.copy()  # Keep original for removed layouts calculation
        
        # Apply filtering method
        if args.method == 'intersection':
            filtered_df = filter_tool.filter_intersection(args.top_percent)
        
        elif args.method == 'score':
            # First add scores to full dataset
            enhanced_df = filter_tool.add_scores(args.score_type, use_poetry=not args.no_poetry)
            filter_tool.df = enhanced_df  # Update the main dataframe
            original_df = enhanced_df.copy()  # Update original to include scores
            filtered_df = filter_tool.filter_by_score(enhanced_df, args.score_type, args.top_percent)
        
        elif args.method == 'intersection_score':
            filtered_df = filter_tool.filter_intersection_score(
                args.intersection_percent, args.score_type, args.score_percent)
            # Update original_df to include any scores that were computed
            original_df = filter_tool.df.copy()
        
        # Add additional scores efficiently to avoid double scoring
        if args.include_scores_in_output:
            additional_scores = [s.strip() for s in args.include_scores_in_output.split(',') if s.strip()]
            if additional_scores:
                print(f"\nAdding additional scores to output: {', '.join(additional_scores)}")
                
                # Step 1: Score the filtered layouts
                filtered_df = filter_tool.add_multiple_scores(
                    filtered_df, additional_scores, use_poetry=not args.no_poetry)
                
                # Step 2: If saving removed layouts, handle the remaining layouts efficiently
                if args.save_removed:
                    # Calculate which layouts need scoring (removed layouts only)
                    filtered_indices = set(filtered_df.index)
                    original_indices = set(original_df.index)
                    removed_indices = original_indices - filtered_indices
                    
                    if removed_indices:
                        print(f"Computing additional scores for {len(removed_indices)} removed layouts...")
                        
                        # Create subset of removed layouts for scoring
                        removed_layouts_to_score = original_df.loc[list(removed_indices)].copy()
                        
                        # Score only the removed layouts
                        scored_removed = filter_tool.add_multiple_scores(
                            removed_layouts_to_score, additional_scores, use_poetry=not args.no_poetry)
                        
                        # Merge scored filtered + scored removed back into original_df
                        original_df = original_df.copy()
                        
                        # Add the additional score columns to original_df (initialize with NaN)
                        for score_type in additional_scores:
                            original_df[score_type] = float('nan')
                        
                        # Update with filtered scores
                        for score_type in additional_scores:
                            if score_type in filtered_df.columns:
                                original_df.loc[filtered_df.index, score_type] = filtered_df[score_type]
                        
                        # Update with removed scores  
                        for score_type in additional_scores:
                            if score_type in scored_removed.columns:
                                original_df.loc[scored_removed.index, score_type] = scored_removed[score_type]
                        
                        print(f"Successfully merged scores for all {len(original_df)} layouts")
                    else:
                        # No removed layouts, just copy additional columns to original_df
                        for score_type in additional_scores:
                            if score_type in filtered_df.columns:
                                original_df[score_type] = float('nan')  # Initialize
                                original_df.loc[filtered_df.index, score_type] = filtered_df[score_type]
                else:
                    # Not saving removed layouts, so we don't need to score anything else
                    print("Not saving removed layouts - additional scoring complete")
                            
        # Calculate removed layouts
        filtered_indices = set(filtered_df.index)
        original_indices = set(original_df.index)
        removed_indices = original_indices - filtered_indices
        removed_df = original_df.loc[list(removed_indices)].copy()
        
        # Generate output filenames
        if args.output:
            output_file = args.output
            # Generate removed filename based on output filename
            output_path = Path(args.output)
            removed_file = output_path.parent / f"removed_{output_path.name}"
        else:
            # Generate both filenames based on input filename and method
            input_stem = Path(args.input).stem
            output_file = f"output/filtered_{input_stem}_{args.method}.csv"
            removed_file = f"output/removed_{input_stem}_{args.method}.csv"
        
        # Save filtered results
        filtered_df.to_csv(output_file, index=False)
        
        # Save removed results if requested
        if args.save_removed and len(removed_df) > 0:
            removed_df.to_csv(removed_file, index=False)
            print(f"Removed layouts saved to: {removed_file}")
        
        print(f"\nFiltering complete!")
        print(f"Input: {len(original_df)} layouts")
        print(f"Filtered (kept): {len(filtered_df)} layouts ({len(filtered_df)/len(original_df)*100:.1f}%)")
        print(f"Removed: {len(removed_df)} layouts ({len(removed_df)/len(original_df)*100:.1f}%)")
        print(f"Filtered results saved to: {output_file}")
        
        # Add summary of removal reasons for intersection method
        if args.method in ['intersection', 'intersection_score'] and args.verbose:
            print(f"\nRemoval breakdown:")
            if 'intersection' in filter_tool.filtering_stats:
                stats = filter_tool.filtering_stats['intersection']
                print(f"  Failed individual thresholds: {stats['filtered_by_thresholds']} layouts")
                print(f"  Failed intersection requirement: {stats['filtered_by_intersection']} layouts")
                
                # Show per-objective failure analysis for removed layouts
                if len(removed_df) > 0:
                    print(f"\nPer-objective analysis of removed layouts:")
                    for col in filter_tool.objective_columns:
                        if col in stats['thresholds'] and col in removed_df.columns:
                            threshold = stats['thresholds'][col]
                            below_threshold = (removed_df[col] < threshold).sum()
                            print(f"  {col}: {below_threshold}/{len(removed_df)} removed layouts below threshold {threshold:.4f}")
        
        # Generate statistical report if requested
        report_path = None
        if args.report:
            report_path = filter_tool.generate_statistical_report(filtered_df, args.output_dir)
            print(f"Statistical report: {report_path}")
        
        # Generate visualization if requested
        plot_path = None
        filtered_scatter_path = None
        if args.plot:
            plot_path = filter_tool.plot_filtering_results(filtered_df, args.output_dir)
            print(f"Filtering visualization: {plot_path}")
            
            # Generate filtered objective scatter plot
            filtered_scatter_path = filter_tool.plot_filtered_objective_scatter(filtered_df, args.output_dir)
            print(f"Filtered objective scatter: {filtered_scatter_path}")
        
        # Print summary statistics
        if args.verbose and len(filtered_df) > 0:
            print(f"\nRetained dataset summary:")
            for col in filter_tool.objective_columns:
                if col in filtered_df.columns:
                    orig_mean = original_df[col].mean()
                    filt_mean = filtered_df[col].mean()
                    filt_std = filtered_df[col].std()
                    improvement = (filt_mean - orig_mean) / orig_mean * 100
                    print(f"  {col}: {filtered_df[col].min():.4f} - {filtered_df[col].max():.4f} (mean: {filt_mean:.4f}, std: {filt_std:.4f}, +{improvement:.1f}%)")
                    
                    # Warning for zero standard deviation
                    if filt_std < 1e-10:
                        print(f"    ⚠️  WARNING: {col} has zero variance - all retained layouts are identical for this objective!")
            
            # Also show removed dataset summary
            if args.save_removed and len(removed_df) > 0:
                print(f"\nRemoved dataset summary:")
                for col in filter_tool.objective_columns:
                    if col in removed_df.columns:
                        removed_mean = removed_df[col].mean()
                        removed_std = removed_df[col].std()
                        orig_mean = original_df[col].mean()
                        degradation = (removed_mean - orig_mean) / orig_mean * 100
                        print(f"  {col}: {removed_df[col].min():.4f} - {removed_df[col].max():.4f} (mean: {removed_mean:.4f}, std: {removed_std:.4f}, {degradation:.1f}%)")
        
        # Enhanced quality check with warnings
        total_unique_values = 0
        if args.method in ['intersection', 'intersection_score']:
            print(f"\nQuality check:")
            
            # Check if we have enough layouts for meaningful analysis
            if len(filtered_df) < 5:
                print(f"⚠️  WARNING: Only {len(filtered_df)} layouts retained - may be insufficient for analysis")
                print(f"   Consider using higher percentages or different filtering method")
            
            # Count how many layouts are in top 10% of ALL original objectives
            quality_count = len(filtered_df)
            for col in filter_tool.objective_columns:
                if col in filtered_df.columns:
                    p90_threshold = np.percentile(original_df[col], 90)
                    in_top10 = (filtered_df[col] >= p90_threshold).sum()
                    quality_count = min(quality_count, in_top10)
            
            print(f"Layouts in top 10% of ALL objectives: {quality_count} ({quality_count/len(filtered_df)*100:.1f}% of retained set)")
            
            # Check for diversity
            for col in filter_tool.objective_columns:
                if col in filtered_df.columns:
                    unique_values = filtered_df[col].nunique()
                    total_unique_values += unique_values
                    if unique_values <= 1:
                        print(f"⚠️  {col}: All retained layouts have identical values")
            
            if total_unique_values <= len(filter_tool.objective_columns):
                print(f"\n⚠️  WARNING: Retained layouts show little diversity across objectives")
                print(f"   This suggests filtering was too restrictive")
        
        # Success message with recommendations based on actual diversity
        has_diversity = total_unique_values > len(filter_tool.objective_columns)
        retention_rate = len(filtered_df) / len(original_df) * 100
        
        if len(filtered_df) >= 10 and has_diversity and retention_rate >= 1.0:
            print(f"\n✅ Filtering successful! Retained {len(filtered_df)} diverse layouts ({retention_rate:.1f}%)")
        elif len(filtered_df) >= 10 and has_diversity:
            print(f"\n⚠️  Filtering very restrictive but diverse: {len(filtered_df)} layouts ({retention_rate:.1f}%)")
            print(f"   Consider higher percentages for more layouts")
        elif len(filtered_df) >= 10:
            print(f"\n⚠️  Filtering retained {len(filtered_df)} layouts but they lack diversity")
            print(f"   All layouts are nearly identical - intersection method too restrictive")
        elif len(filtered_df) >= 2:
            print(f"\n⚠️  Filtering completed but retained only {len(filtered_df)} layouts ({retention_rate:.2f}%)")
            print(f"   Consider using higher percentages or 'score' method for more diversity")
        else:
            print(f"\n⚠️  Filtering too restrictive! Only {len(filtered_df)} layout(s) retained")
            print(f"   Recommended: use 'score' method or much higher --top-percent (70-90%)")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
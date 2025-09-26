#!/usr/bin/env python3
"""
Comprehensive Keyboard Layout Analysis Tool

This script provides weighted scoring and visualization for keyboard layouts.
It combines multi-objective optimization analysis with weighted average scoring.

Input/Output Format:
- items: letters in assignment order (e.g., "etaoinsrhldcum") 
- positions: QWERTY positions where those letters go (e.g., "KJ;ASDVRLFUEIM")
- layout_qwerty: layout string in 32-key QWERTY order (QWERTYUIOPASDFGHJKL;ZXCVBNM,./[')
  with spaces for unassigned positions, such as: "  cr  du  oinl  teha   s  m     "

Usage:
    # Directory analysis with weighted scoring
    python layouts_compare.py --input data_dir/ --scores auto --plot --output-dir analysis_output

    # Equal weights
    poetry run python3 layouts_compare.py \
        --input output/layouts_consolidate_moo_solutions.csv \
        --scores "engram_key_preference,engram_avg4_score" \
        --include-scores "engram_order" \
        --output output/layouts_compare_results.csv --plot --report

    # Custom weights (50% engram_key_preference, 50% engram_avg4_score) 
    poetry run python3 layouts_compare.py \
        --input output/layouts_consolidate_moo_solutions.csv \
        --scores "engram_key_preference,engram_avg4_score" \
        --weights "0.5,0.5" \
        --include-scores "engram_order" \
        --score-table "engram_3key_scores_order.csv" \
        --output output/layouts_compare_results.csv --plot --report --verbose

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import sys
from typing import List
from io import StringIO
import subprocess
import tempfile
import glob
from collections import defaultdict
from datetime import datetime

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

class ComprehensiveLayoutAnalyzer:
    """Comprehensive layout analysis combining weighted scoring and visualization."""
    
    def __init__(self, input_path: str, output_dir: str = "output", verbose: bool = False, score_layouts_path: str = None):
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.score_layouts_path = score_layouts_path or self._find_score_layouts_script()
        
        self.df = None
        self.input_type = None
        self.score_columns = []
        self.objective_columns = []
        self.item_col = None
        self.pair_col = None
        
        # High-contrast color palette
        self.colors = [
            '#1f77b4',  # Blue
            "#000000",  # Black  
            '#800080',  # Purple
            '#ff9896',  # Light Red
            '#d62728',  # Red
            '#2ca02c',  # Green
            '#008080',  # Teal
            '#17becf',  # Cyan
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Olive
            '#c5b0d5',  # Light Purple
            '#c49c94',  # Light Brown
            '#f7b6d3',  # Light Pink
            '#dbdb8d',  # Light Olive
            '#9edae5',  # Light Cyan
            '#1a1a1a',  # Near Black
            '#800000'   # Maroon
        ]
        
    def get_colors(self, n_colors):
        """Get high-contrast colors, cycling if needed."""
        if n_colors <= len(self.colors):
            return self.colors[:n_colors]
        else:
            return [self.colors[i % len(self.colors)] for i in range(n_colors)]
    
    def _find_score_layouts_script(self) -> str:
        """Find the score_layouts.py script in common locations."""
        possible_paths = [
            '../keyboard_layout_scorers/score_layouts.py',  # Most likely location
            'score_layouts.py',  # Same directory
            '../score_layouts.py',  # Parent directory
            '../../keyboard_layout_scorers/score_layouts.py',  # Another level up
            './keyboard_layout_scorers/score_layouts.py',  # Subdirectory
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                if self.verbose:
                    print(f"Found score_layouts.py at: {path}")
                return path
        
        return '../keyboard_layout_scorers/score_layouts.py'  # Default expected location
    
    def detect_input_type(self, input_path):
        """Detect whether input is directory or file."""
        path = Path(input_path)
        if path.is_dir():
            return "directory"
        elif path.is_file():
            return "file"
        else:
            raise ValueError(f"Input path does not exist: {input_path}")
    
    def load_directory_input(self, input_dir, file_pattern="moo_results_*.csv", max_files=None):
        """Load multiple individual MOO result files from directory."""
        pattern_path = f"{input_dir}/{file_pattern}"
        files = glob.glob(pattern_path)
        
        if max_files:
            files = files[:max_files]
            
        if not files:
            print(f"No CSV files found matching pattern: {pattern_path}")
            return pd.DataFrame()
        
        print(f"Loading {len(files)} individual MOO result files...")
        
        all_results = []
        successful_files = 0
        
        for i, filepath in enumerate(files):
            if i % 10 == 0:
                print(f"Processing file {i+1}/{len(files)}")
                
            try:
                df_file = pd.read_csv(filepath)
                
                # Add metadata
                config_id = Path(filepath).stem.replace('moo_results_config_', '').replace('moo_results_', '')
                if '_' in config_id:
                    config_id = config_id.split('_')[0]
                    
                df_file['config_id'] = config_id
                df_file['source_file'] = Path(filepath).name
                
                # Rename 'rank' to 'source_rank'
                if 'rank' in df_file.columns:
                    df_file['source_rank'] = df_file['rank']
                    df_file = df_file.drop('rank', axis=1)
                
                all_results.append(df_file)
                successful_files += 1
                
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                
        if not all_results:
            return pd.DataFrame()
            
        combined_df = pd.concat(all_results, ignore_index=True)
        print(f"Loaded {len(combined_df):,} total solutions from {successful_files} files")
        return combined_df

    def load_file_input(self, filepath):
        """Load single consolidated CSV file with enhanced validation and format detection."""
        print(f"Loading layout analysis file: {filepath}")
        
        try:
            # Read all lines first
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"File has {len(lines)} total lines")
            
            # Find the actual CSV header
            data_start_idx = 0
            header_found = False
            
            expected_columns = ['config_id', 'items', 'positions', 'layout_qwerty']
            
            for i, line in enumerate(lines):
                line_clean = line.strip()
                
                # Skip empty lines
                if not line_clean:
                    continue
                
                # Check if this looks like the actual CSV header
                is_quoted_line = line_clean.startswith('"') and line_clean.endswith('"')
                comma_count = line_clean.count(',')
                matching_cols = sum(1 for col in expected_columns if col in line_clean)
                
                if (not is_quoted_line and comma_count >= 10 and matching_cols >= 3):
                    data_start_idx = i
                    header_found = True
                    if self.verbose:
                        print(f"Found CSV header at line {i + 1}")
                    break
            
            # Parse from detected header
            if data_start_idx > 0:
                csv_content = '\n'.join(line.strip() for line in lines[data_start_idx:] if line.strip())
                df = pd.read_csv(StringIO(csv_content))
            else:
                df = pd.read_csv(filepath)
            
            # Validate layout data if available
            if 'items' in df.columns and 'positions' in df.columns:
                valid_mask = df.apply(lambda row: validate_layout_data(row['items'], row['positions']), axis=1)
                invalid_count = len(df) - valid_mask.sum()
                
                if invalid_count > 0:
                    if self.verbose:
                        print(f"Filtered out {invalid_count} rows with invalid layout data")
                    df = df[valid_mask].copy()
            
            if self.verbose:
                print(f"Loaded {len(df)} valid layouts")
                #print(f"Available columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return pd.DataFrame()
                                
    def detect_objective_columns(self, df):
        """Detect objective columns and special item/pair columns."""
        # Comprehensive metadata columns to exclude
        metadata_cols = [
            'config_id', 'rank', 'source_rank', 'items', 'positions', 'layout', 'layout_qwerty',
            'combined_score', 'source_file', 'global_rank', 'weighted_score',
            # Config metadata columns
            'config_items_to_assign', 'config_positions_to_assign', 
            'config_items_assigned', 'config_positions_assigned',
            'config_items_constrained', 'config_positions_constrained',
            'objectives_used', 'weights_used', 'maximize_used',
            # Other potential metadata
            'source_config', 'timestamp', 'run_id'
        ]
        
        # Find objective columns (likely numeric and not metadata)
        potential_objectives = []
        for col in df.columns:
            if col not in metadata_cols:
                # Check if column contains numeric data and has variation
                try:
                    if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 1:
                        potential_objectives.append(col)
                except (ValueError, TypeError):
                    continue
        
        # Try to identify specific item/pair columns for Pareto analysis
        item_possibilities = ['Complete Item', 'item_score', 'Opt Item Score']
        pair_possibilities = ['Complete Pair', 'item_pair_score', 'Opt Item-Pair Score']
        
        item_col = None
        pair_col = None
        
        for col in item_possibilities:
            if col in df.columns:
                item_col = col
                break
        
        for col in pair_possibilities:
            if col in df.columns:
                pair_col = col
                break
        
        return potential_objectives, item_col, pair_col
    
    def load_data(self, **kwargs):
        """Load data from either directory or file input."""
        self.input_type = self.detect_input_type(self.input_path)
        
        if self.input_type == "directory":
            self.df = self.load_directory_input(self.input_path, **kwargs)
        else:
            self.df = self.load_file_input(self.input_path)
        
        if self.df.empty:
            raise ValueError("No data loaded successfully")
        
        # Detect objectives
        self.objective_columns, self.item_col, self.pair_col = self.detect_objective_columns(self.df)
        
        if not self.objective_columns:
            print("Warning: No objective columns detected")
            print(f"Available columns: {list(self.df.columns)}")
        else:
            if self.verbose:
                print(f"Detected objective columns: {self.objective_columns}")
                if self.item_col and self.pair_col:
                    print(f"Detected item/pair columns: {self.item_col}, {self.pair_col}")
        
        return self.df
    
    def add_multiple_scores(self, score_types: List[str], use_poetry: bool = True, score_table: str = None) -> pd.DataFrame:
        """Add multiple score types to the dataframe efficiently."""
        if not score_types or self.df is None:
            return self.df
        
        # Create temporary file with layout data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            # Save layouts in format expected by score_layouts.py
            temp_df = self.df[['items', 'positions']].copy()
            temp_df.to_csv(temp_file.name, index=False)
            temp_file_path = temp_file.name
        
        results_df = self.df.copy()
        
        # Get the absolute path to the script and its directory
        script_path = Path(self.score_layouts_path).resolve()
        script_dir = script_path.parent
        
        if self.verbose:
            print(f"Score layouts script: {script_path}")
            print(f"Working directory: {script_dir}")
        
        for score_type in score_types:
            if self.verbose:
                print(f"Computing {score_type} scores...")
            
            try:
                # Build command
                if use_poetry:
                    cmd = ['poetry', 'run', 'python3', str(script_path)]
                else:
                    cmd = ['python3', str(script_path)]
                                
                cmd.extend([
                    '--compare-file', temp_file_path,
                    '--scorer', score_type,
                    '--format', 'score_only',
                    '--quiet'
                ])

                # Add custom score table if specified
                if score_table:
                    cmd.extend(['--score-table', f'tables/{score_table}'])
                                    
                if self.verbose:
                    print(f"Running command: {' '.join(cmd)}")
                    print(f"Working directory: {script_dir}")
                
                # Run scoring from the script's directory to ensure tables/ can be found
                result = subprocess.run(cmd, 
                                    capture_output=True, 
                                    text=True, 
                                    timeout=300,
                                    cwd=str(script_dir))
                
                if result.returncode != 0:
                    print(f"Warning: Failed to compute {score_type}")
                    print(f"  Command: {' '.join(cmd)}")
                    print(f"  Return code: {result.returncode}")
                    if self.verbose:
                        print(f"  Stdout: {result.stdout}")
                        print(f"  Stderr: {result.stderr}")
                    results_df[score_type] = float('nan')
                    continue
                
                # Parse output - expect just score values, one per line
                scores = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            scores.append(float(line.strip()))
                        except ValueError:
                            print(f"Warning: Could not parse score line: '{line}'")
                            break
                
                if len(scores) == len(results_df):
                    results_df[score_type] = scores
                    if self.verbose:
                        print(f"Added {score_type} scores (range: {min(scores):.4f} - {max(scores):.4f})")
                else:
                    print(f"Warning: Expected {len(results_df)} scores, got {len(scores)} for {score_type}")
                    results_df[score_type] = float('nan')
                    
            except subprocess.TimeoutExpired:
                print(f"Warning: Scoring {score_type} timed out")
                results_df[score_type] = float('nan')
            except Exception as e:
                print(f"Warning: Error computing {score_type}: {e}")
                results_df[score_type] = float('nan')
                            
        # Clean up temp file
        try:
            Path(temp_file_path).unlink()
        except:
            pass
        
        self.df = results_df
        return results_df
    
    def calculate_weighted_scores(self, score_columns: List[str], weights: List[float] = None) -> pd.DataFrame:
        """Calculate weighted average scores instead of rankings."""
        if not score_columns:
            raise ValueError("No score columns specified for scoring")
        
        # Validate score columns exist
        missing_cols = [col for col in score_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Score columns not found in data: {missing_cols}")
        
        df = self.df.copy()
        
        # Add layout_qwerty column if items and positions exist
        if 'items' in df.columns and 'positions' in df.columns and 'layout_qwerty' not in df.columns:
            if self.verbose:
                print("Adding layout_qwerty column...")
            df['layout_qwerty'] = df.apply(
                lambda row: convert_items_positions_to_qwerty_layout(row['items'], row['positions']), 
                axis=1
            )
        
        # Default to equal weights if not specified
        if weights is None:
            weights = [1.0] * len(score_columns)
        elif len(weights) != len(score_columns):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of score columns ({len(score_columns)})")
        
        # Normalize weights to sum to 1
        weight_sum = sum(weights)
        if weight_sum == 0:
            raise ValueError("Sum of weights cannot be zero")
        weights = [w / weight_sum for w in weights]
        
        print(f"Calculating weighted scores for {len(score_columns)} score columns:")
        for col, weight in zip(score_columns, weights):
            print(f"  - {col}: weight = {weight:.3f}")
        
        # Calculate weighted score
        weighted_score = 0
        for col, weight in zip(score_columns, weights):
            if self.verbose:
                col_stats = df[col].describe()
                print(f"  {col}: min={col_stats['min']:.6f}, max={col_stats['max']:.6f}, mean={col_stats['mean']:.6f}")
            weighted_score += weight * df[col]
        
        df['weighted_score'] = weighted_score
        
        # Sort by weighted score (higher is better)
        df = df.sort_values('weighted_score', ascending=False).reset_index(drop=True)
        
        # Add rank based on weighted score
        df['global_rank'] = range(1, len(df) + 1)
        
        print(f"Weighted scoring completed:")
        print(f"  Best weighted score: {df['weighted_score'].iloc[0]:.6f}")
        print(f"  Worst weighted score: {df['weighted_score'].iloc[-1]:.6f}")
        print(f"  Score range: {df['weighted_score'].max() - df['weighted_score'].min():.6f}")
        print(f"  Unique scores: {df['weighted_score'].nunique()}")
        
        self.df = df
        self.score_columns = score_columns
        return df
     
    def apply_constraint_filter(self, constraints_string):
        """Apply letter-position constraints to filter solutions."""
        if not constraints_string or self.df is None:
            return
            
        # Parse constraints
        constraints = {}
        for constraint in constraints_string.split(','):
            constraint = constraint.strip()
            if ':' not in constraint:
                continue
            letter, position = constraint.split(':', 1)
            constraints[letter.strip()] = position.strip()
        
        if not constraints:
            return
            
        print(f"Applying constraints: {constraints}")
        
        # Filter function
        def meets_constraints(row):
            items = list(row['items'])
            positions = list(row['positions'])
            
            for letter, required_position in constraints.items():
                try:
                    letter_index = items.index(letter)
                    if positions[letter_index] != required_position:
                        return False
                except ValueError:
                    return False
            return True
        
        original_count = len(self.df)
        self.df = self.df[self.df.apply(meets_constraints, axis=1)].copy()
        print(f"Filtered from {original_count:,} to {len(self.df):,} solutions")
        
        if len(self.df) == 0:
            raise ValueError("No solutions remain after filtering")
    
    # PLOTTING FUNCTIONS
    
    def plot_objective_scatter(self):
        """Create scatter plot of objective scores."""
        if not self.objective_columns:
            return
            
        plt.figure(figsize=(14, 8))
        
        # Sort by weighted score if available, otherwise by first objective
        if 'weighted_score' in self.df.columns:
            df_sorted = self.df.sort_values('weighted_score', ascending=False)
            sort_desc = " (sorted by weighted score)"
        elif 'global_rank' in self.df.columns:
            df_sorted = self.df.sort_values('global_rank')
            sort_desc = " (sorted by global rank)"
        else:
            df_sorted = self.df.sort_values(self.objective_columns[0], ascending=False)
            sort_desc = f" (sorted by {self.objective_columns[0]})"
        
        colors = self.get_colors(len(self.objective_columns))
        x_positions = range(len(df_sorted))
        
        for i, obj_col in enumerate(self.objective_columns):
            plt.scatter(x_positions, df_sorted[obj_col], 
                       marker='.', s=1, alpha=0.7, 
                       label=obj_col, color=colors[i])
        
        plt.xlabel(f'Solution Index{sort_desc}')
        plt.ylabel('Objective Score')
        plt.title(f'Multi-Objective Scores ({len(df_sorted):,} solutions)')
        
        try:
            legend = plt.legend(markerscale=20, frameon=True)
            if legend:
                legend.get_frame().set_alpha(0.9)
        except Exception as e:
            print(f"Warning: Could not create legend: {e}")
            
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = self.output_dir / 'layouts_compare_moo_scores_scatter.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_path}")
    
    def plot_pareto_front_2d(self):
        """Create 2D Pareto front if we have exactly 2 objectives."""
        if len(self.objective_columns) != 2:
            return
            
        x_col, y_col = self.objective_columns[0], self.objective_columns[1]
        has_source_files = 'source_file' in self.df.columns
        
        plt.figure(figsize=(12, 8))
        
        if has_source_files:
            # Color by source file
            unique_sources = self.df['source_file'].unique()
            colors = self.get_colors(len(unique_sources))
            source_color_map = dict(zip(unique_sources, colors))
            
            for source in unique_sources:
                source_data = self.df[self.df['source_file'] == source]
                plt.scatter(source_data[x_col], source_data[y_col], 
                           alpha=0.6, s=50, label=f'{source} ({len(source_data)})',
                           color=source_color_map[source])
            
            try:
                legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
                if legend:
                    legend.get_frame().set_alpha(0.9)
            except Exception as e:
                print(f"Warning: Could not create legend: {e}")
                
            title_suffix = " - Colored by Source"
        else:
            plt.scatter(self.df[x_col], self.df[y_col], 
                       alpha=0.6, s=50, color='blue')
            title_suffix = ""
        
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'2D Objective Space{title_suffix} ({len(self.df):,} solutions)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = self.output_dir / f'layouts_compare_moo_space_2d{"_colored" if has_source_files else ""}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_path}")

    def plot_correlation_matrix(self):
        """Create correlation matrix of objectives."""
        if len(self.objective_columns) < 2:
            return
            
        obj_data = self.df[self.objective_columns].dropna()
        if obj_data.empty:
            return
            
        corr_matrix = obj_data.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title(f'Objective Correlation Matrix ({len(obj_data):,} solutions)')
        plt.tight_layout()
        plot_path = self.output_dir / 'layouts_compare_moo_correlation_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_path}")
    
    def plot_stability_matrix(self):
        """Create letter-position stability heatmap."""
        if 'items' not in self.df.columns or 'positions' not in self.df.columns:
            return
            
        print("Creating stability matrix...")
        
        # Calculate assignment frequencies
        letter_positions = defaultdict(set)
        position_letters = defaultdict(set)
        assignment_counts = defaultdict(int)
        
        for _, row in self.df.iterrows():
            items = list(row['items'])
            positions = list(row['positions'])
            
            for letter, position in zip(items, positions):
                letter_positions[letter].add(position)
                position_letters[position].add(letter)
                assignment_counts[(letter, position)] += 1
        
        # Sort by stability (fewer positions/letters = more stable)
        letters_by_stability = sorted(letter_positions.keys(), 
                                    key=lambda x: len(letter_positions[x]))
        positions_by_stability = sorted(position_letters.keys(), 
                                      key=lambda x: len(position_letters[x]))
        
        # Create matrix
        matrix = np.zeros((len(letters_by_stability), len(positions_by_stability)))
        letter_indices = {letter: i for i, letter in enumerate(letters_by_stability)}
        position_indices = {pos: i for i, pos in enumerate(positions_by_stability)}
        
        for (letter, position), count in assignment_counts.items():
            if letter in letter_indices and position in position_indices:
                matrix[letter_indices[letter], position_indices[position]] = count
        
        # Plot heatmap
        plt.figure(figsize=(16, 12))
        annot_matrix = np.where(matrix == 0, '', matrix.astype(int).astype(str))
        
        sns.heatmap(matrix, 
                   xticklabels=positions_by_stability,
                   yticklabels=letters_by_stability,
                   cmap='Reds', linewidths=0.5, square=True,
                   annot=annot_matrix, fmt='', annot_kws={'size': 8},
                   cbar_kws={'label': 'Assignment Count'})
        
        plt.title(f'Letter-Position Stability Matrix ({len(self.df):,} solutions)')
        plt.xlabel('Positions (ordered by stability)')
        plt.ylabel('Letters (ordered by stability)')
        plt.tick_params(left=False, bottom=False)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plot_path = self.output_dir / 'layouts_compare_assignment_stability_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_path}")
        
        # Print stability stats
        if self.verbose:
            print(f"\nMost stable letters (fewest positions):")
            for letter in letters_by_stability[:8]:
                print(f"  {letter}: {len(letter_positions[letter])} unique positions")
            
            print(f"Most stable positions (fewest letters):")
            for position in positions_by_stability[:8]:
                print(f"  {position}: {len(position_letters[position])} unique letters")
    
    def plot_ranking_distribution(self):
        """Plot distribution of weighted scores."""
        if not self.score_columns or 'weighted_score' not in self.df.columns:
            return
        
        # Plot score distributions
        fig, axes = plt.subplots(2, len(self.score_columns) + 1, figsize=(4*(len(self.score_columns)+1), 8))
        if len(self.score_columns) == 0:
            return
            
        # Individual score histograms
        for i, score_col in enumerate(self.score_columns):
            axes[0, i].hist(self.df[score_col], bins=50, alpha=0.7, color=self.colors[i])
            axes[0, i].set_title(f'{score_col}\nScore Distribution')
            axes[0, i].set_xlabel('Score Value')
            axes[0, i].set_ylabel('Frequency')
            
            # Box plot of scores
            axes[1, i].boxplot(self.df[score_col], patch_artist=True)
            axes[1, i].set_title(f'{score_col}\nScore Box Plot')
            axes[1, i].set_ylabel('Score Value')
        
        # Weighted score distribution
        axes[0, -1].hist(self.df['weighted_score'], bins=50, alpha=0.7, color='red')
        axes[0, -1].set_title('Weighted Score\nDistribution')
        axes[0, -1].set_xlabel('Weighted Score')
        axes[0, -1].set_ylabel('Frequency')
        
        # Global rank distribution
        if 'global_rank' in self.df.columns:
            axes[1, -1].hist(self.df['global_rank'], bins=50, alpha=0.7, color='red')
            axes[1, -1].set_title('Global Rank\nDistribution')
            axes[1, -1].set_xlabel('Global Rank')
            axes[1, -1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'layouts_compare_score_distributions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_path}")
    
    def standardize_output_columns(self) -> pd.DataFrame:
        """Standardize column ordering for output."""
        if len(self.df) == 0:
            return self.df
        
        # Standard column order
        standard_cols = ['config_id', 'items', 'positions', 'layout_qwerty']
        scoring_cols = ['global_rank', 'weighted_score']
        
        # Identify other columns
        metadata_cols = standard_cols + scoring_cols + ['source_rank', 'source_file', 'layout', 'combined_score']
        other_cols = sorted([col for col in self.df.columns if col not in metadata_cols])
        
        # Build ordered column list
        ordered_cols = []
        
        # Add standard columns that exist
        for col in standard_cols:
            if col in self.df.columns:
                ordered_cols.append(col)
        
        # Add scoring columns
        for col in scoring_cols:
            if col in self.df.columns:
                ordered_cols.append(col)
        
        # Add source_rank after scoring
        if 'source_rank' in self.df.columns:
            ordered_cols.append('source_rank')
        
        # Add score columns used in weighting (in specified order)
        ordered_cols.extend([col for col in self.score_columns if col in self.df.columns])
        
        # Add remaining columns
        remaining = [col for col in other_cols if col not in ordered_cols]
        ordered_cols.extend(remaining)
        
        # Add final metadata columns
        for col in ['source_file', 'layout', 'combined_score']:
            if col in self.df.columns and col not in ordered_cols:
                ordered_cols.append(col)
        
        return self.df[ordered_cols]
    
    def generate_report(self):
        """Generate analysis report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f'layouts_compare_report_{timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write("Layout Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input: {self.input_path}\n")
            f.write(f"Input type: {self.input_type}\n")
            f.write(f"Total solutions: {len(self.df):,}\n\n")
            
            # Weighted scoring summary
            if self.score_columns and 'weighted_score' in self.df.columns:
                f.write("Weighted Scoring Summary:\n")
                f.write("-" * 25 + "\n")
                f.write(f"Score columns used: {len(self.score_columns)}\n")
                for col in self.score_columns:
                    f.write(f"  - {col}\n")
                f.write(f"\nBest weighted score: {self.df['weighted_score'].max():.6f}\n")
                f.write(f"Worst weighted score: {self.df['weighted_score'].min():.6f}\n")
                f.write(f"Score range: {self.df['weighted_score'].max() - self.df['weighted_score'].min():.6f}\n\n")
                
                # Top 10 layouts
                f.write("Top 10 Scored Layouts:\n")
                f.write("-" * 22 + "\n")
                display_cols = ['global_rank', 'weighted_score'] + self.score_columns[:3]
                if 'layout_qwerty' in self.df.columns:
                    display_cols.append('layout_qwerty')
                available_cols = [col for col in display_cols if col in self.df.columns]
                f.write(self.df[available_cols].head(10).to_string(index=False))
                f.write("\n\n")
            
            # Objective statistics
            if self.objective_columns:
                f.write("Objective Statistics:\n")
                f.write("-" * 21 + "\n")
                for col in self.objective_columns:
                    if col in self.df.columns:
                        values = self.df[col].dropna()
                        f.write(f"{col}:\n")
                        f.write(f"  Mean: {values.mean():.6f}\n")
                        f.write(f"  Std:  {values.std():.6f}\n")
                        f.write(f"  Min:  {values.min():.6f}\n")
                        f.write(f"  Max:  {values.max():.6f}\n\n")
        
        print(f"Analysis report saved: {report_path}")
        return str(report_path)
    
    def save_results(self, output_file: str = None):
        """Save analysis results with standardized column ordering."""
        if self.df is None:
            raise ValueError("No data to save")
        
        # Standardize column ordering
        standardized_df = self.standardize_output_columns()
        
        if output_file:
            standardized_df.to_csv(output_file, index=False)
            print(f"Results saved to: {output_file}")
        else:
            # Save to output directory
            output_csv = self.output_dir / 'analysis_results.csv'
            standardized_df.to_csv(output_csv, index=False)
            print(f"Results saved to: {output_csv}")
        
        return standardized_df
    
    def print_summary(self):
        """Print analysis summary."""
        if self.df is None:
            return
            
        print(f"\n=== Analysis Summary ===")
        print(f"Input type: {self.input_type}")
        print(f"Total solutions: {len(self.df):,}")
        
        if 'source_file' in self.df.columns:
            file_counts = self.df['source_file'].value_counts()
            print(f"Number of source files: {len(file_counts)}")
        
        if self.score_columns and 'weighted_score' in self.df.columns:
            print(f"\nWeighted Scoring Summary:")
            print(f"Score columns used: {len(self.score_columns)}")
            print(f"Best weighted score: {self.df['weighted_score'].max():.6f}")
            print(f"Score range: {self.df['weighted_score'].max() - self.df['weighted_score'].min():.6f}")
            
            # Show top 3 layouts
            print(f"\nTop 3 Layouts:")
            display_cols = ['global_rank', 'weighted_score']
            display_cols.extend([col for col in self.score_columns[:2]])  # Limit to first 2 scores
            if 'layout_qwerty' in self.df.columns:
                display_cols.append('layout_qwerty')
            available_cols = [col for col in display_cols if col in self.df.columns]
            print(self.df[available_cols].head(3).to_string(index=False))
        
        if self.objective_columns:
            print(f"\nObjective Summary:")
            for col in self.objective_columns[:5]:  # Limit to first 5
                if col in self.df.columns:
                    values = self.df[col].dropna()
                    print(f"  {col}: {values.min():.4f} to {values.max():.4f} (mean: {values.mean():.4f})")

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Keyboard Layout Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score and analyze layouts with equal weights
  python layouts_compare.py --input layouts.csv --scores "engram_key_preference,engram_avg4_score" --plot

  # Use custom weights (must match number of score columns)
  python layouts_compare.py --input layouts.csv --scores "same_finger_bigrams,alternation,rolls" --weights "0.5,0.3,0.2" --plot

  # Add additional scores and visualize
  python layouts_compare.py --input layouts.csv --scores auto --include-scores "comfort_total" --plot

  # Directory analysis with weighted scoring
  python layouts_compare.py --input data_dir/ --scores auto --weights "0.4,0.3,0.2,0.1" --plot --report
        """
    )
    
    parser.add_argument('--input', required=True, help='Input CSV file or directory with layout data')
    parser.add_argument('--output', help='Output CSV file (if not specified, saves to output-dir)')
    parser.add_argument('--output-dir', default='output', help='Output directory for plots and results')
    
    # Scoring and weighting options
    parser.add_argument('--scores', help='Comma-separated list of score columns for weighting, or "auto" for all available')
    parser.add_argument('--weights', help='Comma-separated list of weights for score columns (default: equal weights)')
    parser.add_argument('--include-scores', help='Comma-separated list of additional scores to compute and include')
    parser.add_argument('--score-table', help='Score table CSV file name (e.g., engram_3key_scores_order.csv)')
    parser.add_argument('--score-layouts-path', help='Path to score_layouts.py script')
    parser.add_argument('--no-poetry', action='store_true', help='Use direct python instead of poetry run')
    
    # Analysis options
    parser.add_argument('--filter-assignments', help='Filter by letter-position assignments: "letter:position,letter:position"')
    parser.add_argument('--list-scores', action='store_true', help='List available score columns and exit')
    
    # Output options
    parser.add_argument('--plot', action='store_true', help='Generate all visualization plots')
    parser.add_argument('--report', action='store_true', help='Generate comprehensive analysis report')
    
    # Directory input options
    parser.add_argument('--file-pattern', default='moo_results_*.csv', help='File pattern for directory input')
    parser.add_argument('--max-files', type=int, help='Maximum files to process for directory input')
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed information')
    
    args = parser.parse_args()
    
    try:
        analyzer = ComprehensiveLayoutAnalyzer(
            args.input, 
            args.output_dir, 
            args.verbose,
            args.score_layouts_path
        )
        
        # Load data
        load_kwargs = {}
        if Path(args.input).is_dir():
            load_kwargs['file_pattern'] = args.file_pattern
            if args.max_files:
                load_kwargs['max_files'] = args.max_files
        
        analyzer.load_data(**load_kwargs)
        
        # List scores if requested
        if args.list_scores:
            print(f"Available score columns in {args.input}:")
            if analyzer.objective_columns:
                for i, col in enumerate(analyzer.objective_columns, 1):
                    print(f"  {i:2d}. {col}")
            else:
                print("  No score columns detected")
            return 0
        
        # Apply constraints if provided
        if args.filter_assignments:
            analyzer.apply_constraint_filter(args.filter_assignments)
        
        # Add additional scores if requested
        if args.include_scores:
            additional_scores = [s.strip() for s in args.include_scores.split(',') if s.strip()]
            if additional_scores:
                print(f"\nAdding additional scores: {', '.join(additional_scores)}")
                analyzer.add_multiple_scores(additional_scores, use_poetry=not args.no_poetry, score_table=getattr(args, 'score_table', None))

        # Calculate weighted scores if scores specified
        if args.scores:
            available_scores = analyzer.objective_columns
            
            if args.scores.lower() == 'auto':
                score_columns = available_scores
                if not score_columns:
                    print("Error: No score columns detected for auto mode")
                    return 1
                print(f"Auto-detected {len(score_columns)} score columns")
            else:
                score_columns = [col.strip() for col in args.scores.split(',')]
                
                # Validate score columns
                invalid_cols = [col for col in score_columns if col not in available_scores]
                if invalid_cols:
                    print(f"Error: Invalid score columns: {invalid_cols}")
                    print(f"Available columns: {available_scores}")
                    return 1
            
            # Parse weights if provided
            weights = None
            if args.weights:
                try:
                    weights = [float(w.strip()) for w in args.weights.split(',')]
                    if len(weights) != len(score_columns):
                        print(f"Error: Number of weights ({len(weights)}) must match number of score columns ({len(score_columns)})")
                        return 1
                    if all(w == 0 for w in weights):
                        print("Error: All weights cannot be zero")
                        return 1
                    print(f"Using custom weights: {weights}")
                except ValueError:
                    print(f"Error: Invalid weight format. Use comma-separated numbers like '0.5,0.3,0.2'")
                    return 1
            else:
                print(f"Using equal weights for all {len(score_columns)} score columns")
            
            # Calculate weighted scores
            analyzer.calculate_weighted_scores(score_columns, weights)
        
        # Generate plots if requested
        if args.plot:
            print("\nGenerating visualizations...")
            analyzer.plot_objective_scatter()
            analyzer.plot_pareto_front_2d()
            analyzer.plot_correlation_matrix()
            analyzer.plot_stability_matrix()
            analyzer.plot_ranking_distribution()
        
        # Generate report if requested
        if args.report:
            analyzer.generate_report()
        
        # Print summary and save results
        analyzer.print_summary()
        analyzer.save_results(args.output)
        
        print(f"\nAnalysis complete! Results saved to {args.output_dir}/")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
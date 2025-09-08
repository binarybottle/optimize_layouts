#!/usr/bin/env python3
"""
Multi-Objective Optimization Analysis Tool

This script analyzes MOO results with automatic input detection:
- Directory input: Analyzes multiple individual CSV files from separate optimization runs
- File input: Analyzes consolidated global Pareto CSV file

Features:
- Auto-detects input type and format
- Unified analysis pipeline regardless of input source
- Visualization suite (Pareto fronts, correlations, stability matrices)
- Flexible filtering by letter-position constraints
- Ranking and statistical analysis

Usage:
    # Analyze directory of individual files
    python moo_analyze.py output/layouts/ 
    
    # Analyze consolidated Pareto file
    python moo_analyze.py output/global_moo_solutions.csv
    
    # With constraints and options
    python moo_analyze.py data.csv --filter-assignments "e:J,t:F,a:S" --output-dir analysis_output
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
from pathlib import Path
from collections import defaultdict
import glob
import os
from io import StringIO
import time


class UnifiedMOOAnalyzer:
    """Unified analyzer for MOO results from any source format."""
    
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.df = None
        self.input_type = None
        self.objective_columns = []
        self.item_col = None
        self.pair_col = None
        
        # High-contrast color palette (excludes yellow, light colors)
        self.colors = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange  
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#17becf',  # Cyan
            '#bcbd22',  # Olive
            '#ff9896',  # Light Red
            '#c5b0d5',  # Light Purple
            '#c49c94',  # Light Brown
            '#f7b6d3',  # Light Pink
            '#dbdb8d',  # Light Olive
            '#9edae5',  # Light Cyan
            '#1a1a1a',  # Near Black
            '#800080',  # Purple
            '#008080',  # Teal
            '#800000'   # Maroon
        ]

    def get_colors(self, n_colors):
        """Get high-contrast colors, cycling if needed."""
        if n_colors <= len(self.colors):
            return self.colors[:n_colors]
        else:
            return [self.colors[i % len(self.colors)] for i in range(n_colors)]
        
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
                
                # Rename 'rank' to 'source_rank' to avoid confusion with global rankings
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
        """Load single consolidated CSV file (like global Pareto results)."""
        print(f"Loading consolidated MOO results file: {filepath}")
        
        try:
            # Try to detect if this is a global Pareto file with metadata header
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Look for data start (after metadata header)
            data_start_idx = 0
            for i, line in enumerate(lines):
                if ('config_id' in line or 'items' in line or 'positions' in line or 
                    'Complete Item' in line or 'Complete Pair' in line):
                    data_start_idx = i
                    break
            
            # If we found a header section, parse from there
            if data_start_idx > 0:
                print(f"Detected metadata header, parsing from line {data_start_idx + 1}")
                data_lines = ''.join(lines[data_start_idx:])
                df = pd.read_csv(StringIO(data_lines))
            else:
                # Standard CSV file
                df = pd.read_csv(filepath)
            
            print(f"Loaded {len(df):,} solutions from consolidated file")
            return df
            
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return pd.DataFrame()
    
    def detect_objective_columns(self, df):
        """Detect objective columns and special item/pair columns."""
        # Comprehensive metadata columns to exclude
        metadata_cols = [
            'config_id', 'rank', 'source_rank', 'items', 'positions', 'layout', 
            'combined_score', 'source_file', 'global_rank', 'item_rank', 'pair_rank',
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
                # Check if column contains numeric data
                try:
                    pd.to_numeric(df[col], errors='raise')
                    potential_objectives.append(col)
                except (ValueError, TypeError):
                    # Skip non-numeric columns
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
    
    def plot_objective_scatter(self):
        """Create scatter plot of objective scores."""
        if not self.objective_columns:
            return
            
        plt.figure(figsize=(14, 8))
        
        # Sort by global rank or combined score
        sort_col = 'global_rank' if 'global_rank' in self.df.columns else 'combined_score'
        if sort_col in self.df.columns:
            df_sorted = self.df.sort_values(sort_col, ascending=False if sort_col == 'combined_score' else True)
        else:
            df_sorted = self.df
        
        colors = self.get_colors(len(self.objective_columns))
        x_positions = range(len(df_sorted))
        
        for i, obj_col in enumerate(self.objective_columns):
            plt.scatter(x_positions, df_sorted[obj_col], 
                    marker='.', s=1, alpha=0.7, 
                    label=obj_col, color=colors[i])
        
        plt.xlabel('Solution Index (sorted by ranking)')
        plt.ylabel('Objective Score')
        plt.title(f'Multi-Objective Scores ({len(df_sorted):,} solutions)')
        
        # Fixed legend creation - remove problematic parameters
        try:
            legend = plt.legend(markerscale=10, frameon=True)
            if legend:
                legend.get_frame().set_alpha(0.9)
        except Exception as e:
            print(f"Warning: Could not create legend: {e}")
            
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'objective_scores_scatter.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pareto_front_2d(self):
        """Create 2D Pareto front if we have item/pair objectives."""
        if not (self.item_col and self.pair_col):
            return
            
        has_source_files = 'source_file' in self.df.columns
        
        plt.figure(figsize=(12, 8))
        
        if has_source_files:
            # Color by source file
            unique_sources = self.df['source_file'].unique()
            colors = self.get_colors(len(unique_sources))
            source_color_map = dict(zip(unique_sources, colors))
            
            for source in unique_sources:
                source_data = self.df[self.df['source_file'] == source]
                plt.scatter(source_data[self.item_col], source_data[self.pair_col], 
                        alpha=0.6, s=50, label=f'{source} ({len(source_data)})',
                        color=source_color_map[source])
            
            # Fixed legend positioning
            try:
                legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                                frameon=True)
                if legend:
                    legend.get_frame().set_alpha(0.9)
            except Exception as e:
                print(f"Warning: Could not create legend: {e}")
                plt.legend()  # Fallback to default legend
                
            title_suffix = " - Colored by Source"
        else:
            plt.scatter(self.df[self.item_col], self.df[self.pair_col], 
                    alpha=0.6, s=50, color='blue')
            title_suffix = ""
        
        plt.xlabel(self.item_col)
        plt.ylabel(self.pair_col)
        plt.title(f'Pareto Front{title_suffix} ({len(self.df):,} solutions)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'pareto_front_2d{"_colored" if has_source_files else ""}.png', 
                dpi=300, bbox_inches='tight')
        plt.show()
            
    def add_global_rankings(self, df):
        """Add global ranking columns based on objectives."""
        if self.item_col and self.pair_col:
            # For Pareto analysis with specific item/pair objectives
            df['item_rank'] = df[self.item_col].rank(ascending=False, method='first').astype(int)
            df['pair_rank'] = df[self.pair_col].rank(ascending=False, method='first').astype(int)
            df['global_rank'] = df['item_rank'] + df['pair_rank']
            print(f"Added global rankings based on {self.item_col} and {self.pair_col}")
            
        elif 'combined_score' in df.columns:
            # Use combined score if available
            df['global_rank'] = df['combined_score'].rank(ascending=False, method='first').astype(int)
            print("Added global ranking based on combined_score")
            
        elif len(self.objective_columns) >= 2:
            # Use first two objectives as proxy
            obj1, obj2 = self.objective_columns[0], self.objective_columns[1]
            df['obj1_rank'] = df[obj1].rank(ascending=False, method='first').astype(int)
            df['obj2_rank'] = df[obj2].rank(ascending=False, method='first').astype(int)
            df['global_rank'] = df['obj1_rank'] + df['obj2_rank']
            print(f"Added global ranking based on {obj1} and {obj2}")
        
        # Sort by global rank if it exists
        if 'global_rank' in df.columns:
            df = df.sort_values('global_rank').reset_index(drop=True)
            
        return df
    
    def load_data(self, input_path, **kwargs):
        """Load data from either directory or file input."""
        self.input_type = self.detect_input_type(input_path)
        
        if self.input_type == "directory":
            self.df = self.load_directory_input(input_path, **kwargs)
        else:
            self.df = self.load_file_input(input_path)
        
        if self.df.empty:
            raise ValueError("No data loaded successfully")
        
        # Detect objectives
        self.objective_columns, self.item_col, self.pair_col = self.detect_objective_columns(self.df)
        
        if not self.objective_columns:
            print("Warning: No objective columns detected")
            print(f"Available columns: {list(self.df.columns)}")
        else:
            print(f"Detected objective columns: {self.objective_columns}")
            if self.item_col and self.pair_col:
                print(f"Detected item/pair columns: {self.item_col}, {self.pair_col}")
        
        # Add rankings
        self.df = self.add_global_rankings(self.df)
        
        return self.df
    
    def apply_constraint_filter(self, constraints_string):
        """Apply letter-position constraints to filter solutions."""
        if not constraints_string:
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
    
    def plot_objective_scatter(self):
        """Create scatter plot of objective scores."""
        if not self.objective_columns:
            return
            
        plt.figure(figsize=(14, 8))
        
        # Sort by global rank or combined score
        sort_col = 'global_rank' if 'global_rank' in self.df.columns else 'combined_score'
        if sort_col in self.df.columns:
            df_sorted = self.df.sort_values(sort_col, ascending=False if sort_col == 'combined_score' else True)
        else:
            df_sorted = self.df
        
        colors = self.get_colors(len(self.objective_columns))
        x_positions = range(len(df_sorted))
        
        for i, obj_col in enumerate(self.objective_columns):
            plt.scatter(x_positions, df_sorted[obj_col], 
                       marker='.', s=1, alpha=0.7, 
                       label=obj_col, color=colors[i])
        
        plt.xlabel('Solution Index (sorted by ranking)')
        plt.ylabel('Objective Score')
        plt.title(f'Multi-Objective Scores ({len(df_sorted):,} solutions)')
        # Create legend with larger markers
        legend = plt.legend(markerscale=20, frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_alpha(0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'objective_scores_scatter.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pareto_front_2d(self):
        """Create 2D Pareto front if we have item/pair objectives."""
        if not (self.item_col and self.pair_col):
            return
            
        has_source_files = 'source_file' in self.df.columns
        
        plt.figure(figsize=(12, 8))
        
        if has_source_files:
            # Color by source file
            unique_sources = self.df['source_file'].unique()
            colors = self.get_colors(len(unique_sources))
            source_color_map = dict(zip(unique_sources, colors))
            
            for source in unique_sources:
                source_data = self.df[self.df['source_file'] == source]
                plt.scatter(source_data[self.item_col], source_data[self.pair_col], 
                           alpha=0.6, s=50, label=f'{source} ({len(source_data)})',
                           color=source_color_map[source])
            
            legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                            markerscale=3, frameon=True, fancybox=True, shadow=True)
            legend.get_frame().set_alpha(0.9)
            title_suffix = " - Colored by Source"
        else:
            plt.scatter(self.df[self.item_col], self.df[self.pair_col], 
                       alpha=0.6, s=50, color='blue')
            title_suffix = ""
        
        plt.xlabel(self.item_col)
        plt.ylabel(self.pair_col)
        plt.title(f'Pareto Front{title_suffix} ({len(self.df):,} solutions)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'pareto_front_2d{"_colored" if has_source_files else ""}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
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
        plt.savefig(self.output_dir / 'objective_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
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
        plt.savefig(self.output_dir / 'stability_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print stability stats
        print(f"\nMost stable letters (fewest positions):")
        for letter in letters_by_stability[:8]:
            print(f"  {letter}: {len(letter_positions[letter])} unique positions")
        
        print(f"Most stable positions (fewest letters):")
        for position in positions_by_stability[:8]:
            print(f"  {position}: {len(position_letters[position])} unique letters")
        
        # Calculate sparsity
        total_cells = len(letters_by_stability) * len(positions_by_stability)
        non_zero_cells = np.count_nonzero(matrix)
        sparsity = (total_cells - non_zero_cells) / total_cells * 100
        print(f"Matrix sparsity: {sparsity:.1f}% ({non_zero_cells}/{total_cells} cells with assignments)")
    
    def plot_source_distribution(self):
        """Plot distribution of solutions by source file."""
        if 'source_file' not in self.df.columns:
            return
            
        file_counts = self.df['source_file'].value_counts()
        
        plt.figure(figsize=(12, 6))
        top_sources = file_counts.head(20)
        plt.bar(range(len(top_sources)), top_sources.values, color=self.colors[0])
        plt.xlabel('Source File Rank')
        plt.ylabel('Number of Solutions')
        plt.title(f'Distribution of Solutions by Source File (Top 20)')
        plt.xticks(range(len(top_sources)), [f'#{i+1}' for i in range(len(top_sources))])
        plt.tight_layout()
        plt.savefig(self.output_dir / 'source_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return file_counts
    
    def print_analysis_summary(self):
        """Print comprehensive analysis summary."""
        print(f"\n=== MOO Analysis Summary ===")
        print(f"Input type: {self.input_type}")
        print(f"Total solutions: {len(self.df):,}")
        
        if 'source_file' in self.df.columns:
            file_counts = self.df['source_file'].value_counts()
            print(f"Number of source files: {len(file_counts)}")
            print(f"Most productive source: {file_counts.index[0]} ({file_counts.iloc[0]} solutions)")
        
        print(f"\nObjective Statistics:")
        for obj_col in self.objective_columns:
            values = self.df[obj_col].dropna()
            print(f"  {obj_col}: {values.min():.6f} to {values.max():.6f} (mean: {values.mean():.6f})")
        
        # Top solutions
        rank_col = 'global_rank' if 'global_rank' in self.df.columns else None
        if rank_col:
            print(f"\nTop 10 Solutions by {rank_col}:")
            top_10 = self.df.head(10)
        else:
            # Sort by first objective
            sort_col = self.objective_columns[0] if self.objective_columns else 'combined_score'
            if sort_col in self.df.columns:
                top_10 = self.df.nlargest(10, sort_col)
                print(f"\nTop 10 Solutions by {sort_col}:")
            else:
                top_10 = self.df.head(10)
                print(f"\nFirst 10 Solutions:")
        
        # Display columns
        display_cols = []
        if rank_col and rank_col in self.df.columns:
            display_cols.append(rank_col)
        display_cols.extend([col for col in self.objective_columns[:3]])  # Limit to first 3 objectives
        if 'positions' in self.df.columns:
            display_cols.append('positions')
        
        available_cols = [col for col in display_cols if col in self.df.columns]
        print(top_10[available_cols].to_string(index=False))
        
        return top_10
    
    def save_results(self, top_solutions):
        """Save analysis results to files."""
        # Save dataset with rankings
        output_csv = self.output_dir / 'moo_analysis_results.csv'
        self.df.to_csv(output_csv, index=False)
        print(f"\nComplete results saved to: {output_csv}")
        
        # Save summary
        with open(self.output_dir / 'analysis_summary.txt', 'w') as f:
            f.write(f"MOO Analysis Summary\n")
            f.write(f"===================\n\n")
            f.write(f"Input type: {self.input_type}\n")
            f.write(f"Total solutions: {len(self.df):,}\n")
            if 'source_file' in self.df.columns:
                file_counts = self.df['source_file'].value_counts()
                f.write(f"Number of source files: {len(file_counts)}\n")
            f.write(f"\nTop 10 Solutions:\n")
            f.write(top_solutions.to_string(index=False))
    
    def run_complete_analysis(self, input_path, filter_assignments=None, **load_kwargs):
        """Run the complete analysis pipeline."""
        print(f"Starting MOO analysis for: {input_path}")
        
        # Load data
        self.load_data(input_path, **load_kwargs)
        
        # Apply constraints if provided
        if filter_assignments:
            self.apply_constraint_filter(filter_assignments)
        
        # Generate all visualizations
        print("\nGenerating visualizations...")
        self.plot_objective_scatter()
        self.plot_pareto_front_2d()
        self.plot_correlation_matrix()
        self.plot_stability_matrix()
        source_counts = self.plot_source_distribution()
        
        # Analysis and summary
        top_solutions = self.print_analysis_summary()
        self.save_results(top_solutions)
        
        return self.df


def main():
    parser = argparse.ArgumentParser(description='Unified MOO Analysis Tool')
    parser.add_argument('input_path', 
                       help='Input path: directory with CSV files OR single consolidated CSV file')
    parser.add_argument('--output-dir', default='output',
                       help='Output directory for plots and results')
    parser.add_argument('--filter-assignments', 
                       help='Filter by letter-position assignments: "letter:position,letter:position"')
    
    # Options for directory input
    parser.add_argument('--file-pattern', default='moo_results_*.csv',
                       help='File pattern for directory input (default: moo_results_*.csv)')
    parser.add_argument('--max-files', type=int,
                       help='Maximum files to process for directory input')
    
    args = parser.parse_args()
    
    try:
        analyzer = UnifiedMOOAnalyzer(args.output_dir)
        
        # Prepare load kwargs for directory input
        load_kwargs = {}
        if Path(args.input_path).is_dir():
            load_kwargs['file_pattern'] = args.file_pattern
            if args.max_files:
                load_kwargs['max_files'] = args.max_files
        
        # Run analysis
        results_df = analyzer.run_complete_analysis(
            args.input_path, 
            filter_assignments=args.filter_assignments,
            **load_kwargs
        )
        
        print(f"\n✓ Analysis complete! Results saved to {args.output_dir}/")
        print(f"Key outputs:")
        print(f"  - moo_analysis_results.csv: Complete dataset with rankings")
        print(f"  - objective_scores_scatter.png: Objective score trends")
        print(f"  - objective_correlation_matrix.png: Objective correlations")
        print(f"  - stability_matrix.png: Letter-position stability heatmap")
        if analyzer.item_col and analyzer.pair_col:
            print(f"  - pareto_front_2d.png: 2D Pareto front visualization")
        if 'source_file' in results_df.columns:
            print(f"  - source_distribution.png: Source file contribution analysis")
        print(f"  - analysis_summary.txt: Text summary")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
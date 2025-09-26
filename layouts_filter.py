#!/usr/bin/env python3
"""
Filter MOO keyboard layout results using intersection filtering approach.

This script helps reduce large sets of MOO solutions to manageable subsets for further analysis.
It finds layouts that perform in the top percentage for ALL objectives simultaneously.

Input/Output Format:
- items: letters in assignment order (e.g., "etaoinsrhldcum") 
- positions: QWERTY positions where those letters go (e.g., "KJ;ASDVRLFUEIM")
- layout_qwerty: layout string in 32-key QWERTY order (QWERTYUIOPASDFGHJKL;ZXCVBNM,./[')
  with spaces for unassigned positions, such as: "  cr  du  oinl  teha   s  m     "

Usage:
    # Intersection filtering (layouts that score well in ALL objectives)
    python layouts_filter.py --input moo_analysis_results.csv --top-percent 10 --report --plot

    # Filter with higher threshold for more results  
    python layouts_filter.py --input results.csv --top-percent 75 --save-removed --verbose

    # Used in study
    poetry run python3 layouts_filter.py --input output/global_moo_solutions.csv --top-percent 75 --save-removed --plot --report --verbose

"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List
import sys
import matplotlib.pyplot as plt
import seaborn as sns
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

class MOOLayoutFilter:
    """MOO Layout Filter using intersection filtering approach."""
    
    def __init__(self, input_file: str, verbose: bool = False):
        self.input_file = input_file
        self.verbose = verbose
        self.df = self._load_data()
        self.objective_columns = self._detect_objectives()
        self.filtering_stats = {}  # Track filtering statistics
        
    def _load_data(self) -> pd.DataFrame:
        """Load MOO results data with format auto-detection and enhanced validation."""
        df = pd.read_csv(self.input_file)
        
        # Validate layout data if available
        if 'items' in df.columns and 'positions' in df.columns:
            valid_mask = df.apply(lambda row: validate_layout_data(row['items'], row['positions']), axis=1)
            invalid_count = len(df) - valid_mask.sum()
            
            if invalid_count > 0:
                if self.verbose:
                    print(f"Filtered out {invalid_count} rows with invalid layout data")
                df = df[valid_mask].copy()
        
        if self.verbose:
            print(f"Loaded {len(df)} valid layouts from {self.input_file}")
        
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
    
        # Add layout_qwerty column if items and positions exist
        if 'items' in filtered_df.columns and 'positions' in filtered_df.columns and 'layout_qwerty' not in filtered_df.columns:
            if self.verbose:
                print("Adding layout_qwerty column to filtered results...")
            filtered_df['layout_qwerty'] = filtered_df.apply(
                lambda row: convert_items_positions_to_qwerty_layout(row['items'], row['positions']), 
                axis=1
            )
        
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
            print(f"\nWarning: Only {len(intersection_layouts)} layouts survived intersection filtering!")
            print(f"    With {len(self.objective_columns)} objectives and {top_percent}% threshold,")
            expected_rate = (top_percent/100) ** len(self.objective_columns) * 100
            print(f"    mathematical expectation: ~{expected_rate:.2f}% retention rate")
            print(f"    actual retention: {len(intersection_layouts)/total_layouts*100:.2f}%")
            if len(intersection_layouts) == 0:
                print(f"    This means NO layouts are simultaneously in top {top_percent}% of ALL objectives")
            else:
                print(f"    This suggests layouts with identical/near-identical objective values")
            print(f"    Consider using a higher --top-percent (e.g., 70-90%).")

        return filtered_df

    def standardize_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column ordering for output compatibility."""
        if len(df) == 0:
            return df
        
        # Standard column order
        standard_cols = ['config_id', 'items', 'positions', 'layout_qwerty']
        
        # Identify objective columns
        metadata_cols = ['config_id', 'items', 'positions', 'layout_qwerty', 'source_file', 'layout']
        objective_cols = sorted([col for col in df.columns if col not in metadata_cols and 
                                col not in standard_cols])
        
        # Build ordered column list
        ordered_cols = []
        
        # Add standard columns that exist
        for col in standard_cols:
            if col in df.columns:
                ordered_cols.append(col)
        
        # Add objective columns
        ordered_cols.extend(objective_cols)
        
        # Add any remaining columns
        remaining = [col for col in df.columns if col not in ordered_cols]
        ordered_cols.extend(remaining)
        
        return df[ordered_cols]

    def generate_statistical_report(self, filtered_df: pd.DataFrame, output_dir: str = '.') -> str:
        """Generate a statistical report comparing filtered vs original datasets."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = Path(output_dir) / f'filtering_report_{timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write("MOO Layout Filtering Statistical Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {self.input_file}\n")
            f.write(f"Filtering method: intersection\n\n")
            
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
                    max_loss_pct = max_loss / orig_stats['max'] * 100 if orig_stats['max'] > 0 else 0
                    f.write(f"  Maximum value lost: {max_loss:.4f} ({max_loss_pct:.1f}%)\n")
        
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
                improvement = (filt_mean - orig_mean) / orig_mean * 100 if orig_mean != 0 else 0
                
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
        
        # Sort by index (no ranking references)
        df_sorted = self.df.sort_index()
        
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
                           marker='.', s=1, alpha=0.3, color='lightgray', 
                           label=f'{obj_col} (removed)' if i == 0 else "")
            
            # Plot retained points
            if retained_mask.any():
                plt.scatter(np.array(x_positions)[retained_mask], df_sorted[obj_col][retained_mask], 
                           marker='.', s=2, alpha=0.8, color=colors[i], label=f'{obj_col} (retained)')
        
        plt.xlabel('Solution Index')
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

    # Filtering method options
    parser.add_argument('--top-percent', type=float, default=25.0,
                       help='Top percentage to keep for intersection filtering')
    
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
        filter_tool = MOOLayoutFilter(args.input, args.verbose)
        original_df = filter_tool.df.copy()  # Keep original for removed layouts calculation
        
        # Apply filtering method
        filtered_df = filter_tool.filter_intersection(args.top_percent)
                                        
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
            # Generate both filenames based on input filename
            input_stem = Path(args.input).stem
            output_file = f"output/filtered_{input_stem}_intersection.csv"
            removed_file = f"output/removed_{input_stem}_intersection.csv"
        
        # Save filtered results
        standardized_df = filter_tool.standardize_output_columns(filtered_df)
        standardized_df.to_csv(output_file, index=False)
        
        # Save removed results if requested
        if args.save_removed and len(removed_df) > 0:
            standardized_removed = filter_tool.standardize_output_columns(removed_df)
            standardized_removed.to_csv(removed_file, index=False)
            print(f"Removed layouts saved to: {removed_file}")
        
        print(f"\nFiltering complete!")
        print(f"Input: {len(original_df)} layouts")
        print(f"Filtered (kept): {len(filtered_df)} layouts ({len(filtered_df)/len(original_df)*100:.1f}%)")
        print(f"Removed: {len(removed_df)} layouts ({len(removed_df)/len(original_df)*100:.1f}%)")
        print(f"Filtered results saved to: {output_file}")
        
        # Add summary of removal reasons for intersection method
        if args.verbose:
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
        if args.report:
            report_path = filter_tool.generate_statistical_report(filtered_df, args.output_dir)
            print(f"Statistical report: {report_path}")
        
        # Generate visualization if requested
        if args.plot:
            plot_path = filter_tool.plot_filtering_results(filtered_df, args.output_dir)
            print(f"Filtering visualization: {plot_path}")
            
            # Generate filtered objective scatter plot
            scatter_path = filter_tool.plot_filtered_objective_scatter(filtered_df, args.output_dir)
            print(f"Filtered objective scatter: {scatter_path}")
        
        # Print summary statistics
        if args.verbose and len(filtered_df) > 0:
            print(f"\nRetained dataset summary:")
            for col in filter_tool.objective_columns:
                if col in filtered_df.columns:
                    orig_mean = original_df[col].mean()
                    filt_mean = filtered_df[col].mean()
                    filt_std = filtered_df[col].std()
                    improvement = (filt_mean - orig_mean) / orig_mean * 100 if orig_mean != 0 else 0
                    print(f"  {col}: {filtered_df[col].min():.4f} - {filtered_df[col].max():.4f} (mean: {filt_mean:.4f}, std: {filt_std:.4f}, +{improvement:.1f}%)")
                    
                    # Warning for zero standard deviation
                    if filt_std < 1e-10:
                        print(f"    Warning: {col} has zero variance - all retained layouts are identical for this objective!")
            
            # Also show removed dataset summary
            if args.save_removed and len(removed_df) > 0:
                print(f"\nRemoved dataset summary:")
                for col in filter_tool.objective_columns:
                    if col in removed_df.columns:
                        removed_mean = removed_df[col].mean()
                        removed_std = removed_df[col].std()
                        orig_mean = original_df[col].mean()
                        degradation = (removed_mean - orig_mean) / orig_mean * 100 if orig_mean != 0 else 0
                        print(f"  {col}: {removed_df[col].min():.4f} - {removed_df[col].max():.4f} (mean: {removed_mean:.4f}, std: {removed_std:.4f}, {degradation:.1f}%)")
        
        # Enhanced quality check with warnings
        print(f"\nQuality check:")

        # Check if we have enough layouts for meaningful analysis
        if len(filtered_df) < 5:
            print(f"Warning: Only {len(filtered_df)} layouts retained - may be insufficient for analysis")
            print(f"   Consider using higher --top-percent values")
        
        # Count how many layouts are in top 10% of ALL original objectives
        quality_count = len(filtered_df)
        for col in filter_tool.objective_columns:
            if col in filtered_df.columns:
                p90_threshold = np.percentile(original_df[col], 90)
                in_top10 = (filtered_df[col] >= p90_threshold).sum()
                quality_count = min(quality_count, in_top10)
        
        print(f"Layouts in top 10% of ALL objectives: {quality_count} ({quality_count/len(filtered_df)*100:.1f}% of retained set)")
        
        # Check for diversity
        total_unique_values = 0
        for col in filter_tool.objective_columns:
            if col in filtered_df.columns:
                unique_values = filtered_df[col].nunique()
                total_unique_values += unique_values
                if unique_values <= 1:
                    print(f"Warning: {col}: All retained layouts have identical values")
        
        if total_unique_values <= len(filter_tool.objective_columns):
            print(f"\nWarning: Retained layouts show little diversity across objectives")
            print(f"   This suggests filtering was too restrictive")
        
        # Success message with recommendations based on actual diversity
        has_diversity = total_unique_values > len(filter_tool.objective_columns)
        retention_rate = len(filtered_df) / len(original_df) * 100
        
        if len(filtered_df) >= 10 and has_diversity and retention_rate >= 1.0:
            print(f"\nFiltering successful! Retained {len(filtered_df)} diverse layouts ({retention_rate:.1f}%)")
        elif len(filtered_df) >= 10 and has_diversity:
            print(f"\nFiltering very restrictive but diverse: {len(filtered_df)} layouts ({retention_rate:.1f}%)")
            print(f"   Consider higher --top-percent values for more layouts")
        elif len(filtered_df) >= 10:
            print(f"\nFiltering retained {len(filtered_df)} layouts but they lack diversity")
            print(f"   All layouts are nearly identical - intersection method too restrictive")
        elif len(filtered_df) >= 2:
            print(f"\nFiltering completed but retained only {len(filtered_df)} layouts ({retention_rate:.2f}%)")
            print(f"   Consider using higher --top-percent values")
        else:
            print(f"\nFiltering too restrictive! Only {len(filtered_df)} layout(s) retained")
            print(f"   Recommended: use much higher --top-percent (70-90%)")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
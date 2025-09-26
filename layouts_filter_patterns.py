#!/usr/bin/env python3
"""
Filter keyboard layout results based on layout pattern matching.

This script allows filtering of keyboard layouts based on:
1. Regex patterns to exclude (filter out)
2. Regex patterns to include (retain only)
3. Specific letter-position constraints (e.g., 'etaio' should not be in positions 'A;R')

The layout_qwerty string follows QWERTY key order: "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"
Positions are indexed from 0, so 'A' is position 10, ';' is position 19, 'R' is position 13, etc.

Usage:
    # Filter out layouts starting with "ht"
    python layouts_filter_patterns.py --input data.csv --exclude "^ht"
    
    # Retain only layouts with 'e' in first 5 positions
    python layouts_filter_patterns.py --input data.csv --include "^.{0,4}e"
    
    # Exclude layouts where letters 'etaio' appear in positions A (10) or R (13)
    python layouts_filter_patterns.py --input data.csv \
        --forbidden-letters "etaio" \
        --forbidden-positions "AR"
    
    # Combine multiple filters
    python layouts_filter_patterns.py --input data.csv \
        --exclude "^ht" \
        --include "e.*a.*o" \
        --forbidden-letters "etaio" \
        --forbidden-positions "A;R"

    # Study
    poetry run python3 layouts_filter_patterns.py \
        --input output/analyze_phase1/layouts_consolidate_moo_solutions.csv \
        --exclude "^.{2}ht,^.{6}th" \
        --forbidden-letters "taoi" --forbidden-positions "A;" \
        --report --output output/layouts_filter_patterns.csv

    # Don't permit top five letters (etaoi) in pinkie positions (instead: nsrhldcum).
    # Don't permit 't' in top row index fingers.
    poetry run python3 layouts_filter_patterns.py \
        --input output/analyze_phase1/layouts_consolidate_moo_solutions.csv \
        --exclude "^.{3}t,^.{6}t" \
        --forbidden-letters "taoi" --forbidden-positions "A;" \
        --report --output output/layouts_filter_patterns.csv
        #--include "^.{10}m,^.{10}u,^.{10}c,^.{10}d,^.{10}l,^.{10}h,^.{10}r,^.{10}s,^.{10}n" \

        
"""

import pandas as pd
import numpy as np
import argparse
import re
from pathlib import Path
from typing import List, Set, Tuple
import sys
from datetime import datetime

class LayoutPatternFilter:
    """Filter keyboard layouts based on pattern matching and position constraints."""
    
    # QWERTY key order for layout_qwerty string
    QWERTY_ORDER = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"
    
    def __init__(self, input_file: str, verbose: bool = False):
        self.input_file = input_file
        self.verbose = verbose
        self.df = self._load_data()
        self.qwerty_pos_map = {char: i for i, char in enumerate(self.QWERTY_ORDER)}
        self.filtering_stats = {}
        
    def _load_data(self) -> pd.DataFrame:
        """Load CSV data."""
        df = pd.read_csv(self.input_file)
        
        if 'layout_qwerty' not in df.columns:
            raise ValueError("Input file must contain 'layout_qwerty' column")
        
        if self.verbose:
            print(f"Loaded {len(df)} layouts from {self.input_file}")
            
        return df
    
    def get_position_indices(self, position_chars: str) -> List[int]:
        """Convert position characters to indices in layout_qwerty string."""
        indices = []
        for char in position_chars:
            if char in self.qwerty_pos_map:
                indices.append(self.qwerty_pos_map[char])
            else:
                raise ValueError(f"Invalid position character: '{char}'. Must be one of: {self.QWERTY_ORDER}")
        return indices
    
    def check_forbidden_combinations(self, layout: str, forbidden_letters: str, forbidden_positions: str) -> bool:
        """
        Check if layout violates forbidden letter-position combinations.
        
        Returns:
            True if layout is valid (no forbidden combinations)
            False if layout violates constraints (has forbidden combinations)
        """
        forbidden_indices = self.get_position_indices(forbidden_positions)
        
        for idx in forbidden_indices:
            if idx < len(layout) and layout[idx] in forbidden_letters:
                return False  # Found forbidden combination
        
        return True  # No forbidden combinations found
    
    def apply_regex_filters(self, df: pd.DataFrame, exclude_patterns: List[str], include_patterns: List[str]) -> pd.DataFrame:
        """Apply regex filters to layout_qwerty column."""
        mask = pd.Series([True] * len(df), index=df.index)
        
        # Apply exclude patterns (filter out matching layouts)
        for pattern in exclude_patterns:
            try:
                regex = re.compile(pattern)
                exclude_mask = df['layout_qwerty'].str.match(regex, na=False)
                mask &= ~exclude_mask
                
                excluded_count = exclude_mask.sum()
                if self.verbose:
                    print(f"Exclude pattern '{pattern}': filtered out {excluded_count} layouts")
                    
            except re.error as e:
                print(f"Invalid regex pattern '{pattern}': {e}")
                return pd.DataFrame()  # Return empty DataFrame on regex error
        
        # Apply include patterns (keep only matching layouts)
        if include_patterns:
            include_mask = pd.Series([False] * len(df), index=df.index)
            
            for pattern in include_patterns:
                try:
                    regex = re.compile(pattern)
                    pattern_mask = df['layout_qwerty'].str.match(regex, na=False)
                    include_mask |= pattern_mask
                    
                    included_count = pattern_mask.sum()
                    if self.verbose:
                        print(f"Include pattern '{pattern}': matched {included_count} layouts")
                        
                except re.error as e:
                    print(f"Invalid regex pattern '{pattern}': {e}")
                    return pd.DataFrame()
            
            mask &= include_mask
            
            if self.verbose:
                total_included = include_mask.sum()
                print(f"Total layouts matching include patterns: {total_included}")
        
        return df[mask].copy()
    
    def apply_position_constraints(self, df: pd.DataFrame, forbidden_letters: str, forbidden_positions: str) -> pd.DataFrame:
        """Apply forbidden letter-position constraints."""
        if not forbidden_letters or not forbidden_positions:
            return df
        
        try:
            forbidden_indices = self.get_position_indices(forbidden_positions)
        except ValueError as e:
            print(f"Error in position constraints: {e}")
            return pd.DataFrame()
        
        valid_mask = df['layout_qwerty'].apply(
            lambda layout: self.check_forbidden_combinations(layout, forbidden_letters, forbidden_positions)
        )
        
        filtered_count = (~valid_mask).sum()
        if self.verbose:
            print(f"Position constraint ('{forbidden_letters}' not in positions '{forbidden_positions}'): filtered out {filtered_count} layouts")
            
            # Show some examples of filtered layouts if verbose
            if filtered_count > 0 and filtered_count <= 5:
                invalid_layouts = df[~valid_mask]['layout_qwerty'].head(5).tolist()
                print(f"Examples of filtered layouts: {invalid_layouts}")
        
        return df[valid_mask].copy()
    
    def filter_layouts(self, exclude_patterns: List[str] = None, include_patterns: List[str] = None,
                      forbidden_letters: str = None, forbidden_positions: str = None) -> pd.DataFrame:
        """Apply all filtering criteria."""
        exclude_patterns = exclude_patterns or []
        include_patterns = include_patterns or []
        
        original_count = len(self.df)
        filtered_df = self.df.copy()
        
        # Track filtering steps
        self.filtering_stats = {
            'original_count': original_count,
            'steps': []
        }
        
        # Apply regex filters
        if exclude_patterns or include_patterns:
            step_df = self.apply_regex_filters(filtered_df, exclude_patterns, include_patterns)
            step_count = len(step_df)
            self.filtering_stats['steps'].append({
                'step': 'regex_filters',
                'before': len(filtered_df),
                'after': step_count,
                'filtered': len(filtered_df) - step_count
            })
            filtered_df = step_df
        
        # Apply position constraints
        if forbidden_letters and forbidden_positions:
            step_df = self.apply_position_constraints(filtered_df, forbidden_letters, forbidden_positions)
            step_count = len(step_df)
            self.filtering_stats['steps'].append({
                'step': 'position_constraints',
                'before': len(filtered_df),
                'after': step_count,
                'filtered': len(filtered_df) - step_count
            })
            filtered_df = step_df
        
        self.filtering_stats['final_count'] = len(filtered_df)
        self.filtering_stats['total_filtered'] = original_count - len(filtered_df)
        self.filtering_stats['retention_rate'] = len(filtered_df) / original_count * 100 if original_count > 0 else 0
        
        return filtered_df
    
    def print_position_map(self):
        """Print the QWERTY position mapping for reference."""
        print("QWERTY Position Reference:")
        print("=" * 50)
        
        # Show the layout in QWERTY keyboard format
        rows = [
            self.QWERTY_ORDER[0:10],   # QWERTYUIOP
            self.QWERTY_ORDER[10:19],  # ASDFGHJKL;
            self.QWERTY_ORDER[19:32]   # ZXCVBNM,./[' 
        ]
        
        for i, row in enumerate(rows):
            positions = []
            chars = []
            for char in row:
                pos = self.qwerty_pos_map[char]
                positions.append(f"{pos:2d}")
                chars.append(f" {char}")
            
            print("Positions: " + " ".join(positions))
            print("Keys:      " + " ".join(chars))
            print()
        
        print("Examples:")
        print("  Position 'A' (index 10) = left pinky home row")
        print("  Position ';' (index 19) = right pinky home row")
        print("  Position 'R' (index 13) = left index finger top row")
    
    def generate_report(self, filtered_df: pd.DataFrame, output_dir: str = '.') -> str:
        """Generate filtering report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = Path(output_dir) / f'pattern_filtering_report_{timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write("Layout Pattern Filtering Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {self.input_file}\n\n")
            
            # Dataset sizes
            f.write("Dataset Sizes:\n")
            f.write(f"Original: {self.filtering_stats['original_count']} layouts\n")
            f.write(f"Filtered: {self.filtering_stats['final_count']} layouts\n")
            f.write(f"Removed: {self.filtering_stats['total_filtered']} layouts\n")
            f.write(f"Retention rate: {self.filtering_stats['retention_rate']:.2f}%\n\n")
            
            # Filtering steps
            if self.filtering_stats['steps']:
                f.write("Filtering Steps:\n")
                f.write("-" * 25 + "\n")
                for step in self.filtering_stats['steps']:
                    f.write(f"{step['step']}: {step['before']} -> {step['after']} "
                           f"(filtered: {step['filtered']})\n")
                f.write("\n")
            
            # Sample layouts
            if len(filtered_df) > 0:
                f.write("Sample Retained Layouts:\n")
                f.write("-" * 30 + "\n")
                sample_size = min(10, len(filtered_df))
                for i, (_, row) in enumerate(filtered_df.head(sample_size).iterrows()):
                    f.write(f"{i+1}. {row['layout_qwerty']}\n")
                if len(filtered_df) > sample_size:
                    f.write(f"... and {len(filtered_df) - sample_size} more\n")
        
        return str(report_path)

def parse_patterns(pattern_string: str) -> List[str]:
    """Parse comma-separated pattern string into list."""
    if not pattern_string:
        return []
    return [p.strip() for p in pattern_string.split(',') if p.strip()]

def main():
    parser = argparse.ArgumentParser(description='Filter keyboard layouts based on pattern matching')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', help='Output CSV file (filtered results)')
    
    # Pattern filtering options
    parser.add_argument('--exclude', type=str, 
                       help='Comma-separated regex patterns to exclude (filter out)')
    parser.add_argument('--include', type=str,
                       help='Comma-separated regex patterns to include (retain only)')
    
    # Position constraint options
    parser.add_argument('--forbidden-letters', type=str,
                       help='Letters that should not appear in forbidden positions (e.g., "etaio")')
    parser.add_argument('--forbidden-positions', type=str,
                       help='QWERTY position characters where forbidden letters should not appear (e.g., "A;R")')
    
    # Output options
    parser.add_argument('--output-dir', default='output',
                       help='Output directory for reports')
    parser.add_argument('--report', action='store_true',
                       help='Generate filtering report')
    parser.add_argument('--save-removed', action='store_true',
                       help='Save removed layouts to separate file')
    
    # Utility options
    parser.add_argument('--show-positions', action='store_true',
                       help='Show QWERTY position reference and exit')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Show position reference if requested
    if args.show_positions:
        # Create a dummy filter just to access the position mapping
        class DummyFilter:
            QWERTY_ORDER = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"
            def __init__(self):
                self.qwerty_pos_map = {char: i for i, char in enumerate(self.QWERTY_ORDER)}
            def print_position_map(self):
                print("QWERTY Position Reference:")
                print("=" * 50)
                rows = [
                    self.QWERTY_ORDER[0:10],   # QWERTYUIOP
                    self.QWERTY_ORDER[10:19],  # ASDFGHJKL;
                    self.QWERTY_ORDER[19:32]   # ZXCVBNM,./[' 
                ]
                for i, row in enumerate(rows):
                    positions = []
                    chars = []
                    for char in row:
                        pos = self.qwerty_pos_map[char]
                        positions.append(f"{pos:2d}")
                        chars.append(f" {char}")
                    print("Positions: " + " ".join(positions))
                    print("Keys:      " + " ".join(chars))
                    print()
                print("Examples:")
                print("  Position 'A' (index 10) = left pinky home row")
                print("  Position ';' (index 19) = right pinky home row")
                print("  Position 'R' (index 3) = left middle finger top row")
                print("  Position 'U' (index 6) = right index finger top row")
        
        dummy_filter = DummyFilter()
        dummy_filter.print_position_map()
        return 0
    
    try:
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize filter
        filter_tool = LayoutPatternFilter(args.input, args.verbose)
        
        # Parse patterns
        exclude_patterns = parse_patterns(args.exclude)
        include_patterns = parse_patterns(args.include)
        
        if args.verbose:
            if exclude_patterns:
                print(f"Exclude patterns: {exclude_patterns}")
            if include_patterns:
                print(f"Include patterns: {include_patterns}")
            if args.forbidden_letters and args.forbidden_positions:
                print(f"Forbidden: letters '{args.forbidden_letters}' in positions '{args.forbidden_positions}'")
        
        # Apply filtering
        filtered_df = filter_tool.filter_layouts(
            exclude_patterns=exclude_patterns,
            include_patterns=include_patterns,
            forbidden_letters=args.forbidden_letters,
            forbidden_positions=args.forbidden_positions
        )
        
        # Calculate removed layouts
        original_indices = set(filter_tool.df.index)
        filtered_indices = set(filtered_df.index)
        removed_indices = original_indices - filtered_indices
        removed_df = filter_tool.df.loc[list(removed_indices)].copy()
        
        # Generate output filename if not provided
        if not args.output:
            input_stem = Path(args.input).stem
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            args.output = f"{args.output_dir}/pattern_filtered_{input_stem}_{timestamp}.csv"
        
        # Save results
        filtered_df.to_csv(args.output, index=False)
        
        # Save removed layouts if requested
        if args.save_removed and len(removed_df) > 0:
            removed_path = Path(args.output).parent / f"removed_{Path(args.output).name}"
            removed_df.to_csv(removed_path, index=False)
            print(f"Removed layouts saved to: {removed_path}")
        
        # Print summary
        print(f"\nFiltering complete!")
        print(f"Input: {len(filter_tool.df)} layouts")
        print(f"Filtered (retained): {len(filtered_df)} layouts ({filter_tool.filtering_stats['retention_rate']:.1f}%)")
        print(f"Removed: {len(removed_df)} layouts")
        print(f"Results saved to: {args.output}")
        
        # Generate report if requested
        if args.report:
            report_path = filter_tool.generate_report(filtered_df, args.output_dir)
            print(f"Report saved to: {report_path}")
        
        # Show sample results
        if args.verbose and len(filtered_df) > 0:
            print(f"\nSample retained layouts:")
            for i, layout in enumerate(filtered_df['layout_qwerty'].head(5)):
                print(f"  {i+1}. {layout}")
            if len(filtered_df) > 5:
                print(f"  ... and {len(filtered_df) - 5} more")
        
        # Warnings
        if len(filtered_df) == 0:
            print(f"\nWarning: No layouts matched the filtering criteria!")
            print(f"Consider relaxing the constraints.")
        elif len(filtered_df) < 10:
            print(f"\nNote: Only {len(filtered_df)} layouts retained. Consider broader criteria if more layouts needed.")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
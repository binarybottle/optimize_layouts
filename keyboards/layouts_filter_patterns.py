#!/usr/bin/env python3
"""
Filter keyboard layout results based on layout pattern matching.

This script allows filtering of keyboard layouts based on:
1. Regex patterns to exclude (filter out)
2. Regex patterns to include (retain only)
3. Specific letter-position constraints (e.g., 'etaio' should not be in positions 'A;R')
4. Vertical bigrams (same column, adjacent rows)
5. Hurdle bigrams (top-to-bottom or bottom-to-top on same hand, any columns)

Vertical bigrams: Two letters in the same column on adjacent rows (e.g., q-a, a-z, w-s)
Hurdle bigrams: Two letters on the same hand where one is on top row and other is on bottom row,
                regardless of column (e.g., q-z, q-x, w-c, e-v on left; y-n, u-, on right)
                This creates a "hurdle" motion over the home row.

The layout_qwerty string follows QWERTY key order: "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"
Positions are indexed from 0, so 'A' is position 10, ';' is position 19, 'R' is position 13, etc.

    # Indices:
    ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
    │   │ 1 │ 2 │ 3 │   │   │ 6 │ 7 │ 8 │   │   │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
    │   │11 │12 │13 │   │   │16 │17 │18 │   │   │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤───┘
    │   │   │   │23 │   │   │26 │   │   │   │
    └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘    

Usage:
    # Filter out layouts starting with "ht":
    python layouts_filter_patterns.py --input data.csv --exclude "^ht"
    
    # Retain only layouts with 'e' in first 5 positions:
    python layouts_filter_patterns.py --input data.csv --include "^.{0,4}e"
    
    # Exclude layouts where letters 'etaio' appear in positions A (10) or R (13):
    python layouts_filter_patterns.py --input data.csv \
        --forbidden-letters "etaio" \
        --forbidden-positions "AR"
    
    # Combine multiple filters:
    python layouts_filter_patterns.py --input data.csv \
        --exclude "^ht" \
        --include "e.*a.*o" \
        --forbidden-letters "etaio" \
        --forbidden-positions "A;R"

    # Remove rows/layouts that have any of the top eight keys empty.
    python layouts_filter_patterns.py \
        --input ../output/layouts_consolidate_moo_solutions.csv \
        --output ../output/layouts_filter_empty_spaces.csv --report \
        --exclude "^.{2}[ ],^.{7}[ ],^.{11}[ ],^.{12}[ ],^.{13}[ ],^.{16}[ ],^.{17}[ ],^.{18}[ ]"

    # Include only layouts with z,q,j,x in last two positions:
    poetry run python3 layouts_filter_patterns.py \
        --input ../output/layouts_consolidate_moo_solutions.csv \
        --output ../output/layouts_filter_patterns_zqjx.csv --report \
        --exclude-vertical-bigrams "$BIGRAMS" \
        --exclude-hurdles "$HURDLE_BIGRAMS" \
        --include "^.{30}[zqjxZQJX][zqjxZQJX]"

    # Keyboard layout optimization study command:
    # Don't permit common bigrams to be stacked vertically or to require hurdles.
    #   Up to 25% (cumulative fraction 0.249612493 at "or")
    BIGRAMS25="th,he,in,er,an,re,on,at,en,nd,ti,es,or"
    #   25% to 50% (cumulative fraction 0.497485843 at "ea")
    BIGRAMS50="te,of,ed,is,it,al,ar,st,to,nt,ng,se,ha,as,ou,io,le,ve,co,me,de,hi,ri,ro,ic,ne,ea"
    #   50% to 75% (cumulative fraction 0.727855962 at "na")
    BIGRAMS75="ra,ce,li,ch,ll,be,ma,si,om,ur,ca,el,ta,la,ns,di,fo,ho,pe,ec,pr,no,ct,us,ac,ot,il,tr,ly,nc,et,ut,ss,so,rs,un,lo,wa,ge,ie,wh,ee,wi,em,ad,ol,rt,po,we,na"
    BIGRAMS="$BIGRAMS25,$BIGRAMS50"
    HURDLE_BIGRAMS="$BIGRAMS25,$BIGRAMS50"
    poetry run python3 layouts_filter_patterns.py \
        --input ../output/layouts_consolidate_moo_solutions.csv \
        --output ../output/layouts_filter_patterns.csv --report \
        --exclude-vertical-bigrams "$BIGRAMS" \
        --exclude-hurdles "$HURDLE_BIGRAMS"

        







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

    def generate_vertical_exclusion_patterns(self, bigrams_str: str) -> List[str]:
        """
        Generate exclusion patterns for bigrams that should not be vertically stacked.
        Checks same-column placement across all row pairs: top-home, home-bottom, and top-bottom.
        
        Args:
            bigrams_str: Comma-separated string of bigrams (e.g., "th,he,er")
        
        Returns:
            List of regex patterns to exclude vertical bigram placements
        """
        if not bigrams_str:
            return []
        
        bigrams = [b.strip() for b in bigrams_str.split(',') if b.strip()]
        patterns = []
        
        for bigram in bigrams:
            if len(bigram) != 2:
                continue
            
            char1, char2 = bigram[0].lower(), bigram[1].lower()
            
            # Top row to home row stacking (positions 0-9 to 10-19)
            for i in range(10):
                top_pos = i
                home_pos = i + 10
                
                # char1 on top, char2 on home row
                if top_pos == 0:
                    pattern1 = f"^{char1}.{{{home_pos-1}}}{char2}"
                else:
                    pattern1 = f"^.{{{top_pos}}}{char1}.{{{home_pos-top_pos-1}}}{char2}"
                
                # char2 on top, char1 on home row  
                if top_pos == 0:
                    pattern2 = f"^{char2}.{{{home_pos-1}}}{char1}"
                else:
                    pattern2 = f"^.{{{top_pos}}}{char2}.{{{home_pos-top_pos-1}}}{char1}"
                
                patterns.extend([pattern1, pattern2])
            
            # Home row to bottom row stacking (positions 10-19 to 20-29)
            for i in range(10):
                home_pos = i + 10
                bottom_pos = i + 20
                
                if bottom_pos >= len(self.QWERTY_ORDER):
                    continue
                    
                # char1 on home, char2 on bottom
                pattern3 = f"^.{{{home_pos}}}{char1}.{{{bottom_pos-home_pos-1}}}{char2}"
                
                # char2 on home, char1 on bottom
                pattern4 = f"^.{{{home_pos}}}{char2}.{{{bottom_pos-home_pos-1}}}{char1}"
                
                patterns.extend([pattern3, pattern4])
            
            # Top row to bottom row stacking (positions 0-9 to 20-29) - NEW!
            for i in range(10):
                top_pos = i
                bottom_pos = i + 20
                
                if bottom_pos >= len(self.QWERTY_ORDER):
                    continue
                
                # char1 on top, char2 on bottom
                if top_pos == 0:
                    pattern5 = f"^{char1}.{{{bottom_pos-1}}}{char2}"
                else:
                    pattern5 = f"^.{{{top_pos}}}{char1}.{{{bottom_pos-top_pos-1}}}{char2}"
                
                # char2 on top, char1 on bottom
                if top_pos == 0:
                    pattern6 = f"^{char2}.{{{bottom_pos-1}}}{char1}"
                else:
                    pattern6 = f"^.{{{top_pos}}}{char2}.{{{bottom_pos-top_pos-1}}}{char1}"
                
                patterns.extend([pattern5, pattern6])
        
        if self.verbose:
            print(f"Generated {len(patterns)} vertical exclusion patterns for {len(bigrams)} bigrams (including non-adjacent rows)")
        
        return patterns

    def generate_hurdle_exclusion_patterns(self, bigrams_str: str) -> List[str]:
            """
            Generate exclusion patterns for bigrams that create top-to-bottom movements on same hand.
            Excludes any bigram where one letter is on the top row and the other is on the bottom row
            of the same hand, regardless of column (creates a "hurdle" over the home row).
            
            Args:
                bigrams_str: Comma-separated string of bigrams (e.g., "th,he,qz,wc")
            
            Returns:
                List of regex patterns to exclude hurdle bigram placements
            """
            if not bigrams_str:
                return []
            
            bigrams = [b.strip() for b in bigrams_str.split(',') if b.strip()]
            patterns = []
            
            # Define hand positions
            # Left hand: top=0-4 (QWERT), home=10-14 (ASDFG), bottom=20-24 (ZXCVB)
            # Right hand: top=5-9 (YUIOP), home=15-19 (HJKLñ), bottom=25-29 (NM,./)
            left_top = list(range(0, 5))      # Q, W, E, R, T
            left_bottom = list(range(20, 25))  # Z, X, C, V, B
            
            right_top = list(range(5, 10))     # Y, U, I, O, P
            right_bottom = list(range(25, 30)) # N, M, ,, ., /
            
            # Filter bottom positions that exist in layout
            left_bottom = [p for p in left_bottom if p < len(self.QWERTY_ORDER)]
            right_bottom = [p for p in right_bottom if p < len(self.QWERTY_ORDER)]
            
            for bigram in bigrams:
                if len(bigram) != 2:
                    continue
                
                char1, char2 = bigram[0].lower(), bigram[1].lower()
                
                # Left hand: char1 on top, char2 on bottom (any columns)
                for top_pos in left_top:
                    for bottom_pos in left_bottom:
                        if top_pos == 0:
                            pattern = f"^{char1}.{{{bottom_pos-1}}}{char2}"
                        else:
                            pattern = f"^.{{{top_pos}}}{char1}.{{{bottom_pos-top_pos-1}}}{char2}"
                        patterns.append(pattern)
                
                # Left hand: char2 on top, char1 on bottom (any columns)
                for top_pos in left_top:
                    for bottom_pos in left_bottom:
                        if top_pos == 0:
                            pattern = f"^{char2}.{{{bottom_pos-1}}}{char1}"
                        else:
                            pattern = f"^.{{{top_pos}}}{char2}.{{{bottom_pos-top_pos-1}}}{char1}"
                        patterns.append(pattern)
                
                # Right hand: char1 on top, char2 on bottom (any columns)
                for top_pos in right_top:
                    for bottom_pos in right_bottom:
                        pattern = f"^.{{{top_pos}}}{char1}.{{{bottom_pos-top_pos-1}}}{char2}"
                        patterns.append(pattern)
                
                # Right hand: char2 on top, char1 on bottom (any columns)
                for top_pos in right_top:
                    for bottom_pos in right_bottom:
                        pattern = f"^.{{{top_pos}}}{char2}.{{{bottom_pos-top_pos-1}}}{char1}"
                        patterns.append(pattern)
            
            if self.verbose:
                print(f"Generated {len(patterns)} hurdle exclusion patterns for {len(bigrams)} bigrams (top-to-bottom on same hand)")
            
            return patterns

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
        report_path = Path(output_dir) / f'layouts_filter_patterns_report_{timestamp}.txt'
        
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
    parser.add_argument('--exclude-vertical-bigrams', type=str,
                        help='Comma-separated bigrams to exclude when vertically stacked (e.g., "th,he,er")')
    parser.add_argument('--exclude-hurdles', type=str,
                        help='Comma-separated bigrams to exclude when creating top-to-bottom movements on same hand (e.g., "th,qz,wc")')

    # Position constraint options
    parser.add_argument('--forbidden-letters', type=str,
                       help='Letters that should not appear in forbidden positions (e.g., "etaio")')
    parser.add_argument('--forbidden-positions', type=str,
                       help='QWERTY position characters where forbidden letters should not appear (e.g., "A;R")')
    
    # Output options
    parser.add_argument('--output-dir', default='../output',
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
        
        # Initialize filter FIRST
        filter_tool = LayoutPatternFilter(args.input, args.verbose)
        
        # Parse patterns
        exclude_patterns = parse_patterns(args.exclude)
        include_patterns = parse_patterns(args.include)
        
        # Add vertical bigrams if specified
        if args.exclude_vertical_bigrams:
            vertical_patterns = filter_tool.generate_vertical_exclusion_patterns(args.exclude_vertical_bigrams)
            exclude_patterns.extend(vertical_patterns)

        # Add hurdle bigrams if specified (top-to-bottom movements on same hand)
        if args.exclude_hurdles:
            hurdle_patterns = filter_tool.generate_hurdle_exclusion_patterns(args.exclude_hurdles)
            exclude_patterns.extend(hurdle_patterns)

        if args.verbose:
            if exclude_patterns:
                print(f"Total exclude patterns: {len(exclude_patterns)}")
            if include_patterns:
                print(f"Include patterns: {include_patterns}")
            if args.forbidden_letters and args.forbidden_positions:
                print(f"Forbidden: letters '{args.forbidden_letters}' in positions '{args.forbidden_positions}'")
            if args.exclude_hurdles:
                print(f"Excluding hurdle bigrams (top-to-bottom on same hand): {args.exclude_hurdles}")

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
            args.output = f"{args.output_dir}/layouts_filter_patterns_{input_stem}_{timestamp}.csv"
        
        # Save results
        filtered_df.to_csv(args.output, index=False)
        
        # Save removed layouts if requested
        if args.save_removed and len(removed_df) > 0:
            removed_path = Path(args.output).parent / f"layouts_filter_patterns_removed_{Path(args.output).name}"
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
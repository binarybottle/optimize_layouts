#!/usr/bin/env python3
"""
Filter keyboard layout results based on layout pattern matching.

This script allows filtering of keyboard layouts based on:
1. Regex patterns to exclude (filter out)
2. Regex patterns to include (retain only)
3. Specific letter-position constraints (e.g., 'etaio' should not be in positions 'A;R')
4. Vertical bigrams (same column, any rows)
5. Hurdle bigrams (top-to-bottom or bottom-to-top on same hand, any columns)
6. Same-finger hurdle bigrams (top-to-bottom on same column/finger)
7. Adjacent-finger hurdle bigrams (top-to-bottom on adjacent columns/fingers)

IMPORTANT: All bigram filters are layout-agnostic. They prevent the specified bigrams from 
being placed in problematic positions in ANY new layout, regardless of where those letters 
appear in QWERTY.

Vertical bigrams: Two letters in the same column, any rows (e.g., if layout has 'e' at top 
                  and 'd' at bottom of same column). Useful for common same-hand bigrams 
                  like "ed", "er", "de", "as", "an", "re" and high-frequency bigrams like 
                  "th", "he", "in" to prevent awkward vertical stacking.

Hurdle bigrams: Two letters on the same hand where one is on top row and other is on bottom row,
                regardless of column. This creates a "hurdle" motion over the home row.
                Examples: For a new layout, prevent "ed", "re", "an" (typically same-hand) or 
                "th", "he", "in" (typically cross-hand) from being placed as top-to-bottom 
                on the same hand.

Same-finger hurdles: Subset of hurdles - only top-to-bottom on the same column (same finger).
                     Most restrictive for same-column placement.

Adjacent-finger hurdles: Top-to-bottom movements on adjacent columns (adjacent fingers).
                        Specific position pairs like Q-X, R-C, Y-M, P-.

The layout_qwerty string follows QWERTY key order: "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"
Positions are indexed from 0, so 'A' is position 10, ';' is position 19, 'R' is position 13, etc.

    # Indices:
    ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
    │ 0 │ 1 │ 2 │ 3 │   │   │ 6 │ 7 │ 8 │ 9 │   │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
    │10 │11 │12 │13 │   │   │16 │17 │18 │19 │   │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤───┘
    │20 │21 │22 │23 │   │   │26 │27 │28 │29 │
    └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘      

Usage:

    # Keyboard layout optimization study command:
    # Don't permit common bigrams to be stacked vertically or to require hurdles.
    BIGRAMS25="th,he,in,er,an,re,on,at,en,nd,ti,es,or"
    BIGRAMS50="te,of,ed,is,it,al,ar,st,to,nt,ng,se,ha,as,ou,io,le,ve,co,me,de,hi,ri,ro,ic,ne,ea"
    BIGRAMS75="ra,ce,li,ch,ll,be,ma,si,om,ur,ca,el,ta,la,ns,di,fo,ho,pe,ec,pr,no,ct,us,ac,ot,il,tr,ly,nc,et,ut,ss,so,rs,un,lo,wa,ge,ie,wh,ee,wi,em,ad,ol,rt,po,we,na"
    BIGRAMS90="ul,ni,ts,mo,ow,pa,im,mi,ai,sh,ir,su,id,os,iv,ia,am,fi,ci,vi,pl,ig,tu,ev,ld,ry,mp,fe,bl,ab,gh,ty,op,wo,sa,ay,ex,ke,fr,oo,av,ag,if,ap,gr,od,bo,sp,rd,do,uc,bu,ei,ov,by,rm,ep,tt,oc,fa,ef,cu,rn,sc,gi,da,yo,cr,cl,du,ga,qu,ue,ff,ba,ey,ls,va,um,pp,ua,up,lu,go,ht,ru,ug,ds,lt,pi,rc,rr,eg,au"
    SAME_FINGER_BIGRAMS="$BIGRAMS25,$BIGRAMS50"
    HURDLE_BIGRAMS="$BIGRAMS25,$BIGRAMS50"
    SAME_FINGER_HURDLE_BIGRAMS="$BIGRAMS25,$BIGRAMS50,$BIGRAMS75,$BIGRAMS75,$BIGRAMS90"
    poetry run python3 layouts_filter_patterns.py \
        --input ../output/layouts_consolidate_moo_solutions.csv \
        --output ../output/layouts_filter_patterns.csv --report \
        --exclude-vertical-bigrams "$SAME_FINGER_BIGRAMS" \
        --exclude-hurdles "$HURDLE_BIGRAMS" \
        --exclude-same-finger-hurdles "$SAME_FINGER_HURDLE_BIGRAMS"

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
        
        This function is layout-agnostic: it prevents the specified bigrams from being placed
        in the same column (same finger) regardless of which columns they occupy. For each bigram,
        it generates patterns for all 10 possible columns.
        
        Examples of bigrams to exclude:
        - Common same-hand bigrams in QWERTY: "ed" and "de"
        - High-frequency bigrams that would be awkward vertically: "th", "he", "in"
          (even if typically cross-hand in QWERTY, we prevent them from being
          placed vertically in any new layout)
        
        For instance, if "th" is in the bigram list:
        - Excludes layouts where 't' is at position 0 (Q) and 'h' is at position 10 (A)
        - Excludes layouts where 't' is at position 3 (R) and 'h' is at position 23 (V)
        - ...and so on for all 10 columns in both directions
        
        Args:
            bigrams_str: Comma-separated string of bigrams (e.g., "th,he,in,er,an,re")
        
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
            
            # Top row to bottom row stacking (positions 0-9 to 20-29)
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
            print(f"Generated {len(patterns)} vertical exclusion patterns for {len(bigrams)} bigrams (all same-column combinations)")
        
        return patterns

    def generate_hurdle_exclusion_patterns(self, bigrams_str: str) -> List[str]:
        """
        Generate exclusion patterns for bigrams that create top-to-bottom movements on same hand.
        Excludes any bigram where one letter is on the top row and the other is on the bottom row
        of the same hand, regardless of column (creates a "hurdle" over the home row).
        
        This checks all possible top-to-bottom placements on each hand:
        - Left hand: Any combination of positions 0-4 (QWERT) with 20-24 (ZXCVB)
        - Right hand: Any combination of positions 5-9 (YUIOP) with 25-29 (NM,./)
        
        Examples of bigrams to exclude:
        - Common same-hand QWERTY bigrams: "ed" and "de"
        - High-frequency cross-hand bigrams: "th" and "he"
          (prevents them from being placed as hurdles in any new layout)
        
        For instance, if "th" is in the list:
        - Excludes 't' at position 0 (Q) with 'h' at any of positions 20-24 (Z,X,C,V,B)
        
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
        # Left hand: top=0-4 (QWERT), bottom=20-24 (ZXCVB)
        # Right hand: top=5-9 (YUIOP), bottom=25-29 (NM,./)
        left_top = list(range(0, 5))       # Q, W, E, R, T
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
            print(f"Generated {len(patterns)} hurdle exclusion patterns for {len(bigrams)} bigrams (top-to-bottom on same hand, any columns)")
        
        return patterns

    def generate_same_finger_hurdle_exclusion_patterns(self, bigrams_str: str) -> List[str]:
        """
        Generate exclusion patterns for bigrams that create same-finger hurdle movements.
        Same-finger hurdles are top-to-bottom movements in the same column (same finger).
        This is the most restrictive hurdle filter, focusing only on same-column transitions.
        
        This function is layout-agnostic: it checks all possible same-column placements:
        - Left hand: (0,20) Q-Z, (1,21) W-X, (2,22) E-C, (3,23) R-V
        - Right hand: (6,26) Y-N, (7,27) U-M, (8,28) I-comma, (9,29) O-period
        
        For instance, if "ed" is in the list:
        - Excludes 'e' at position 2 (E) with 'd' at position 22 (C)
        - Excludes 'd' at position 7 (U) with 'e' at position 27 (M)
        - ...and so on for all 8 same-column pairs
        
        Args:
            bigrams_str: Comma-separated string of bigrams (e.g., "th,he,in,ed,er,an")
        
        Returns:
            List of regex patterns to exclude same-finger hurdle bigram placements
        """

        if not bigrams_str:
            return []
        
        bigrams = [b.strip() for b in bigrams_str.split(',') if b.strip()]
        patterns = []
        
        # Define same-finger hurdle position pairs: (top_pos, bottom_pos) in same column
        # Left hand columns
        same_finger_pairs = [
            (0, 20),   # Q-Z (left pinky)
            (1, 21),   # W-X (left ring)
            (2, 22),   # E-C (left middle)
            (3, 23),   # R-V (left index)
        ]
        
        # Right hand columns  
        same_finger_pairs.extend([
            (6, 26),   # Y-N (right index)
            (7, 27),   # U-M (right middle)
            (8, 28),   # I-, (right ring)
            (9, 29),   # O-. (right pinky)
        ])
        
        for bigram in bigrams:
            if len(bigram) != 2:
                continue
            
            char1, char2 = bigram[0].lower(), bigram[1].lower()
            
            for top_pos, bottom_pos in same_finger_pairs:
                # Skip if positions are outside layout bounds
                if bottom_pos >= len(self.QWERTY_ORDER):
                    continue
                
                # char1 on top, char2 on bottom
                if top_pos == 0:
                    pattern1 = f"^{char1}.{{{bottom_pos-1}}}{char2}"
                else:
                    pattern1 = f"^.{{{top_pos}}}{char1}.{{{bottom_pos-top_pos-1}}}{char2}"
                
                # char2 on top, char1 on bottom (reverse direction)
                if top_pos == 0:
                    pattern2 = f"^{char2}.{{{bottom_pos-1}}}{char1}"
                else:
                    pattern2 = f"^.{{{top_pos}}}{char2}.{{{bottom_pos-top_pos-1}}}{char1}"
                
                patterns.extend([pattern1, pattern2])
        
        if self.verbose:
            print(f"Generated {len(patterns)} same-finger hurdle exclusion patterns for {len(bigrams)} bigrams (top-to-bottom, same column)")
        
        return patterns

    def generate_adjacent_finger_hurdle_exclusion_patterns(self, bigrams_str: str) -> List[str]:
        """
        Generate exclusion patterns for bigrams that create adjacent-finger hurdle motions.
        Adjacent-finger hurdles are top-to-bottom movements on adjacent columns (adjacent fingers).
        
        This function is layout-agnostic: it checks specific adjacent-finger position pairs:
        - Left hand: (0,21) Q-X pinky→ring, 
                     (1,22) W-C ring→middle, 
                     (2,21) E-X middle→ring,
                     (3,22) R-C index→middle
        - Right hand: (6,27) Y-M index→middle, 
                      (7,28) U-comma middle→ring, 
                      (8,27) I-M ring→middle, 
                      (9,28) P-period pinky→ring

        # Keyboard layout:
        ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
        │ 0 │ 1 │ 2 │ 3 │   │   │ 6 │ 7 │ 8 │ 9 │
        ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        │10 │11 │12 │13 │   │   │16 │17 │18 │19 │
        ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        │20 │21 │22 │23 │   │   │26 │27 │28 │29 │
        └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
        
        For instance, if "th" is in the list:
        - Excludes 't' at position 3 (R) with 'h' at position 22 (C)
        - Excludes 'h' at position 1 (W) with 't' at position 22 (C)
        - ...and so on for all 8 adjacent-finger pairs
        
        Args:
            bigrams_str: Comma-separated string of bigrams (e.g., "th,he,in,ed,er,an")
        
        Returns:
            List of regex patterns to exclude adjacent-finger hurdle bigram placements
        """
        if not bigrams_str:
            return []
        
        bigrams = [b.strip() for b in bigrams_str.split(',') if b.strip()]
        patterns = []
        
        # Define adjacent-finger hurdle position pairs: (top_pos, bottom_pos)
        adjacent_finger_pairs = [
            (0, 21),   # Q-X (left pinky to left ring)
            (1, 22),   # W-C (left ring to left middle)
            (2, 23),   # E-V (middle index to left index)
            (1, 20),   # W-Z (left ring to left pinky)
            (2, 21),   # E-X (left middle to left ring)
            (3, 22),   # R-C (left index to left middle)
            (6, 27),   # Y-M (right index to right middle)
            (7, 28),   # U-. (right middle to right ring)
            (8, 29),   # O-/ (right ring to right pinky)
            (7, 26),   # I-N (right middle to right index)
            (8, 27),   # I-M (right ring to right middle)
            (9, 28),   # P-. (right pinky to right ring)
        ]
        
        for bigram in bigrams:
            if len(bigram) != 2:
                continue
            
            char1, char2 = bigram[0].lower(), bigram[1].lower()
            
            for top_pos, bottom_pos in adjacent_finger_pairs:
                # Skip if positions are outside layout bounds
                if bottom_pos >= len(self.QWERTY_ORDER):
                    continue
                
                # char1 on top, char2 on bottom
                if top_pos == 0:
                    pattern1 = f"^{char1}.{{{bottom_pos-1}}}{char2}"
                else:
                    pattern1 = f"^.{{{top_pos}}}{char1}.{{{bottom_pos-top_pos-1}}}{char2}"
                
                # char2 on top, char1 on bottom (reverse direction)
                if top_pos == 0:
                    pattern2 = f"^{char2}.{{{bottom_pos-1}}}{char1}"
                else:
                    pattern2 = f"^.{{{top_pos}}}{char2}.{{{bottom_pos-top_pos-1}}}{char1}"
                
                patterns.extend([pattern1, pattern2])
        
        if self.verbose:
            print(f"Generated {len(patterns)} adjacent-finger hurdle exclusion patterns for {len(bigrams)} bigrams (top-to-bottom, adjacent columns)")
        
        return patterns

    def generate_scissor_hurdle_exclusion_patterns(self, bigrams_str: str) -> List[str]:
        """
        Generate exclusion patterns for bigrams that create adjacent-finger scissor hurdle motions.
        Scissor hurdles are top-to-bottom movements on adjacent columns (adjacent fingers)
        in which the shorter finger reaches up and the longer finger reaches down.
        
        # Keyboard layout:
        ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
        │ 0 │ 1 │ 2 │ 3 │   │   │ 6 │ 7 │ 8 │ 9 │
        ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        │10 │11 │12 │13 │   │   │16 │17 │18 │19 │
        ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        │20 │21 │22 │23 │   │   │26 │27 │28 │29 │
        └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
        
        For instance, if "th" is in the list:
        - Excludes 't' at position 3 (R) with 'h' at position 22 (C)
        - Excludes 'h' at position 0 (Q) with 't' at position 21 (X)
        
        Args:
            bigrams_str: Comma-separated string of bigrams (e.g., "th,he,in,ed,er,an")
        
        Returns:
            List of regex patterns to exclude adjacent-finger hurdle bigram placements
        """
        if not bigrams_str:
            return []
        
        bigrams = [b.strip() for b in bigrams_str.split(',') if b.strip()]
        patterns = []
        
        # Define adjacent-finger hurdle position pairs: (top_pos, bottom_pos)
        adjacent_finger_pairs = [
            (0, 21),   # Q-X (left pinky to left ring)
            (3, 22),   # R-C (left index to left middle)
            (6, 27),   # Y-M (right index to right middle)
            (9, 28),   # P-. (right pinky to right ring)
        ]
        
        for bigram in bigrams:
            if len(bigram) != 2:
                continue
            
            char1, char2 = bigram[0].lower(), bigram[1].lower()
            
            for top_pos, bottom_pos in adjacent_finger_pairs:
                # Skip if positions are outside layout bounds
                if bottom_pos >= len(self.QWERTY_ORDER):
                    continue
                
                # char1 on top, char2 on bottom
                if top_pos == 0:
                    pattern1 = f"^{char1}.{{{bottom_pos-1}}}{char2}"
                else:
                    pattern1 = f"^.{{{top_pos}}}{char1}.{{{bottom_pos-top_pos-1}}}{char2}"
                
                # char2 on top, char1 on bottom (reverse direction)
                if top_pos == 0:
                    pattern2 = f"^{char2}.{{{bottom_pos-1}}}{char1}"
                else:
                    pattern2 = f"^.{{{top_pos}}}{char2}.{{{bottom_pos-top_pos-1}}}{char1}"
                
                patterns.extend([pattern1, pattern2])
        
        if self.verbose:
            print(f"Generated {len(patterns)} adjacent-finger hurdle exclusion patterns for {len(bigrams)} bigrams (top-to-bottom, adjacent columns)")
        
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
    parser.add_argument('--exclude-same-finger-hurdles', type=str,
                        help='Comma-separated bigrams to exclude in same-finger hurdle positions (top-to-bottom, same column)')
    parser.add_argument('--exclude-adjacent-hurdles', type=str,
                        help='Comma-separated bigrams to exclude in adjacent-finger hurdle positions: [0,21], [3,22], [6,27], [9,28], [1,22], [2,21], [7,28], [8,27]')
    parser.add_argument('--exclude-scissor-hurdles', type=str,
                        help='Comma-separated bigrams to exclude in scissor hurdle positions (adjacent fingers, shorter reaches up): [0,21], [3,22], [6,27], [9,28]')

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

        # Add same-finger hurdle bigrams if specified (top-to-bottom, same column)
        if args.exclude_same_finger_hurdles:
            same_finger_patterns = filter_tool.generate_same_finger_hurdle_exclusion_patterns(args.exclude_same_finger_hurdles)
            exclude_patterns.extend(same_finger_patterns)

        # Add adjacent-finger hurdle bigrams if specified (specific position pairs)
        if args.exclude_adjacent_hurdles:
            adjacent_finger_patterns = filter_tool.generate_adjacent_finger_hurdle_exclusion_patterns(args.exclude_adjacent_hurdles)
            exclude_patterns.extend(adjacent_finger_patterns)

        # Add scissor hurdle bigrams if specified (specific position pairs)
        if args.exclude_scissor_hurdles:
            scissor_hurdle_patterns = filter_tool.generate_scissor_hurdle_exclusion_patterns(args.exclude_scissor_hurdles)
            exclude_patterns.extend(scissor_hurdle_patterns)

        if args.verbose:
            if exclude_patterns:
                print(f"Total exclude patterns: {len(exclude_patterns)}")
            if include_patterns:
                print(f"Include patterns: {include_patterns}")
            if args.forbidden_letters and args.forbidden_positions:
                print(f"Forbidden: letters '{args.forbidden_letters}' in positions '{args.forbidden_positions}'")
            if args.exclude_hurdles:
                print(f"Excluding hurdle bigrams (top-to-bottom on same hand): {args.exclude_hurdles}")
            if args.exclude_same_finger_hurdles:
                print(f"Excluding same-finger hurdle bigrams (top-to-bottom, same column): {args.exclude_same_finger_hurdles}")
            if args.exclude_adjacent_hurdles:
                print(f"Excluding adjacent-finger hurdle bigrams: {args.exclude_adjacent_hurdles}")
            if args.exclude_scissor_hurdles:
                print(f"Excluding scissor hurdle bigrams: {args.exclude_scissor_hurdles}")

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
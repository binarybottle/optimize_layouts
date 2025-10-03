#!/usr/bin/env python3
"""
Find common rows between two CSV files based on a column value.

Usage:
    python find_common_table_rows.py table1.csv table2.csv --column-name layout_qwerty
    python find_common_table_rows.py file1.csv file2.csv --column-name items --output intersecting.csv

    poetry run python3 find_common_table_rows.py ../output/layouts_compare_2objectives.csv ../output/layouts_compare_5objectives.csv --column-name positions --output ../output/compare_layouts_common_rows_2and5objectives.csv

"""

import pandas as pd
import argparse
import sys
from pathlib import Path

def compare_tables(file1: str, file2: str, column_name: str, output_file: str = None, verbose: bool = False):
    """
    Find rows where column values appear in both tables.
    
    Args:
        file1: Path to first CSV file
        file2: Path to second CSV file
        column_name: Column name to compare
        output_file: Optional output file for intersecting rows from file1
        verbose: Print detailed statistics
    """
    try:
        # Load files
        print(f"Loading {file1}...")
        df1 = pd.read_csv(file1)
        print(f"Loading {file2}...")
        df2 = pd.read_csv(file2)
        
        # Check column exists
        if column_name not in df1.columns:
            print(f"Error: Column '{column_name}' not found in {file1}")
            print(f"Available columns: {', '.join(df1.columns)}")
            return 1
        
        if column_name not in df2.columns:
            print(f"Error: Column '{column_name}' not found in {file2}")
            print(f"Available columns: {', '.join(df2.columns)}")
            return 1
        
        # Get unique values
        set1 = set(df1[column_name].dropna())
        set2 = set(df2[column_name].dropna())
        
        # Calculate intersections
        both = set1 & set2
        only_1 = set1 - set2
        only_2 = set2 - set1
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Comparison Results for column: '{column_name}'")
        print(f"{'='*60}")
        
        print(f"\nFile 1: {Path(file1).name}")
        print(f"  Total rows: {len(df1):,}")
        print(f"  Unique '{column_name}' values: {len(set1):,}")
        
        print(f"\nFile 2: {Path(file2).name}")
        print(f"  Total rows: {len(df2):,}")
        print(f"  Unique '{column_name}' values: {len(set2):,}")
        
        print(f"\n{'='*60}")
        print(f"INTERSECTION: {len(both):,} unique values appear in BOTH files")
        print(f"Only in file 1: {len(only_1):,} unique values")
        print(f"Only in file 2: {len(only_2):,} unique values")
        print(f"{'='*60}")
        
        # Calculate percentage
        if len(set1) > 0:
            pct1 = (len(both) / len(set1)) * 100
            print(f"\n{pct1:.1f}% of file1's unique values are in file2")
        if len(set2) > 0:
            pct2 = (len(both) / len(set2)) * 100
            print(f"{pct2:.1f}% of file2's unique values are in file1")
        
        # Show examples if verbose
        if verbose and both:
            print(f"\nFirst 10 intersecting values:")
            for i, value in enumerate(sorted(both)[:10], 1):
                print(f"  {i:2d}. {value}")
        
        # Get actual rows that intersect
        df1_intersecting = df1[df1[column_name].isin(set2)]
        df2_intersecting = df2[df2[column_name].isin(set1)]
        
        print(f"\nRows in file1 with intersecting values: {len(df1_intersecting):,}")
        print(f"Rows in file2 with intersecting values: {len(df2_intersecting):,}")
        
        # Save output if requested
        if output_file:
            df1_intersecting.to_csv(output_file, index=False)
            print(f"\nSaved intersecting rows from file1 to: {output_file}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

def main():
    parser = argparse.ArgumentParser(
        description='Find common rows between two CSV files based on a column',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python find_common_rows_in_two_tables.py table1.csv table2.csv --column-name layout_qwerty
  
  # Save intersecting rows
  python find_common_rows_in_two_tables.py file1.csv file2.csv --column-name items --output results.csv
  
  # Verbose output with examples
  python find_common_rows_in_two_tables.py data1.csv data2.csv --column-name positions --verbose
        """
    )
    
    parser.add_argument('table1', help='First CSV file')
    parser.add_argument('table2', help='Second CSV file')
    parser.add_argument('--column-name', '--column', '-c', required=True,
                       help='Column name to compare between files')
    parser.add_argument('--output', '-o', help='Output CSV file for intersecting rows from table1')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed statistics and examples')
    
    args = parser.parse_args()
    
    return compare_tables(
        args.table1,
        args.table2,
        args.column_name,
        args.output,
        args.verbose
    )

if __name__ == '__main__':
    sys.exit(main())
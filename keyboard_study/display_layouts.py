#!/usr/bin/env python3
"""
Display multiple keyboard layouts from CSV file

Data Format Support
=============================
CSV Input Formats (auto-detected):
1. Preferred: layout_qwerty column (layout string in QWERTY key order)
2. Standard: letters column (layout string in QWERTY key order) 
3. MOO: items + positions columns (converted to QWERTY order automatically)

Column Meanings:
- layout_qwerty: What's at each QWERTY position (e.g., "  cr  du  oinl  teha   s  m     ")
- letters: What's at each QWERTY position (same as layout_qwerty)
- items: Letters in assignment order (e.g., "etaoinsrhldcum")
- positions: QWERTY positions where those letters go (e.g., "KJ;ASDVRLFUEIM")

Usage:
    python display_layouts.py layouts.csv

CSV Examples:
  # Preferred format
  layout,layout_qwerty
  Dvorak,',.pyfgcrlaeoiduhtns;qjkxbmwvz
  
  # Standard format  
  layout,letters,positions
  Dvorak,',.pyfgcrlaeoiduhtns;qjkxbmwvz,QWERTYUIOPASDFGHJKL;ZXCVBNM,./[
  
  # MOO format (auto-converted)
  config_id,items,positions
  2438,etaoinsrhldcum,KJ;ASDVRLFUEIM
"""

import argparse
import csv
import subprocess
import sys
import os

QWERTY_ORDER = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"

def detect_csv_format(df_or_fieldnames):
    """
    Detect which CSV format is being used.
    
    Args:
        df_or_fieldnames: Either pandas DataFrame or list of column names
        
    Returns:
        Tuple of (format_type, letters_col, positions_col)
    """
    if hasattr(df_or_fieldnames, 'columns'):
        # pandas DataFrame
        fieldnames = list(df_or_fieldnames.columns)
    else:
        # list of column names
        fieldnames = df_or_fieldnames
    
    if 'layout_qwerty' in fieldnames:
        return ('layout_qwerty', 'layout_qwerty', None)
    elif 'letters' in fieldnames:
        return ('letters', 'letters', 'positions')  
    elif 'items' in fieldnames and 'positions' in fieldnames:
        return ('moo', 'items', 'positions')
    else:
        raise ValueError("CSV must contain 'layout_qwerty', 'letters', or 'items'+'positions' columns")

def convert_moo_to_qwerty_layout(items, positions):
    """
    Convert MOO format to QWERTY layout string.
    
    Args:
        items: Letters in assignment order (e.g., "etaoinsrhldcum")
        positions: QWERTY positions where those letters go (e.g., "KJ;ASDVRLFUEIM")
        
    Returns:
        Layout string in QWERTY key order (e.g., "  cr  du  oinl  teha   s  m     ")
    """
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

def extract_layout_data(row, format_type, letters_col, positions_col):
    """
    Extract layout data from a CSV row based on detected format.
    
    Args:
        row: CSV row (dict)
        format_type: Detected format ('layout_qwerty', 'letters', or 'moo')
        letters_col: Column name containing letters/layout data
        positions_col: Column name containing positions (or None)
        
    Returns:
        Tuple of (letters_in_qwerty_order, qwerty_positions)
    """
    if format_type == 'layout_qwerty':
        # layout_qwerty format - preserve all spaces (they represent empty key positions)
        letters = row[letters_col]  # Don't strip!
        positions = QWERTY_ORDER
        
    elif format_type == 'letters':
        # Standard letters + positions format - preserve all spaces
        letters = row[letters_col]  # Don't strip!
        positions = row.get(positions_col, QWERTY_ORDER).strip() if positions_col else QWERTY_ORDER
        
    elif format_type == 'moo':
        # MOO items + positions format - convert to QWERTY order
        items = row[letters_col].strip()  # OK to strip - just letter names
        item_positions = row[positions_col].strip()  # OK to strip - just position names
        
        # Convert to QWERTY layout string
        letters = convert_moo_to_qwerty_layout(items, item_positions)
        positions = QWERTY_ORDER
        
    else:
        raise ValueError(f"Unknown format type: {format_type}")
    
    return letters, positions

def process_layout(layout_name, letters, positions, format_info=""):
    """
    Process a single layout and display it using display_layout.py
    
    Args:
        layout_name: Name of the layout
        letters: Letters in QWERTY order (what's at each QWERTY position)
        positions: QWERTY positions (reference)
        format_info: Optional format information for display
    """
    if not letters or not positions:
        print(f"{layout_name}: (empty layout)")
        return
    
    # Print layout name with format info
    print(f"{layout_name}{format_info}: {letters} → {positions}")
    
    # Call display_layout.py with QWERTY-ordered data
    try:
        cmd = [
            sys.executable, 'display_layout.py',
            '--letters', letters,
            '--positions', positions,
            '--quiet'
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error displaying layout: {e}")
    except FileNotFoundError:
        print("Error: display_layout.py not found in current directory")

def main():
    parser = argparse.ArgumentParser(
        description='Display multiple keyboard layouts from CSV file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CSV Format Support (auto-detected):
==================================

1. Preferred format:
   layout,layout_qwerty
   Dvorak,',.pyfgcrlaeoiduhtns;qjkxbmwvz

2. Standard format:
   layout,letters,positions  
   Dvorak,',.pyfgcrlaeoiduhtns;qjkxbmwvz,QWERTYUIOPASDFGHJKL;ZXCVBNM,./[

3. MOO format (auto-converted):
   config_id,items,positions
   2438,etaoinsrhldcum,KJ;ASDVRLFUEIM

Examples:
  python display_layouts.py layouts.csv
  python display_layouts.py moo_results.csv  
        """
    )
    
    parser.add_argument('csv_file', 
                       help='CSV file with layouts (auto-detects format)')
    
    parser.add_argument('--html', action='store_true',
                       help='Generate HTML output for each layout')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: File '{args.csv_file}' not found")
        sys.exit(1)
    
    # Check if display_layout.py exists
    if not os.path.exists('display_layout.py'):
        print("Error: display_layout.py not found in current directory")
        sys.exit(1)
    
    # Read and process CSV
    try:
        with open(args.csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames or []
            
            # Detect format
            format_type, letters_col, positions_col = detect_csv_format(fieldnames)
            
            print(f"Detected CSV format: {format_type}")
            if format_type == 'moo':
                print("  Converting from MOO format (items + positions) to QWERTY layout strings")
            
            # Check for required columns
            if not letters_col or letters_col not in fieldnames:
                print(f"Error: Required column '{letters_col}' not found")
                print(f"Found columns: {', '.join(fieldnames)}")
                sys.exit(1)
            
            if positions_col and positions_col not in fieldnames:
                print(f"Error: Required column '{positions_col}' not found")
                print(f"Found columns: {', '.join(fieldnames)}")
                sys.exit(1)
            
            # Find layout name column
            layout_col = None
            for col in ['layout', 'config_id', 'name']:
                if col in fieldnames:
                    layout_col = col
                    break
            
            if not layout_col:
                print("Warning: No layout name column found, using row numbers")
                layout_col = None
            
            layouts_processed = 0
            for i, row in enumerate(reader):
                layout_name = row.get(layout_col, f"Layout_{i+1}") if layout_col else f"Layout_{i+1}"
                
                try:
                    letters, positions = extract_layout_data(row, format_type, letters_col, positions_col)
                    
                    if layout_name and letters and positions:
                        format_info = f" ({format_type})" if format_type == 'moo' else ""
                        process_layout(layout_name, letters, positions, format_info)
                        layouts_processed += 1
                        
                except Exception as e:
                    print(f"Error processing layout {layout_name}: {e}")
            
            if layouts_processed == 0:
                print("No valid layouts found in CSV file")
            else:
                print(f"\nProcessed {layouts_processed} layouts from {format_type} format")
                
    except FileNotFoundError:
        print(f"Error: Could not read file '{args.csv_file}'")
        sys.exit(1)
    except csv.Error as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Display multiple keyboard layouts from CSV file using display_layout.py

Takes a CSV with 'layout', and either ('letters'/'items') and ('positions'/'keys') columns.

Usage:
    python display_layouts.py layouts.csv

CSV format:
  layout,letters,positions
  Dvorak,'.pyfgcrl,aoeuidhtns;qjkxbmwvz,QWERTYUIOPASDFGHJKL;ZXCVBNM,./['
  Colemak,qwfpgjluy;arstdhneio'zxcvbkm,./, QWERTYUIOPASDFGHJKL;ZXCVBNM,./['
"""
import argparse
import csv
import subprocess
import sys
import os

QWERTY_ORDER = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"

def process_layout(layout_name, letters, positions):
    """
    Process a single layout and display it using display_layout.py
    
    Args:
        layout_name: Name of the layout
        letters: Letters/items to place
        positions: Corresponding QWERTY positions/keys
    """
    if not letters or not positions:
        print(f"{layout_name}: (empty layout)")
        return
    
    # Print layout name
    print(f"{layout_name}: {letters} → {positions}")
    
    # Call display_layout.py
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
CSV format:
  layout,qwerty_letters
  Dvorak,'.pyfgcrlaeoiduhtns;qjkxbmwvz
  Colemak,qwfpgjluy;arstdhneio'zxcvbkm,./

The qwerty_letters column should contain 32 characters representing
the keys in QWERTY order: QWERTYUIOPASDFGHJKL;ZXCVBNM,./['

Examples:
  python display_layouts.py layouts.csv
  python display_layouts.py my_layouts.csv
        """
    )
    
    parser.add_argument('csv_file', 
                       help='CSV file with layout and letters columns')
    
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
            
            # Check for required columns (accept both names)
            fieldnames = reader.fieldnames or []
            
            # Find letters column
            letters_col = None
            for col in ['letters', 'items']:
                if col in fieldnames:
                    letters_col = col
                    break
            
            # Find positions column  
            positions_col = None
            for col in ['positions', 'keys']:
                if col in fieldnames:
                    positions_col = col
                    break
            
            if not letters_col or not positions_col:
                print("Error: CSV must have ('letters' or 'items') and ('positions' or 'keys') columns")
                print(f"Found columns: {', '.join(fieldnames)}")
                sys.exit(1)
            
            layouts_processed = 0
            for row in reader:
                layout_name = row['layout'].strip()
                letters = row[letters_col].strip()
                positions = row[positions_col].strip()
                
                if layout_name and letters and positions:
                    process_layout(layout_name, letters, positions)
                    layouts_processed += 1
            
            if layouts_processed == 0:
                print("No valid layouts found in CSV file")
            else:
                print(f"\nProcessed {layouts_processed} layouts")
                
    except FileNotFoundError:
        print(f"Error: Could not read file '{args.csv_file}'")
        sys.exit(1)
    except csv.Error as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
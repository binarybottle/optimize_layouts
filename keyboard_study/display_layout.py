#!/usr/bin/env python3
"""
32-Key Keyboard Layout Visualizer

This script takes letters and their corresponding QWERTY positions
and displays them on a 32-key keyboard layout (3 rows: 11+11+10 keys).

Usage:

    # Display ASCII keyboard layout with letters and positions:
    python display_layout.py --letters "kgio'\"dncwphae,.tsrlxjyu-?mfvbqz"

    # The above is in Qwerty order, so it is the same as:
    python display_layout.py --letters "kgio'\"dncwphae,.tsrlxjyu-?mfvbqz" \
        --positions "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"

    # To generate a realistic HTML image:
    python display_layout.py --letters "abcd" --positions "RTSG" --html

    # Arbitrary letters and positions:
    python display_layout.py --letters "abc" --positions "P;/"

"""

import argparse
from ast import arg
import sys
import os

QWERTY_ORDER = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"

def create_32key_layout():
    """
    Create a 32-key keyboard layout (3 rows: 11 + 11 + 10 keys)
    Returns layout with default empty character
    """
    layout = [
        [' '] * 11,  # Top row (11 keys)
        [' '] * 11,  # Home row (11 keys) 
        [' '] * 10   # Bottom row (10 keys)
    ]
    return layout

def create_qwerty_mapping():
    """
    Map QWERTY keys to positions in the 32-key layout
    Returns dictionary with QWERTY key -> (row, col) mapping
    """
    qwerty_positions = {
        # Top row (QWERTY row) - 11 keys
        'Q': (0, 0), 'W': (0, 1), 'E': (0, 2), 'R': (0, 3), 'T': (0, 4),
        'Y': (0, 5), 'U': (0, 6), 'I': (0, 7), 'O': (0, 8), 'P': (0, 9), '[': (0, 10),
        
        # Home row (ASDF row) - 11 keys
        'A': (1, 0), 'S': (1, 1), 'D': (1, 2), 'F': (1, 3), 'G': (1, 4),
        'H': (1, 5), 'J': (1, 6), 'K': (1, 7), 'L': (1, 8), ';': (1, 9), "'": (1, 10),
        
        # Bottom row (ZXCV row) - 10 keys
        'Z': (2, 0), 'X': (2, 1), 'C': (2, 2), 'V': (2, 3), 'B': (2, 4),
        'N': (2, 5), 'M': (2, 6), ',': (2, 7), '.': (2, 8), '/': (2, 9)
    }
    return qwerty_positions

def display_ascii(layout, letters, positions, quiet=False):
    """Display keyboard with unicode box-drawing characters"""

    if not quiet:
        print("\nLetters → Qwerty keys:\n{letters}\n{positions}\n".format(letters=letters, positions=positions))
    
    # Row 1 (11 keys)
    print("┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐")
    row1 = "│ " + " │ ".join([f"{key}" for key in layout[0]]) + " │"
    print(row1)
    
    # Row 2 (11 keys)  
    print("├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤")
    row2 = "│ " + " │ ".join([f"{key}" for key in layout[1]]) + " │"
    print(row2)
    
    # Row 3 (10 keys - shorter)
    print("├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤───┘")
    row3 = "│ " + " │ ".join([f"{key}" for key in layout[2]]) + " │"
    print(row3)
    print("└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘    ")

    if not quiet:
        print(f"\n\n")

def display_html(layout, letters, positions, output_file="keyboard_layout.html"):
    """Generate clean HTML file with realistic keyboard layout"""
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Keyboard Layout</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Atkinson Hyperlegible', sans-serif;
            background: white;
            padding: 50px;
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        
        .keyboard {
            background: #444444;
            padding: 20px;
            border-radius: 8px;
        }
        
        .row {
            display: flex;
            margin-bottom: 6px;
        }
        
        .row:last-child {
            margin-bottom: 0;
        }
        
        .key {
            width: 40px;
            height: 40px;
            background: white;
            border: 1px solid #ccc;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 6px;
            font-size: 16px;
            font-weight: 400;
            color: black;
        }
        
        .key:last-child {
            margin-right: 0;
        }
        
        .row-home {
            margin-left: 10px;
        }

        .row-bottom {
            margin-left: 34px;
        }

    </style>
</head>
<body>
    <div class="keyboard">"""
    
    # Generate rows
    for row_idx, row in enumerate(layout):
        if row_idx == 1:
            row_class = "row-home"
        elif row_idx == 2:
            row_class = "row-bottom"
        else:
            row_class = ""
        html_content += f'\n        <div class="row {row_class}">'
        
        for key in row:
            display_key = key if key != ' ' else ''
            html_content += f'\n            <div class="key">{display_key}</div>'
        
        html_content += '\n        </div>'
    
    html_content += """
    </div>
</body>
</html>"""
    
    # Write the HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nHTML keyboard generated: {output_file}")
    print(f"Letters placed: {len(letters)}/32 keys used")

def validate_inputs(letters, positions):
    """Validate the input letters and positions"""
    if len(letters) != len(positions):
        print(f"Error: Number of characters ({len(letters)}) must match number of positions ({len(positions)})")
        return False
    
    if len(letters) > 32:  # Remove the 26 limit
        print(f"Error: Maximum 32 keys available, got {len(letters)} characters")
        return False
    
    # Check for duplicate positions
    if len(set(positions.upper())) != len(positions):
        print("Error: Duplicate positions found")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Visualize letters on a 32-key keyboard layout',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Display modes:
  (default)  Unicode box-drawing keyboard
  --html     Generate clean HTML keyboard

Examples:
  python display_layout.py --letters "etaoin" --positions "FDESGH"
  python display_layout.py --items "etaoin" --keys "FDESGH" --html
        """
)
    
    parser.add_argument('--letters', '--items',
                        dest='letters',
                        required=True,
                        help='Characters to place on keyboard (up to 32 characters)')

    parser.add_argument('--positions', '--keys',      
                        dest='positions',
                        required=False,
                        help='Corresponding QWERTY positions (e.g., "QWERTY" for top row)')
    
    # Display mode options
    parser.add_argument('--html', action='store_true',
                        help='Generate clean HTML file')
    
    parser.add_argument('--output', default='keyboard_layout.html',
                        help='Output filename for HTML mode (default: keyboard_layout.html)')
    
    parser.add_argument('--show-empty', action='store_true',
                        help='Show positions of empty keys')

    parser.add_argument('--quiet', action='store_true',
                        help='Only show keyboard layout')
    
    args = parser.parse_args()
    
    quiet = args.quiet
    letters = args.letters.upper()
    positions = args.positions.upper() if args.positions else QWERTY_ORDER[:len(letters)]

    # Validate inputs
    if not validate_inputs(letters, positions):
        sys.exit(1)
    
    # Create layout and mapping
    layout = create_32key_layout()
    qwerty_map = create_qwerty_mapping()
    
    # Place letters on the keyboard
    invalid_positions = []
    placed_count = 0
    
    for letter, position in zip(letters, positions):
        if position in qwerty_map:
            row, col = qwerty_map[position]
            if row < len(layout) and col < len(layout[row]):
                layout[row][col] = letter.upper()
                placed_count += 1
            else:
                invalid_positions.append(position)
        else:
            invalid_positions.append(position)
    
    # Report any invalid positions
    if invalid_positions:
        print(f"Warning: Invalid QWERTY positions: {', '.join(invalid_positions)}")
        print("Valid positions are: Q W E R T Y U I O P [ A S D F G H J K L ; ' Z X C V B N M , . /")
    
    # Choose display mode
    if args.html:
        display_html(layout, letters, positions, args.output)
    else:  # Default to ASCII
        display_ascii(layout, letters, positions, quiet)
    
    # Show empty keys if requested (ASCII mode only)
    if args.show_empty and not args.html:
        print(f"\nEMPTY KEY POSITIONS:")
        print("-" * 40)
        empty_positions = []
        for qwerty_key, (row, col) in qwerty_map.items():
            if row < len(layout) and col < len(layout[row]) and layout[row][col] == ' ':
                empty_positions.append(qwerty_key)
        
        if empty_positions:
            print("Available positions:", ", ".join(empty_positions))
        else:
            print("All keys are occupied!")

if __name__ == "__main__":
    main()
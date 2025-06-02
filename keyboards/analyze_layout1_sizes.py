#!/usr/bin/env python3
"""
Analyze the distribution of layout sizes in your Step 1 results
"""
import glob
import csv
from collections import Counter

def analyze_layout_sizes():
    """Analyze the sizes of layouts in all result files"""
    
    pattern = "../output/layouts1/moo_results_config_*.csv"
    files = glob.glob(pattern)
    
    size_distribution = Counter()
    configs_by_size = {}
    sample_layouts = {}
    
    print(f"Analyzing {len(files)} CSV files...\n")
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                
                # Skip first 4 lines
                for _ in range(4):
                    next(reader, None)
                
                # Skip header
                header = next(reader, None)
                
                # Read first layout from this file
                first_row = next(reader, None)
                if first_row and len(first_row) >= 3:
                    items = first_row[1]
                    positions = first_row[2]
                    
                    config_num = extract_config_number(file_path)
                    layout_size = len(items)
                    
                    size_distribution[layout_size] += 1
                    
                    if layout_size not in configs_by_size:
                        configs_by_size[layout_size] = []
                        sample_layouts[layout_size] = {
                            'items': items,
                            'positions': positions,
                            'config': config_num,
                            'file': file_path
                        }
                    
                    configs_by_size[layout_size].append(config_num)
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Report results
    print("=== Layout Size Distribution ===")
    for size in sorted(size_distribution.keys()):
        count = size_distribution[size]
        percentage = (count / len(files)) * 100
        print(f"Size {size:2d}: {count:4d} configs ({percentage:5.1f}%)")
    
    print(f"\nTotal configs analyzed: {len(files)}")
    
    # Show samples
    print("\n=== Sample Layouts by Size ===")
    for size in sorted(sample_layouts.keys()):
        sample = sample_layouts[size]
        print(f"\nSize {size} example (config {sample['config']}):")
        print(f"  Items:     {sample['items']}")
        print(f"  Positions: {sample['positions']}")
        print(f"  File: {sample['file'].split('/')[-1]}")
    
    # Recommendations
    print("\n=== Recommendations ===")
    if 16 in size_distribution:
        print("✓ Good: You have layouts with 16 items (expected size)")
    
    if any(size < 16 for size in size_distribution.keys()):
        print("⚠ Warning: Some layouts have fewer than 16 items")
        print("  This could affect Step 2 generation consistency")
        
    if len(size_distribution) > 1:
        print("⚠ Mixed sizes: Your layouts have different sizes")
        print("  Consider filtering to use only size-16 layouts")
        
        # Show how to filter
        if 16 in size_distribution:
            size_16_count = size_distribution[16]
            total_count = sum(size_distribution.values())
            print(f"\n  Option: Use only size-16 layouts:")
            print(f"    Would use {size_16_count} out of {total_count} configs")

def extract_config_number(file_path):
    """Extract config number from filename"""
    import os
    try:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part == "config" and i + 1 < len(parts):
                return int(parts[i + 1])
        return 0
    except (IndexError, ValueError):
        return 0

if __name__ == "__main__":
    analyze_layout_sizes()
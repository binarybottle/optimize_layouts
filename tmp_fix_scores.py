#!/usr/bin/env python3
"""
Fix complete layout scores in existing SLURM CSV result files.

This script reads CSV files with buggy "Complete Layout Score" values
and recalculates them using the correct scoring function.

Usage:
    # Fix all CSV files in the results directory
    python fix_csv_scores.py --config config.yaml
    
    # Fix specific file
    python fix_csv_scores.py --config config.yaml --file "output/layouts/moo_results_config_331_20250601_034347.csv"
    
    # Preview changes without saving
    python fix_csv_scores.py --config config.yaml --dry-run
"""

import argparse
import csv
import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple
import shutil
from datetime import datetime

from config import load_config
from scoring import load_normalized_scores, calculate_complete_layout_score_direct

def create_mapping_from_strings(items_str: str, positions_str: str) -> Dict[str, str]:
    """Create item->position mapping from CSV strings."""
    if len(items_str) != len(positions_str):
        raise ValueError(f"Items length ({len(items_str)}) != positions length ({len(positions_str)})")
    
    return dict(zip(items_str.lower(), positions_str.upper()))

def detect_csv_format(filepath: str) -> str:
    """Detect if CSV is MOO or SOO format."""
    with open(filepath, 'r') as f:
        content = f.read()
        if 'Multi-Objective Optimization Results' in content:
            return 'moo'
        elif 'Single-Objective Optimization Results' in content:
            return 'soo'
        else:
            return 'unknown'

def fix_moo_csv(filepath: str, normalized_scores: Tuple, dry_run: bool = False) -> Dict:
    """Fix MOO CSV file and return statistics."""
    print(f"Fixing MOO file: {filepath}")
    
    # Read original file
    with open(filepath, 'r') as f:
        content = f.readlines()
    
    # Find data section
    data_start = None
    header_row = None
    
    for i, line in enumerate(content):
        if 'Rank' in line and 'Items' in line and 'Positions' in line:
            data_start = i
            header_row = line.strip()
            break
    
    if data_start is None:
        raise ValueError("Could not find data header in CSV file")
    
    # Parse header to find column indices
    header_parts = [part.strip('"') for part in header_row.split('","')]
    
    try:
        rank_col = header_parts.index('Rank')
        items_col = header_parts.index('Items')
        positions_col = header_parts.index('Positions')
        complete_score_col = header_parts.index('Complete Layout Score')
        complete_item_col = header_parts.index('Complete Item')
        complete_pair_col = header_parts.index('Complete Pair')
    except ValueError as e:
        raise ValueError(f"Could not find required columns: {e}")
    
    # Process data rows
    fixed_rows = []
    stats = {'total_rows': 0, 'fixed_rows': 0, 'errors': 0, 'score_changes': []}
    
    for i in range(data_start + 1, len(content)):
        line = content[i].strip()
        if not line:
            continue
            
        # Parse CSV row
        row_parts = [part.strip('"') for part in line.split('","')]
        if len(row_parts) < max(complete_score_col, complete_item_col, complete_pair_col) + 1:
            continue
            
        stats['total_rows'] += 1
        
        try:
            # Extract items and positions
            items_str = row_parts[items_col]
            positions_str = row_parts[positions_col]
            old_complete_score = float(row_parts[complete_score_col])
            
            # Create mapping and recalculate
            mapping = create_mapping_from_strings(items_str, positions_str)
            new_total, new_item, new_pair = calculate_complete_layout_score_direct(
                mapping, normalized_scores
            )
            
            # Update row with new values
            row_parts[complete_score_col] = f"{new_total:.9f}"
            row_parts[complete_item_col] = f"{new_item:.6f}"
            row_parts[complete_pair_col] = f"{new_pair:.6f}"
            
            # Rebuild CSV line with proper quoting
            fixed_line = '"' + '","'.join(row_parts) + '"\n'
            fixed_rows.append(fixed_line)
            
            stats['fixed_rows'] += 1
            stats['score_changes'].append({
                'rank': row_parts[rank_col],
                'old_score': old_complete_score,
                'new_score': new_total,
                'difference': abs(old_complete_score - new_total)
            })
            
        except Exception as e:
            print(f"  Error processing row {i}: {e}")
            stats['errors'] += 1
            # Keep original row on error
            fixed_rows.append(line + '\n' if not line.endswith('\n') else line)
    
    # Write fixed file
    if not dry_run:
        # Backup original
        backup_path = filepath + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup_path)
        print(f"  Backup saved: {backup_path}")
        
        # Write fixed version
        with open(filepath, 'w') as f:
            # Write header section unchanged
            for i in range(data_start + 1):
                f.write(content[i])
            
            # Write fixed data rows
            for row in fixed_rows:
                f.write(row)
        
        print(f"  ✅ Fixed file saved")
    else:
        print(f"  🔍 Dry run - would fix {stats['fixed_rows']} rows")
    
    return stats

def fix_soo_csv(filepath: str, normalized_scores: Tuple, dry_run: bool = False) -> Dict:
    """Fix SOO CSV file and return statistics."""
    print(f"Fixing SOO file: {filepath}")
    
    # Similar implementation for SOO format
    # Read original file
    with open(filepath, 'r') as f:
        content = f.readlines()
    
    # Find data section
    data_start = None
    header_row = None
    
    for i, line in enumerate(content):
        if 'Rank' in line and 'Complete Items' in line and 'Complete Positions' in line:
            data_start = i
            header_row = line.strip()
            break
    
    if data_start is None:
        raise ValueError("Could not find data header in CSV file")
    
    # Parse header
    header_parts = [part.strip('"') for part in header_row.split('","')]
    
    try:
        rank_col = header_parts.index('Rank')
        complete_items_col = header_parts.index('Complete Items')
        complete_positions_col = header_parts.index('Complete Positions')
        complete_score_col = header_parts.index('Complete Layout Score')
        complete_item_col = header_parts.index('Complete Item Score')
        complete_pair_col = header_parts.index('Complete Pair Score')
    except ValueError as e:
        raise ValueError(f"Could not find required columns: {e}")
    
    # Process data rows (similar logic to MOO)
    fixed_rows = []
    stats = {'total_rows': 0, 'fixed_rows': 0, 'errors': 0, 'score_changes': []}
    
    for i in range(data_start + 1, len(content)):
        line = content[i].strip()
        if not line:
            continue
            
        row_parts = [part.strip('"') for part in line.split('","')]
        if len(row_parts) < max(complete_score_col, complete_item_col, complete_pair_col) + 1:
            continue
            
        stats['total_rows'] += 1
        
        try:
            # Extract complete items and positions
            items_str = row_parts[complete_items_col]
            positions_str = row_parts[complete_positions_col]
            old_complete_score = float(row_parts[complete_score_col])
            
            # Create mapping and recalculate
            mapping = create_mapping_from_strings(items_str, positions_str)
            new_total, new_item, new_pair = calculate_complete_layout_score_direct(
                mapping, normalized_scores
            )
            
            # Update row with new values
            row_parts[complete_score_col] = f"{new_total:.9f}"
            row_parts[complete_item_col] = f"{new_item:.6f}"
            row_parts[complete_pair_col] = f"{new_pair:.6f}"
            
            # Rebuild CSV line
            fixed_line = '"' + '","'.join(row_parts) + '"\n'
            fixed_rows.append(fixed_line)
            
            stats['fixed_rows'] += 1
            stats['score_changes'].append({
                'rank': row_parts[rank_col],
                'old_score': old_complete_score,
                'new_score': new_total,
                'difference': abs(old_complete_score - new_total)
            })
            
        except Exception as e:
            print(f"  Error processing row {i}: {e}")
            stats['errors'] += 1
            fixed_rows.append(line + '\n' if not line.endswith('\n') else line)
    
    # Write fixed file (same logic as MOO)
    if not dry_run:
        backup_path = filepath + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup_path)
        print(f"  Backup saved: {backup_path}")
        
        with open(filepath, 'w') as f:
            for i in range(data_start + 1):
                f.write(content[i])
            for row in fixed_rows:
                f.write(row)
        
        print(f"  ✅ Fixed file saved")
    else:
        print(f"  🔍 Dry run - would fix {stats['fixed_rows']} rows")
    
    return stats

def print_summary_stats(all_stats: List[Dict], dry_run: bool = False):
    """Print summary statistics."""
    if not all_stats:
        print("No files processed.")
        return
    
    total_files = len(all_stats)
    total_rows = sum(s['total_rows'] for s in all_stats)
    total_fixed = sum(s['fixed_rows'] for s in all_stats)
    total_errors = sum(s['errors'] for s in all_stats)
    
    # Calculate score change statistics
    all_changes = []
    for stats in all_stats:
        all_changes.extend(stats['score_changes'])
    
    if all_changes:
        changes = [change['difference'] for change in all_changes]
        avg_change = sum(changes) / len(changes)
        max_change = max(changes)
        significant_changes = sum(1 for c in changes if c > 0.001)
    else:
        avg_change = max_change = significant_changes = 0
    
    print(f"\n" + "="*60)
    print(f"SUMMARY STATISTICS")
    print(f"="*60)
    print(f"Files processed:     {total_files}")
    print(f"Total rows:          {total_rows}")
    print(f"Rows fixed:          {total_fixed}")
    print(f"Errors:              {total_errors}")
    print(f"")
    print(f"Score Changes:")
    print(f"  Average difference: {avg_change:.9f}")
    print(f"  Maximum difference: {max_change:.9f}")
    print(f"  Significant changes (>0.001): {significant_changes}")
    
    if dry_run:
        print(f"\n🔍 This was a dry run - no files were actually modified.")
        print(f"   Run without --dry-run to apply fixes.")
    else:
        print(f"\n✅ All fixes applied successfully!")
        print(f"   Original files backed up with .backup_TIMESTAMP extensions.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Fix complete layout scores in existing CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fix all CSV files in results directory
    python fix_csv_scores.py --config config.yaml
    
    # Fix specific file
    python fix_csv_scores.py --config config.yaml --file "output/layouts/moo_results_config_331.csv"
    
    # Preview changes without applying
    python fix_csv_scores.py --config config.yaml --dry-run
    
    # Fix files matching pattern
    python fix_csv_scores.py --config config.yaml --pattern "*config_33*.csv"
        """
    )
    
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--file', help='Fix specific file')
    parser.add_argument('--pattern', help='Fix files matching pattern (e.g., "*moo_results*.csv")')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')
    parser.add_argument('--results-dir', help='Override results directory')
    
    args = parser.parse_args()
    
    try:
        # Load config and scores
        print("Loading configuration and normalized scores...")
        config = load_config(args.config)
        normalized_scores = load_normalized_scores(config)
        
        # Determine files to process
        results_dir = args.results_dir or config.paths.layout_results_folder
        
        if args.file:
            files_to_process = [args.file]
        elif args.pattern:
            files_to_process = glob.glob(os.path.join(results_dir, args.pattern))
        else:
            # Find all result CSV files
            moo_files = glob.glob(os.path.join(results_dir, "moo_results_*.csv"))
            soo_files = glob.glob(os.path.join(results_dir, "soo_results_*.csv"))
            files_to_process = moo_files + soo_files
        
        if not files_to_process:
            print("No CSV files found to process.")
            return
        
        print(f"Found {len(files_to_process)} files to process")
        if args.dry_run:
            print("🔍 DRY RUN MODE - No files will be modified")
        
        # Process each file
        all_stats = []
        
        for filepath in files_to_process:
            if not os.path.exists(filepath):
                print(f"❌ File not found: {filepath}")
                continue
            
            try:
                csv_format = detect_csv_format(filepath)
                
                if csv_format == 'moo':
                    stats = fix_moo_csv(filepath, normalized_scores, args.dry_run)
                elif csv_format == 'soo':
                    stats = fix_soo_csv(filepath, normalized_scores, args.dry_run)
                else:
                    print(f"❓ Unknown CSV format, skipping: {filepath}")
                    continue
                
                all_stats.append(stats)
                
                # Show file-specific stats
                if stats['score_changes']:
                    max_change = max(c['difference'] for c in stats['score_changes'])
                    print(f"  📊 Fixed {stats['fixed_rows']} rows, max change: {max_change:.9f}")
                
            except Exception as e:
                print(f"❌ Error processing {filepath}: {e}")
        
        # Print summary
        print_summary_stats(all_stats, args.dry_run)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
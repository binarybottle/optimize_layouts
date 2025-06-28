#!/usr/bin/env python3
import os
import re
import hashlib
import sys
from collections import defaultdict
import glob

def get_file_hash(filepath):
    """Calculate SHA-256 hash of file content."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def extract_id_from_filename(filename):
    """Extract ID from filename pattern: *_config_{ID}_*.csv"""
    match = re.search(r'_config_(\d+)_', filename)
    return match.group(1) if match else None

def analyze_duplicate_files(directory="."):
    """Analyze files for duplicate content by ID."""
    
    # Find all CSV files matching the pattern
    pattern = os.path.join(directory, "*_config_*_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print("No files found matching pattern *_config_*_*.csv")
        return
    
    # Group files by ID
    files_by_id = defaultdict(list)
    
    for filepath in files:
        filename = os.path.basename(filepath)
        file_id = extract_id_from_filename(filename)
        
        if file_id:
            files_by_id[file_id].append(filepath)
        else:
            print(f"Warning: Could not extract ID from {filename}")
    
    # Analyze each ID group
    results = []
    
    for file_id, filepaths in files_by_id.items():
        # Get content hashes for all files with this ID
        hashes = []
        valid_files = []
        
        for filepath in filepaths:
            file_hash = get_file_hash(filepath)
            if file_hash:
                hashes.append(file_hash)
                valid_files.append(filepath)
        
        if hashes:
            total_files = len(valid_files)
            unique_content = len(set(hashes))
            results.append((int(file_id), total_files, unique_content, valid_files))
    
    # Sort results by ID
    results.sort(key=lambda x: x[0])
    
    # Print results
    print("ID   Total  Unique > 1")
    print("----------------")
    
    for file_id, total_files, unique_content, filepaths in results:
        if unique_content > 1:
            print(f"{file_id:<4} {total_files:<6} {unique_content}")
        
        # Optionally show file details if requested
        if len(sys.argv) > 1 and sys.argv[1] == "--verbose":
            for filepath in filepaths:
                print(f"  {os.path.basename(filepath)}")
            print()

if __name__ == "__main__":
    # Allow specifying directory as command line argument
    directory = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "."
    
    print(f"Analyzing files in: {os.path.abspath(directory)}")
    print()
    
    analyze_duplicate_files(directory)
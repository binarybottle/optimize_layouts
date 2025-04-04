#!/usr/bin/env python
"""
Data normalization script

This script preprocesses all raw input data files for layout optimization,
normalizes the values, and saves them to new files. This ensures consistent
normalization across all optimization and evaluation scripts.

Usage:
    python normalize_inputs.py --config config.yaml --output-dir output/normalized_input
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import yaml
import datetime
from typing import Dict, Tuple, List, Set, Any
from io import StringIO

class TeeLogger:
    """
    Class to capture stdout and write to both console and a file.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
        self.buffer = StringIO()
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.buffer.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        self.buffer.flush()
        
    def get_log_contents(self):
        return self.buffer.getvalue()
        
    def close(self):
        self.log.close()

def detect_and_normalize_distribution(scores: np.ndarray, name: str = '') -> np.ndarray:
    """
    Automatically detect distribution type and apply appropriate normalization.
    Returns scores normalized to [0,1] range.
    """
    # Handle empty or constant arrays
    if len(scores) == 0 or np.all(scores == scores[0]):
        return np.zeros_like(scores)

    # Get basic statistics
    non_zeros = scores[scores != 0]
    if len(non_zeros) == 0:
        return np.zeros_like(scores)
        
    min_nonzero = np.min(non_zeros)
    max_val = np.max(scores)
    mean = np.mean(non_zeros)
    median = np.median(non_zeros)
    skew = np.mean(((non_zeros - mean) / np.std(non_zeros)) ** 3)
    
    # Calculate ratio between consecutive sorted values
    sorted_nonzero = np.sort(non_zeros)
    ratios = sorted_nonzero[1:] / sorted_nonzero[:-1]
    
    # Detect distribution type and apply appropriate normalization
    if len(scores[scores == 0]) / len(scores) > 0.3:
        # Sparse distribution with many zeros
        print(f"{name}: Sparse distribution detected")
        norm_scores = np.sqrt(scores)
        return (norm_scores - np.min(norm_scores)) / (np.max(norm_scores) - np.min(norm_scores))

    elif skew > 2 or np.median(ratios) > 1.5:
        # Heavy-tailed/exponential/zipfian distribution
        print(f"{name}: Heavy-tailed distribution detected")
        norm_scores = np.sqrt(np.abs(scores))
        return (norm_scores - np.min(norm_scores)) / (np.max(norm_scores) - np.min(norm_scores))
        
    elif abs(mean - median) / mean < 0.1:
        # Roughly symmetric distribution
        print(f"{name}: Symmetric distribution detected")
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        
    else:
        # Default to robust scaling
        print(f"{name}: Using robust scaling")
        q1, q99 = np.percentile(scores, [1, 99])
        scaled = (scores - q1) / (q99 - q1)
        return np.clip(scaled, 0, 1)

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def normalize_and_save_data(config: dict, output_dir: str) -> None:
    """
    Normalize all data files and save normalized versions.
    
    Args:
        config: Configuration dictionary with file paths
        output_dir: Directory to save normalized files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging to both console and file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"normalization_log_{timestamp}.txt")
    logger = TeeLogger(log_path)
    sys.stdout = logger
    
    # Print header information
    print(f"Normalization Log - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config file: {config.get('config_file_path', 'Not specified')}")
    print(f"Output directory: {output_dir}")
    print("-" * 80)
    
    # Normalize item scores
    print("\nNormalizing item scores...")
    item_df = pd.read_csv(config['paths']['input']['raw_item_scores_file'], dtype={'item': str})
    scores = item_df['score'].values
    norm_scores = detect_and_normalize_distribution(scores, 'Item scores')
    print("  - original:", min(scores), "to", max(scores))
    print("  - normalized:", min(norm_scores), "to", max(norm_scores))
    
    # Save normalized item scores
    item_df['score'] = norm_scores
    item_df = item_df.drop(columns=['normalized_score'], errors='ignore')
    item_output_path = os.path.join(output_dir, 'normalized_item_scores.csv')
    item_df.to_csv(item_output_path, index=False)
    print(f"  - saved to: {item_output_path}")

    # Normalize position scores
    print("\nNormalizing position scores...")
    position_df = pd.read_csv(config['paths']['input']['raw_position_scores_file'], dtype={'position': str})
    scores = position_df['score'].values
    norm_scores = detect_and_normalize_distribution(scores, 'Position scores')
    print("  - original:", min(scores), "to", max(scores))
    print("  - normalized:", min(norm_scores), "to", max(norm_scores))
    
    # Save normalized position scores
    position_df['score'] = norm_scores
    position_output_path = os.path.join(output_dir, 'normalized_position_scores.csv')
    position_df.to_csv(position_output_path, index=False)
    print(f"  - saved to: {position_output_path}")
    
    # Normalize item pair scores
    print("\nNormalizing item pair scores...")
    item_pair_df = pd.read_csv(config['paths']['input']['raw_item_pair_scores_file'], dtype={'item_pair': str})
    scores = item_pair_df['score'].values
    norm_scores = detect_and_normalize_distribution(scores, 'Item pair scores')
    print("  - original:", min(scores), "to", max(scores))
    print("  - normalized:", min(norm_scores), "to", max(norm_scores))
    
    # Save normalized item pair scores
    item_pair_df['score'] = norm_scores
    item_pair_output_path = os.path.join(output_dir, 'normalized_item_pair_scores.csv')
    item_pair_df.to_csv(item_pair_output_path, index=False)
    print(f"  - saved to: {item_pair_output_path}")
    
    # Normalize position pair scores
    print("\nNormalizing position pair scores...")
    position_pair_df = pd.read_csv(config['paths']['input']['raw_position_pair_scores_file'], dtype={'position_pair': str})
    scores = position_pair_df['score'].values
    norm_scores = detect_and_normalize_distribution(scores, 'Position pair scores')
    print("  - original:", min(scores), "to", max(scores))
    print("  - normalized:", min(norm_scores), "to", max(norm_scores))
    
    # Save normalized position pair scores
    position_pair_df['score'] = norm_scores
    position_pair_output_path = os.path.join(output_dir, 'normalized_position_pair_scores.csv')
    position_pair_df.to_csv(position_pair_output_path, index=False)
    print(f"  - saved to: {position_pair_output_path}")
    
    # Create a modified config file that points to normalized data
    #normalized_config = config.copy()
    #normalized_config['paths']['input']['item_scores_file'] = item_output_path
    #normalized_config['paths']['input']['position_scores_file'] = position_output_path
    #normalized_config['paths']['input']['item_pair_scores_file'] = item_pair_output_path
    #normalized_config['paths']['input']['position_pair_scores_file'] = position_pair_output_path
    
    # Flag to indicate this config uses pre-normalized data
    #normalized_config['optimization']['use_normalized_data'] = True
    
    # Save the modified config
    #config_output_path = os.path.join(output_dir, 'normalized_config.yaml')
    #with open(config_output_path, 'w') as f:
    #    yaml.dump(normalized_config, f, default_flow_style=False)
    #print(f"\nSaved normalized config to: {config_output_path}")
    
    print("\nNormalization complete!")
    print(f"Log saved to: {log_path}")
    
    # Restore stdout and close logger
    sys.stdout = logger.terminal
    logger.close()

def main():
    parser = argparse.ArgumentParser(description='Normalize keyboard layout data.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='output/normalized_input', help='Directory to save normalized data')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Add the config file path to the config dict for logging
    config['config_file_path'] = args.config
    
    normalize_and_save_data(config, args.output_dir)

if __name__ == "__main__":
    main()
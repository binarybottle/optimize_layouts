#!/usr/bin/env python3
"""Normalize columns in a table with automatic distribution detection"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import argparse

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
    std_dev = np.std(non_zeros)
    
    if std_dev == 0:
        return np.zeros_like(scores)
        
    skew = np.mean(((non_zeros - mean) / std_dev) ** 3)
    
    # Calculate ratio between consecutive sorted values
    sorted_nonzero = np.sort(non_zeros)
    if len(sorted_nonzero) > 1:
        ratios = sorted_nonzero[1:] / sorted_nonzero[:-1]
        median_ratio = np.median(ratios)
    else:
        median_ratio = 1.0
    
    # Detect distribution type and apply appropriate normalization
    zero_ratio = len(scores[scores == 0]) / len(scores)
    
    if zero_ratio > 0.3:
        # Sparse distribution with many zeros
        print(f"{name}: Sparse distribution detected (zero ratio: {zero_ratio:.2f})")
        norm_scores = np.sqrt(scores)
        return (norm_scores - np.min(norm_scores)) / (np.max(norm_scores) - np.min(norm_scores))
        
    elif skew > 2 or median_ratio > 1.5:
        # Heavy-tailed/exponential/zipfian distribution
        print(f"{name}: Heavy-tailed distribution detected (skew: {skew:.2f}, ratio: {median_ratio:.2f})")
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
        if q99 == q1:
            return np.zeros_like(scores)
        scaled = (scores - q1) / (q99 - q1)
        return np.clip(scaled, 0, 1)

def normalize_table(input_file, output_file=None, method='auto', columns=None, verbose=True):
    """
    Normalize specified columns in a table
    
    Args:
        input_file: Path to input CSV/TSV file
        output_file: Path to output file (default: adds '_normalized' suffix)
        method: 'auto', 'standard', 'minmax', or 'robust'
        columns: List of column names to normalize (default: all numeric columns)
        verbose: Print distribution detection info
    """
    # Read the file
    sep = '\t' if input_file.endswith('.tsv') else ','
    df = pd.read_csv(input_file, sep=sep)
    
    # Determine columns to normalize
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    print(f"Normalizing columns: {columns}")
    
    # Apply normalization method
    if method == 'auto':
        # Use automatic distribution detection
        for col in columns:
            if verbose:
                print(f"\nAnalyzing column: {col}")
            df[col] = detect_and_normalize_distribution(df[col].values, col)
            
    else:
        # Use traditional scaling methods
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown method: {method}")
            
        df[columns] = scaler.fit_transform(df[columns])
    
    # Set output filename
    if output_file is None:
        base = input_file.rsplit('.', 1)[0]
        ext = input_file.rsplit('.', 1)[1] if '.' in input_file else 'csv'
        output_file = f"{base}_normalized.{ext}"
    
    # Save result
    df.to_csv(output_file, sep=sep, index=False)
    print(f"\nNormalized table saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize table columns with distribution detection")
    parser.add_argument("input_file", help="Input CSV/TSV file")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-m", "--method", 
                       choices=['auto', 'standard', 'minmax', 'robust'], 
                       default='auto', 
                       help="Normalization method (auto detects distribution)")
    parser.add_argument("-c", "--columns", nargs='+', 
                       help="Specific columns to normalize")
    parser.add_argument("-q", "--quiet", action='store_true',
                       help="Suppress distribution detection output")
    
    args = parser.parse_args()
    normalize_table(args.input_file, args.output, args.method, 
                   args.columns, verbose=not args.quiet)
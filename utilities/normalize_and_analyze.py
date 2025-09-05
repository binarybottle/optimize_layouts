#!/usr/bin/env python3
"""
Normalize and analyze data

This script can normalize data with automatic distribution detection and/or
analyze the comparison between raw and normalized data through visual plots
and statistical analysis.

Features:
- Normalize data with automatic distribution detection
- Compare raw vs normalized data with visualizations
- Generate summary reports and statistics
- Analyze participation in pairs (if pair data is provided)
- Flexible workflow: normalize-only, analyze-only, or normalize-then-analyze

Usage:
    # Normalize only
    python normalize_and_analyze.py normalize input.csv
    
    # Analyze existing raw vs normalized data
    python normalize_and_analyze.py analyze --raw raw.csv --norm normalized.csv
    
    # Normalize and immediately analyze
    python normalize_and_analyze.py normalize-analyze input.csv
    
    # Full workflow with custom settings
    python normalize_and_analyze.py normalize-analyze input.csv --method auto --output-dir results/

Examples:
    python normalize_and_analyze.py normalize item_pairs.csv
    python normalize_and_analyze.py analyze --raw item_pairs.csv --norm item_pairs_normalized.csv
    python normalize_and_analyze.py normalize-analyze position_pairs.csv --output-dir plots/
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Set random seed for reproducible plots
np.random.seed(42)

class DataNormalizer:
    """Handles data normalization with automatic distribution detection."""
    
    def detect_and_normalize_distribution(self, scores: np.ndarray, name: str = '') -> np.ndarray:
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

    def normalize_table(self, input_file, output_file=None, method='auto', columns=None, verbose=True):
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
                df[col] = self.detect_and_normalize_distribution(df[col].values, col)
                
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
        
        return df, output_file


class DataAnalyzer:
    """Handles analysis and comparison of raw vs normalized data."""
    
    def detect_data_type(self, df):
        """Detect the type of data based on column names."""
        columns = [col.lower() for col in df.columns]
        
        if any('item_pair' in col for col in columns):
            return 'item_pair_scores'
        elif any('position_pair' in col for col in columns):
            return 'position_pair_scores'
        elif any('item' in col for col in columns):
            return 'item_scores'
        elif any('position' in col for col in columns):
            return 'position_scores'
        else:
            # Try to infer from data patterns
            if len(df.columns) >= 2:
                first_col = df.iloc[:, 0].astype(str)
                # Check if first column looks like pairs (2-character strings)
                if first_col.str.len().eq(2).all():
                    return 'item_pair_scores'  # Default to item pairs
            return 'unknown'

    def get_title_for_type(self, data_type):
        """Return a properly formatted title for a given data type."""
        titles = {
            'item_scores': 'Item scores',
            'item_pair_scores': 'Item-pair scores',
            'position_scores': 'Position scores',
            'position_pair_scores': 'Position-pair scores',
            'unknown': 'Data scores'
        }
        return titles.get(data_type, data_type.replace("_", " ").title())

    def load_and_validate_data(self, raw_file, norm_file):
        """Load and validate raw and normalized CSV files."""
        # Load raw data
        try:
            raw_df = pd.read_csv(raw_file)
            print(f"Loaded raw data: {len(raw_df)} rows, {len(raw_df.columns)} columns")
        except Exception as e:
            raise ValueError(f"Could not load raw file '{raw_file}': {e}")
        
        # Load normalized data
        try:
            norm_df = pd.read_csv(norm_file)
            print(f"Loaded normalized data: {len(norm_df)} rows, {len(norm_df.columns)} columns")
        except Exception as e:
            raise ValueError(f"Could not load normalized file '{norm_file}': {e}")
        
        # Detect data type
        data_type = self.detect_data_type(norm_df)
        print(f"Detected data type: {data_type}")
        
        # Find score columns
        raw_score_col = None
        norm_score_col = None
        
        # Look for score column in raw data
        for col in raw_df.columns:
            if any(keyword in col.lower() for keyword in ['score', 'frequency', 'comfort', 'value']):
                raw_score_col = col
                break
        
        # Look for score column in normalized data
        for col in norm_df.columns:
            if any(keyword in col.lower() for keyword in ['score', 'frequency', 'comfort', 'value']):
                norm_score_col = col
                break
        
        if raw_score_col is None:
            # Use last column as default
            raw_score_col = raw_df.columns[-1]
            print(f"Warning: No score column found in raw data, using '{raw_score_col}'")
        
        if norm_score_col is None:
            # Use last column as default
            norm_score_col = norm_df.columns[-1]
            print(f"Warning: No score column found in normalized data, using '{norm_score_col}'")
        
        return raw_df, norm_df, data_type, raw_score_col, norm_score_col

    def create_distribution_comparison(self, raw_df, norm_df, raw_score_col, norm_score_col, 
                                     data_type, output_dir):
        """Create distribution comparison plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Score distribution: {self.get_title_for_type(data_type)}', fontsize=16)
        
        # Raw distribution
        raw_values = raw_df[raw_score_col].dropna()
        ax1.hist(raw_values, bins=30, alpha=0.7, color='black', edgecolor='gray')
        ax1.set_title('Raw data')
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Count')
        
        # Use log scale if range is large
        if len(raw_values) > 0 and raw_values.max() / max(raw_values.min(), 1e-10) > 1000:
            ax1.set_xscale('log')
            ax1.set_title('Raw data (log scale)')
        
        # Normalized distribution
        norm_values = norm_df[norm_score_col].dropna()
        ax2.hist(norm_values, bins=30, alpha=0.7, color='gray', edgecolor='black')
        ax2.set_title('Normalized data')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{data_type}_distribution_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    def create_top_scores_plot(self, norm_df, norm_score_col, data_type, output_dir, n_show=30):
        """Create top scores plot for normalized data."""
        # Sort by score and take top N
        sorted_df = norm_df.sort_values(norm_score_col, ascending=False).head(n_show)
        
        plt.figure(figsize=(12, 8))
        x = np.arange(len(sorted_df))
        plt.bar(x, sorted_df[norm_score_col], color='black', alpha=0.8, 
                edgecolor='gray', linewidth=0.5)
        
        plt.title(f'{self.get_title_for_type(data_type)}: normalized (top {len(sorted_df)})')
        plt.ylabel('Normalized score')
        plt.xlabel('Items')
        
        # Set x-tick labels (use first column as labels)
        if len(sorted_df) > 0:
            labels = sorted_df.iloc[:, 0].astype(str)
            # Convert to uppercase if it looks like position data
            if data_type.startswith('position'):
                labels = labels.str.upper()
            
            plt.xticks(x, labels, rotation=45 if len(labels) > 15 else 0)
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'{data_type}_normalized_top{len(sorted_df)}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    def analyze_pair_participation(self, raw_df, norm_df, raw_score_col, norm_score_col, 
                                 data_type, output_dir):
        """Analyze participation in pairs if this is pair data."""
        if not data_type.endswith('_pair_scores'):
            return
        
        print(f"Analyzing {data_type} participation...")
        
        # Get the pair column (first column)
        pair_col = raw_df.columns[0]
        
        # Extract individual entities from pairs
        entities = set()
        for pair_str in raw_df[pair_col].astype(str):
            if len(pair_str) >= 2:
                entities.add(pair_str[0].lower())
                entities.add(pair_str[1].lower())
        
        # Analyze each entity
        results = []
        for entity in sorted(entities):
            # Find pairs containing this entity in raw data
            entity_raw_scores = []
            entity_norm_scores = []
            
            for i, pair_str in enumerate(raw_df[pair_col].astype(str)):
                if len(pair_str) >= 2 and entity in pair_str.lower():
                    entity_raw_scores.append(raw_df.iloc[i][raw_score_col])
                    
                    # Find corresponding normalized score
                    if i < len(norm_df):
                        entity_norm_scores.append(norm_df.iloc[i][norm_score_col])
            
            if entity_raw_scores:
                result = {
                    'entity': entity.upper() if data_type.startswith('position') else entity,
                    'pair_count': len(entity_raw_scores),
                    'raw_min': min(entity_raw_scores),
                    'raw_max': max(entity_raw_scores),
                    'norm_min': min(entity_norm_scores) if entity_norm_scores else 0,
                    'norm_max': max(entity_norm_scores) if entity_norm_scores else 0,
                }
                results.append(result)
        
        if results:
            # Save participation analysis
            results_df = pd.DataFrame(results)
            entity_type = 'item' if data_type.startswith('item') else 'position'
            output_path = os.path.join(output_dir, f'{entity_type}_pair_participation.csv')
            results_df.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")
            
            # Create visualization
            self.create_participation_plot(results_df, data_type, output_dir)

    def create_participation_plot(self, results_df, data_type, output_dir):
        """Create a plot showing entity participation in pairs."""
        if results_df.empty:
            return
        
        # Sort by normalized max score
        sorted_df = results_df.sort_values('norm_max', ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(sorted_df) * 0.3)))
        
        y_pos = np.arange(len(sorted_df))
        
        # Create horizontal bars showing score ranges
        for i, (_, row) in enumerate(sorted_df.iterrows()):
            width = row['norm_max'] - row['norm_min']
            ax.barh(i, width, left=row['norm_min'], height=0.6, 
                    alpha=0.8, color='black', edgecolor='gray', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_df['entity'])
        ax.set_xlabel('Normalized score range')
        ax.set_title(f'{self.get_title_for_type(data_type)}: Entity participation ranges')
        ax.grid(True, alpha=0.3)
        
        entity_type = 'item' if data_type.startswith('item') else 'position'
        output_path = os.path.join(output_dir, f'{entity_type}_pair_score_ranges.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    def create_summary_report(self, raw_df, norm_df, raw_score_col, norm_score_col, 
                             data_type, raw_file, norm_file, output_dir):
        """Create a summary report."""
        report_path = os.path.join(output_dir, "comparison_summary.txt")
        
        with open(report_path, 'w') as f:
            f.write("=== DATA COMPARISON SUMMARY REPORT ===\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Raw file: {raw_file}\n")
            f.write(f"Normalized file: {norm_file}\n")
            f.write(f"Data type: {data_type}\n\n")
            
            # Raw data statistics
            raw_values = raw_df[raw_score_col].dropna()
            f.write("RAW DATA STATISTICS:\n")
            f.write(f"  Count: {len(raw_values)}\n")
            if len(raw_values) > 0:
                f.write(f"  Min: {raw_values.min():.6g}\n")
                f.write(f"  Max: {raw_values.max():.6g}\n")
                f.write(f"  Mean: {raw_values.mean():.6g}\n")
                f.write(f"  Std: {raw_values.std():.6g}\n")
            
            # Normalized data statistics
            norm_values = norm_df[norm_score_col].dropna()
            f.write("\nNORMALIZED DATA STATISTICS:\n")
            f.write(f"  Count: {len(norm_values)}\n")
            if len(norm_values) > 0:
                f.write(f"  Min: {norm_values.min():.6g}\n")
                f.write(f"  Max: {norm_values.max():.6g}\n")
                f.write(f"  Mean: {norm_values.mean():.6g}\n")
                f.write(f"  Std: {norm_values.std():.6g}\n")
            
            # Data quality checks
            f.write("\nDATA QUALITY:\n")
            f.write(f"  Raw missing values: {raw_df[raw_score_col].isna().sum()}\n")
            f.write(f"  Normalized missing values: {norm_df[norm_score_col].isna().sum()}\n")
            f.write(f"  Row count match: {len(raw_df) == len(norm_df)}\n")
        
        print(f"Summary report saved: {report_path}")

    def analyze_data(self, raw_file, norm_file, output_dir):
        """Complete analysis workflow."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and validate data
        print("Loading and validating data...")
        raw_df, norm_df, data_type, raw_score_col, norm_score_col = self.load_and_validate_data(
            raw_file, norm_file)
        
        # Create comparison plots
        print("Creating distribution comparison...")
        self.create_distribution_comparison(raw_df, norm_df, raw_score_col, norm_score_col, 
                                           data_type, output_dir)
        
        print("Creating top scores plot...")
        self.create_top_scores_plot(norm_df, norm_score_col, data_type, output_dir)
        
        # Analyze pairs if applicable
        self.analyze_pair_participation(raw_df, norm_df, raw_score_col, norm_score_col, 
                                       data_type, output_dir)
        
        # Create summary report
        print("Creating summary report...")
        self.create_summary_report(raw_df, norm_df, raw_score_col, norm_score_col, 
                                  data_type, raw_file, norm_file, output_dir)
        
        print(f"\nAnalysis complete! Output saved to: {output_dir}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Normalize and analyze data with automatic distribution detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Normalize only
    python normalize_and_analyze.py normalize input.csv
    
    # Analyze existing files
    python normalize_and_analyze.py analyze --raw raw.csv --norm normalized.csv
    
    # Normalize and analyze in one go
    python normalize_and_analyze.py normalize-analyze input.csv --output-dir results/
        """)
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Normalize-only mode
    norm_parser = subparsers.add_parser('normalize', help='Normalize data only')
    norm_parser.add_argument('input_file', help='Input CSV/TSV file')
    norm_parser.add_argument('-o', '--output', help='Output file path')
    norm_parser.add_argument('-m', '--method', 
                           choices=['auto', 'standard', 'minmax', 'robust'], 
                           default='auto', 
                           help='Normalization method')
    norm_parser.add_argument('-c', '--columns', nargs='+', 
                           help='Specific columns to normalize')
    norm_parser.add_argument('-q', '--quiet', action='store_true',
                           help='Suppress distribution detection output')
    
    # Analyze-only mode
    analyze_parser = subparsers.add_parser('analyze', help='Analyze existing raw vs normalized data')
    analyze_parser.add_argument('--raw', type=str, required=True, help='Path to raw CSV file')
    analyze_parser.add_argument('--norm', type=str, required=True, help='Path to normalized CSV file')
    analyze_parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    
    # Normalize-and-analyze mode
    both_parser = subparsers.add_parser('normalize-analyze', help='Normalize data then analyze')
    both_parser.add_argument('input_file', help='Input CSV/TSV file')
    both_parser.add_argument('-m', '--method', 
                           choices=['auto', 'standard', 'minmax', 'robust'], 
                           default='auto', 
                           help='Normalization method')
    both_parser.add_argument('-c', '--columns', nargs='+', 
                           help='Specific columns to normalize')
    both_parser.add_argument('--output-dir', type=str, default='output', 
                           help='Output directory for analysis')
    both_parser.add_argument('--save-normalized', type=str, 
                           help='Save normalized data to specific file')
    both_parser.add_argument('-q', '--quiet', action='store_true',
                           help='Suppress distribution detection output')
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        return 1
    
    try:
        if args.mode == 'normalize':
            # Normalize only
            normalizer = DataNormalizer()
            normalizer.normalize_table(
                args.input_file, 
                args.output, 
                args.method, 
                args.columns, 
                verbose=not args.quiet
            )
            
        elif args.mode == 'analyze':
            # Analyze only
            if not os.path.exists(args.raw):
                raise FileNotFoundError(f"Raw file not found: {args.raw}")
            if not os.path.exists(args.norm):
                raise FileNotFoundError(f"Normalized file not found: {args.norm}")
            
            analyzer = DataAnalyzer()
            analyzer.analyze_data(args.raw, args.norm, args.output_dir)
            
        elif args.mode == 'normalize-analyze':
            # Normalize and analyze
            if not os.path.exists(args.input_file):
                raise FileNotFoundError(f"Input file not found: {args.input_file}")
            
            print("=== STEP 1: NORMALIZING DATA ===")
            normalizer = DataNormalizer()
            _, norm_file = normalizer.normalize_table(
                args.input_file,
                args.save_normalized,
                args.method,
                args.columns,
                verbose=not args.quiet
            )
            
            print("\n=== STEP 2: ANALYZING DATA ===")
            analyzer = DataAnalyzer()
            analyzer.analyze_data(args.input_file, norm_file, args.output_dir)
            
            print(f"\n=== COMPLETE WORKFLOW FINISHED ===")
            print(f"Normalized data: {norm_file}")
            print(f"Analysis output: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
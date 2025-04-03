# optimize_layouts/analyze_inputs.py
"""
Analyze and plot item/item_pair frequencies and scores 
and position/position_pair scores.

Usage:
>> python plots.py --config path/to/config.yaml

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add parent directory to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration loading function from optimize_layout
try:
    from optimize_layout import load_config, detect_and_normalize_distribution
except ImportError:
    print("Error: Could not import from optimize_layout. Make sure the file exists and is accessible.")
    print("Current directory:", os.getcwd())
    print("Python path:", sys.path)
    sys.exit(1)

def setup_paths(config):
    """Set up all paths based on configuration."""
    # Extract input paths from config
    input_paths = {
        'item_scores': config['paths']['input']['item_scores_file'],
        'item_pair_scores': config['paths']['input']['item_pair_scores_file'],
        'position_scores': config['paths']['input']['position_scores_file'],
        'position_pair_scores': config['paths']['input']['position_pair_scores_file']
    }
    
    # Create output plot directory (next to layout results directory)
    base_output_dir = os.path.dirname(config['paths']['output']['layout_results_folder'])
    plot_dir = os.path.join(base_output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Using input files from paths specified in config:")
    for name, path in input_paths.items():
        print(f"  - {name}: {path}")
    print(f"Created/verified plot directory: {plot_dir}")
    
    return input_paths, plot_dir

def plot_item_frequencies(input_path, plot_dir):
    """Plot item frequencies from CSV data"""
    print("Generating item frequencies plot...")
    
    # Load item frequencies
    df = pd.read_csv(input_path)
    # Using correct column names based on the actual CSV structure
    items = df['item'].values
    frequencies = df['score'].values / np.sum(df['score'].values)  # Normalize to get frequencies
    
    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")
    
    # Create scatter plot
    plt.scatter(range(len(items)), frequencies * 100, s=100, alpha=0.6)
    plt.plot(range(len(items)), frequencies * 100, 'b-', alpha=0.3)
    
    # Add labels for each point
    for i, (item, freq) in enumerate(zip(items, frequencies)):
        plt.annotate(f'{item.upper()}\n{freq*100:.1f}%',
                    (i, freq*100),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')
    
    plt.title('Item Frequencies Distribution', fontsize=14, pad=20)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Frequency (%)', fontsize=12)
    plt.xticks([])
    
    plt.tight_layout()
    output_path = os.path.join(plot_dir, 'item_frequencies.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved item frequencies plot to: {output_path}")
    plt.close()

def plot_item_pair_frequencies(input_path, plot_dir, n_pairs=None, filename_suffix=''):
    """Plot item pair frequencies from CSV data"""
    print(f"Generating item pair frequencies plot{' (top ' + str(n_pairs) + ')' if n_pairs else ''}...")
    
    # Load item pair frequencies
    df = pd.read_csv(input_path)
    # Using correct column names based on the actual CSV structure
    pairs = df['item_pair'].values
    # Normalize scores to get frequencies
    total_score = np.sum(df['score'].values)
    frequencies = df['score'].values / total_score
    
    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")
    
    # Sort pairs by frequency
    sorted_indices = np.argsort(frequencies)[::-1]
    if n_pairs:
        sorted_indices = sorted_indices[:n_pairs]
    sorted_freqs = frequencies[sorted_indices] * 100
    sorted_pairs = pairs[sorted_indices]
    
    # Create scatter plot
    plt.scatter(range(len(sorted_freqs)), sorted_freqs, s=20, alpha=0.4)
    plt.plot(range(len(sorted_freqs)), sorted_freqs, 'b-', alpha=0.2)
    
    # Calculate indices for labels
    if n_pairs:
        n_labels = min(20, n_pairs)
    else:
        n_labels = 20
    label_indices = np.linspace(0, len(sorted_freqs) - 1, n_labels).astype(int)
    
    # Add labels
    for i in label_indices:
        if i < len(sorted_freqs):
            pair = sorted_pairs[i]
            label = f"{pair.upper()}\n{sorted_freqs[i]:.2f}%"
            plt.annotate(label,
                        (i, sorted_freqs[i]),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=8)
    
    title_suffix = f" (Top {n_pairs})" if n_pairs else ""
    plt.title(f'Item Pair Frequencies Distribution{title_suffix}', fontsize=14, pad=20)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Frequency (%)', fontsize=12)
    plt.yscale('log')
    
    plt.tight_layout()
    output_path = os.path.join(plot_dir, f'item_pair_frequencies{filename_suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved item pair frequencies plot to: {output_path}")
    plt.close()

def load_and_normalize_scores(input_paths, config):
    """Load raw scores from data files and normalize them, returning both normalized scores and scaling methods"""
    print("\nLoading and analyzing CSV files...")
    
    # Extract weights from config for reference
    item_weight = config['optimization']['scoring']['item_weight']
    item_pair_weight = config['optimization']['scoring']['item_pair_weight']
    
    # Load raw scores
    try:
        item_df = pd.read_csv(input_paths['item_scores'])
        print(f"Item scores: {item_df.shape[0]} rows with columns {item_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading item scores: {e}")
        raise
        
    try:
        item_pair_df = pd.read_csv(input_paths['item_pair_scores'])
        print(f"Item pair scores: {item_pair_df.shape[0]} rows with columns {item_pair_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading item pair scores: {e}")
        raise
        
    try:
        position_df = pd.read_csv(input_paths['position_scores'])
        print(f"Position scores: {position_df.shape[0]} rows with columns {position_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading position scores: {e}")
        raise
        
    try:
        position_pair_df = pd.read_csv(input_paths['position_pair_scores'])
        print(f"Position pair scores: {position_pair_df.shape[0]} rows with columns {position_pair_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading position pair scores: {e}")
        raise
    
    # Extract scores using correct column names
    item_scores = item_df['score'].values
    item_pair_scores = item_pair_df['score'].values
    position_scores = position_df['score'].values
    position_pair_scores = position_pair_df['score'].values
    
    # Store original data for plotting
    original_scores = {
        'item': {'data': item_scores, 'label': item_df['item'].values},
        'item_pair': {'data': item_pair_scores, 'label': item_pair_df['item_pair'].values},
        'position': {'data': position_scores, 'label': position_df['position'].values},
        'position_pair': {'data': position_pair_scores, 'label': position_pair_df['position_pair'].values}
    }
    
    # Print summary statistics
    print("\nSummary statistics of raw scores:")
    for name, data in original_scores.items():
        scores = data['data']
        print(f"{name} scores: min={np.min(scores)}, max={np.max(scores)}, mean={np.mean(scores):.2f}, median={np.median(scores)}")
    
    # Normalize all score types (using the same function as optimize_layout.py)
    print("\nNormalizing scores:")
    normalized_scores = {}
    
    # Normalize item scores
    norm_scores_item = detect_and_normalize_distribution(item_scores, 'item scores')
    normalized_scores['item'] = norm_scores_item
    print("  - original:", min(item_scores), "to", max(item_scores))
    print("  - normalized:", min(norm_scores_item), "to", max(norm_scores_item))
    
    # Normalize item pair scores
    norm_scores_item_pair = detect_and_normalize_distribution(item_pair_scores, 'item_pair scores')
    normalized_scores['item_pair'] = norm_scores_item_pair
    print("  - original:", min(item_pair_scores), "to", max(item_pair_scores))
    print("  - normalized:", min(norm_scores_item_pair), "to", max(norm_scores_item_pair))
    
    # Normalize position scores
    norm_scores_position = detect_and_normalize_distribution(position_scores, 'position scores')
    normalized_scores['position'] = norm_scores_position
    print("  - original:", min(position_scores), "to", max(position_scores))
    print("  - normalized:", min(norm_scores_position), "to", max(norm_scores_position))
    
    # Normalize position pair scores
    norm_scores_position_pair = detect_and_normalize_distribution(position_pair_scores, 'position_pair scores')
    normalized_scores['position_pair'] = norm_scores_position_pair
    print("  - original:", min(position_pair_scores), "to", max(position_pair_scores))
    print("  - normalized:", min(norm_scores_position_pair), "to", max(norm_scores_position_pair))
    
    return normalized_scores, original_scores

def plot_score_distributions(normalized_scores, original_scores, plot_dir):
    """Plot distributions of both original and normalized scores"""
    print("\nGenerating score distribution plots...")
    
    # Create figure with 2 rows, 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('Score Distributions: Original vs Normalized', fontsize=16)
    
    score_types = ['item', 'item_pair', 'position', 'position_pair']
    row_titles = ['Original Scores', 'Normalized Scores [0-1]']
    
    for row in range(2):
        for col, score_type in enumerate(score_types):
            ax = axes[row, col]
            
            if row == 0:  # Original scores
                data = original_scores[score_type]['data']
                title = f'Original {score_type.replace("_", " ").title()}'
                
                # Use log scale for original item_pair scores if heavy-tailed
                if score_type == 'item_pair':
                    if np.min(data) > 0:  # Only use log scale if all values are positive
                        ax.set_xscale('log')
            else:  # Normalized scores
                data = normalized_scores[score_type]
                title = f'Normalized {score_type.replace("_", " ").title()}'
            
            # Plot histogram
            sns.histplot(data, ax=ax, kde=True)
            ax.set_title(title)
            ax.set_xlabel('Score')
            ax.set_ylabel('Count')
            
    plt.tight_layout()
    output_path = os.path.join(plot_dir, 'score_distributions_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved score distributions plot to: {output_path}")
    plt.close()

def plot_normalization_comparison(normalized_scores, original_scores, plot_dir):
    """Plot comparison of original vs normalized scores to visualize the normalization effect"""
    print("\nGenerating normalization comparison plots...")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Normalization Effect on Scores', fontsize=16)
    
    score_types = ['item', 'item_pair', 'position', 'position_pair']
    # Determine normalization approach based on detected distribution
    dist_types = {
        'item': 'auto-detected', 
        'item_pair': 'auto-detected',
        'position': 'auto-detected', 
        'position_pair': 'auto-detected'
    }
    
    for (score_type, ax) in zip(score_types, axes.flat):
        # Get original data and labels
        orig_data = original_scores[score_type]['data']
        labels = original_scores[score_type]['label']
        norm_data = normalized_scores[score_type]
        method = dist_types[score_type]
        
        # Sort by original score for better visualization
        sort_idx = np.argsort(orig_data)[::-1]  # Descending
        sorted_orig = orig_data[sort_idx]
        sorted_norm = norm_data[sort_idx]
        sorted_labels = labels[sort_idx]
        
        # Limit to top 30 for readability
        n_show = min(30, len(sorted_orig))
        
        # Create bar chart
        x = np.arange(n_show)
        width = 0.35
        
        # Plot original scores
        orig_bars = ax.bar(x - width/2, sorted_orig[:n_show], width, label='Original')
        
        # Create second y-axis for normalized scores
        ax2 = ax.twinx()
        norm_bars = ax2.bar(x + width/2, sorted_norm[:n_show], width, color='orange', 
                           label=f'Normalized ({method})')
        
        # Set labels and title
        ax.set_title(f'{score_type.replace("_", " ").title()} - Original vs Normalized')
        ax.set_xlabel('Items')
        ax.set_ylabel('Original Score', color='blue')
        ax2.set_ylabel('Normalized Score [0-1]', color='orange')
        
        # Set x-ticks as labels
        if score_type in ['item', 'item_pair']:  # For single items or pairs
            ax.set_xticks(x)
            ax.set_xticklabels(sorted_labels[:n_show], rotation=45 if score_type.endswith('pair') else 0)
        else:
            # For position and position_pair, show every 3rd label to avoid crowding
            step = 3
            indices = np.arange(0, n_show, step)
            ax.set_xticks(indices)
            ax.set_xticklabels([sorted_labels[i] for i in indices], rotation=45)
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Optional: logarithmic scale for original scores if they span many orders of magnitude
        if np.max(orig_data) / np.min(orig_data) > 1000 and np.min(orig_data) > 0:
            ax.set_yscale('log')
    
    plt.tight_layout()
    output_path = os.path.join(plot_dir, 'normalization_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved normalization comparison plot to: {output_path}")
    plt.close()

def plot_score_heatmaps(normalized_scores, original_scores, plot_dir):
    """Plot heatmaps of both original and normalized pair scores"""
    print("\nGenerating score heatmap plots...")
    
    # Create figure with 2 rows, 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Pair Scores Heatmaps: Original vs Normalized', fontsize=16)
    
    # Plot item pair scores - reshape to square matrix if possible
    item_pair_data = original_scores['item_pair']['data']
    item_pair_labels = original_scores['item_pair']['label']
    
    # Get unique elements from pairs
    unique_items = set()
    for pair in item_pair_labels:
        if len(pair) == 2:  # Ensure it's a valid pair
            unique_items.update(pair)
    unique_items = sorted(list(unique_items))
    n_unique_items = len(unique_items)
    
    if n_unique_items > 0:
        # Create item-to-index mapping
        item_to_idx = {item: i for i, item in enumerate(unique_items)}
        
        # Initialize matrices
        orig_matrix = np.full((n_unique_items, n_unique_items), np.nan)
        norm_matrix = np.full((n_unique_items, n_unique_items), np.nan)

        # Fill matrices
        n_filled = 0
        for pair, orig_score, norm_score in zip(item_pair_labels, item_pair_data, normalized_scores['item_pair']):
            if len(pair) == 2:  # Ensure it's a valid pair
                i, j = item_to_idx[pair[0]], item_to_idx[pair[1]]
                orig_matrix[i, j] = orig_score
                norm_matrix[i, j] = norm_score
                n_filled += 1
        
        print(f"Created item pair matrices ({n_unique_items}x{n_unique_items}) with {n_filled} filled cells")
        
        # Plot original matrix
        axes[0, 0].set_title('Original Item Pair Scores')
        im1 = sns.heatmap(orig_matrix, ax=axes[0, 0], cmap='viridis', 
                           xticklabels=unique_items, yticklabels=unique_items)
        
        # Plot normalized matrix
        axes[1, 0].set_title(f'Normalized Item Pair Scores')
        im2 = sns.heatmap(norm_matrix, ax=axes[1, 0], cmap='viridis',
                           xticklabels=unique_items, yticklabels=unique_items)
    else:
        # Fallback if creating a square matrix isn't possible
        axes[0, 0].set_title('Original Item Pair Scores')
        im1 = sns.heatmap(item_pair_data.reshape(1, -1), ax=axes[0, 0], cmap='viridis')
        
        axes[1, 0].set_title(f'Normalized Item Pair Scores')
        im2 = sns.heatmap(normalized_scores['item_pair'].reshape(1, -1), ax=axes[1, 0], cmap='viridis')
    
    # Plot position pair scores - reshape to square matrix if possible
    position_pair_data = original_scores['position_pair']['data']
    position_pair_labels = original_scores['position_pair']['label']
    
    # Get unique positions from position pairs
    unique_positions = set()
    for pair in position_pair_labels:
        if len(pair) == 2:  # Ensure it's a valid pair
            unique_positions.update(pair)
    unique_positions = sorted(list(unique_positions))
    n_positions = len(unique_positions)
    
    if n_positions > 0:
        # Create position-to-index mapping
        position_to_idx = {pos: i for i, pos in enumerate(unique_positions)}
        
        # Initialize matrices
        orig_matrix = np.full((n_positions, n_positions), np.nan)
        norm_matrix = np.full((n_positions, n_positions), np.nan)
        
        # Fill matrices
        n_filled = 0
        for pair, orig_score, norm_score in zip(position_pair_labels, position_pair_data, normalized_scores['position_pair']):
            if len(pair) == 2:  # Ensure it's a valid pair
                i, j = position_to_idx[pair[0]], position_to_idx[pair[1]]
                orig_matrix[i, j] = orig_score
                norm_matrix[i, j] = norm_score
                n_filled += 1
        
        print(f"Created position pair matrices ({n_positions}x{n_positions}) with {n_filled} filled cells")
        
        # Plot original matrix
        axes[0, 1].set_title('Original Position Pair Scores')
        im3 = sns.heatmap(orig_matrix, ax=axes[0, 1], cmap='viridis',
                        mask=np.isnan(orig_matrix),  
                        xticklabels=unique_positions, yticklabels=unique_positions)

        axes[1, 1].set_title('Normalized Position Pair Scores')
        im4 = sns.heatmap(norm_matrix, ax=axes[1, 1], cmap='viridis',
                        mask=np.isnan(norm_matrix),  
                        xticklabels=unique_positions, yticklabels=unique_positions)
    
    plt.tight_layout()
    output_path = os.path.join(plot_dir, 'pair_scores_heatmaps.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved pair scores heatmaps to: {output_path}")
    plt.close()

def main(config_path="config.yaml"):
    """Main function to generate all plots."""
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Setup paths based on configuration
    input_paths, plot_dir = setup_paths(config)
    
    plt.rcParams.update({'font.size': 12})
    
    # Plot frequencies
    plot_item_frequencies(input_paths['item_scores'], plot_dir)
    plot_item_pair_frequencies(input_paths['item_pair_scores'], plot_dir)
    plot_item_pair_frequencies(input_paths['item_pair_scores'], plot_dir, n_pairs=100, filename_suffix='_top100')
    
    # Load and normalize scores
    normalized_scores, original_scores = load_and_normalize_scores(input_paths, config)
    
    # Plot score distributions and normalization comparison
    plot_score_distributions(normalized_scores, original_scores, plot_dir)
    plot_normalization_comparison(normalized_scores, original_scores, plot_dir)
    plot_score_heatmaps(normalized_scores, original_scores, plot_dir)
    
    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate data visualization plots for optimization")
    parser.add_argument("--config", type=str, default="config.yaml", 
                        help="Path to configuration file (default: config.yaml)")
    
    args = parser.parse_args()
    main(args.config)
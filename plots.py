import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create output directory
PLOT_DIR = Path('output/plots')
PLOT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Created/verified plot directory at: {PLOT_DIR.absolute()}")

# Import the exact normalization function from optimize_layout.py for consistency
def detect_and_normalize_distribution(scores: np.ndarray, name: str = '') -> np.ndarray:
    """
    Automatically detect distribution type and apply appropriate normalization.
    Returns scores normalized to [0,1] range.
    
    This is the exact same function used in optimize_layout.py
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
    
    # Detect distribution type
    if len(scores[scores == 0]) / len(scores) > 0.3:
        # Sparse distribution with many zeros
        print(f"{name}: Sparse distribution detected")
        #adjusted_scores = np.where(scores > 0, scores, min_nonzero / 10)
        norm_scores = np.sqrt(scores)  #np.log10(adjusted_scores)
        return (norm_scores - np.min(norm_scores)) / (np.max(norm_scores) - np.min(norm_scores))
    
    elif skew > 2 or np.median(ratios) > 1.5:
        # Heavy-tailed/exponential/zipfian distribution
        print(f"{name}: Heavy-tailed distribution detected")
        norm_scores = np.sqrt(np.abs(scores))  #np.log10(scores + min_nonzero/10)
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

def plot_letter_frequencies():
    """Plot letter frequencies from CSV data"""
    print("Generating letter frequencies plot...")
    
    # Load letter frequencies
    df = pd.read_csv("input/letter_frequencies_english.csv")
    # Using correct column names based on the actual CSV structure
    letters = df['item'].values
    frequencies = df['score'].values / np.sum(df['score'].values)  # Normalize to get frequencies
    
    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")
    
    # Create scatter plot
    plt.scatter(range(len(letters)), frequencies * 100, s=100, alpha=0.6)
    plt.plot(range(len(letters)), frequencies * 100, 'b-', alpha=0.3)
    
    # Add labels for each point
    for i, (letter, freq) in enumerate(zip(letters, frequencies)):
        plt.annotate(f'{letter.upper()}\n{freq*100:.1f}%',
                    (i, freq*100),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')
    
    plt.title('Letter Frequencies in English Text', fontsize=14, pad=20)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Frequency (%)', fontsize=12)
    plt.xticks([])
    
    plt.tight_layout()
    output_path = PLOT_DIR / 'letter_frequencies.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved letter frequencies plot to: {output_path.absolute()}")
    plt.close()

def plot_bigram_frequencies(n_bigrams=None, filename_suffix=''):
    """Plot bigram frequencies from CSV data"""
    print(f"Generating bigram frequencies plot{' (top ' + str(n_bigrams) + ')' if n_bigrams else ''}...")
    
    # Load bigram frequencies
    df = pd.read_csv("input/letter_pair_frequencies_english.csv")
    # Using correct column names based on the actual CSV structure
    bigrams = df['item_pair'].values
    # Normalize scores to get frequencies
    total_score = np.sum(df['score'].values)
    frequencies = df['score'].values / total_score
    
    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")
    
    # Sort bigrams by frequency
    sorted_indices = np.argsort(frequencies)[::-1]
    if n_bigrams:
        sorted_indices = sorted_indices[:n_bigrams]
    sorted_freqs = frequencies[sorted_indices] * 100
    sorted_bigrams = bigrams[sorted_indices]
    
    # Create scatter plot
    plt.scatter(range(len(sorted_freqs)), sorted_freqs, s=20, alpha=0.4)
    plt.plot(range(len(sorted_freqs)), sorted_freqs, 'b-', alpha=0.2)
    
    # Calculate indices for labels
    if n_bigrams:
        n_labels = min(20, n_bigrams)
    else:
        n_labels = 20
    label_indices = np.linspace(0, len(sorted_freqs) - 1, n_labels).astype(int)
    
    # Add labels
    for i in label_indices:
        if i < len(sorted_freqs):
            bigram = sorted_bigrams[i]
            label = f"{bigram.upper()}\n{sorted_freqs[i]:.2f}%"
            plt.annotate(label,
                        (i, sorted_freqs[i]),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=8)
    
    title_suffix = f" (Top {n_bigrams})" if n_bigrams else ""
    plt.title(f'Bigram Frequencies in English Text{title_suffix}', fontsize=14, pad=20)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Frequency (%)', fontsize=12)
    plt.yscale('log')
    
    plt.tight_layout()
    output_path = PLOT_DIR / f'bigram_frequencies{filename_suffix}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved bigram frequencies plot to: {output_path.absolute()}")
    plt.close()

def load_and_normalize_scores():
    """Load raw scores from data files and normalize them, returning both normalized scores and scaling methods"""
    print("\nLoading and analyzing CSV files...")
    
    # Load raw scores
    try:
        letter_df = pd.read_csv("input/letter_frequencies_english.csv")
        print(f"Letter frequencies: {letter_df.shape[0]} rows with columns {letter_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading letter frequencies: {e}")
        raise
        
    try:
        pair_df = pd.read_csv("input/letter_pair_frequencies_english.csv")
        print(f"Letter pair frequencies: {pair_df.shape[0]} rows with columns {pair_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading letter pair frequencies: {e}")
        raise
        
    try:
        position_df = pd.read_csv("input/key_comfort_estimates.csv")
        print(f"Key comfort estimates: {position_df.shape[0]} rows with columns {position_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading key comfort estimates: {e}")
        raise
        
    try:
        position_pair_df = pd.read_csv("input/key_pair_comfort_estimates.csv")
        print(f"Key pair comfort estimates: {position_pair_df.shape[0]} rows with columns {position_pair_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading key pair comfort estimates: {e}")
        raise
    
    # Extract scores using correct column names
    item_scores = letter_df['score'].values
    item_pair_scores = pair_df['score'].values
    position_scores = position_df['score'].values
    position_pair_scores = position_pair_df['score'].values
    
    # Store original data for plotting
    original_scores = {
        'item': {'data': item_scores, 'label': letter_df['item'].values},
        'item_pair': {'data': item_pair_scores, 'label': pair_df['item_pair'].values},
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
    norm_data_items = []
    
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

def plot_score_distributions():
    """Plot distributions of both original and normalized scores"""
    print("\nGenerating score distribution plots...")
    normalized_scores, original_scores = load_and_normalize_scores()
    
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
    output_path = PLOT_DIR / 'score_distributions_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved score distributions plot to: {output_path.absolute()}")
    plt.close()

def plot_normalization_comparison():
    """Plot comparison of original vs normalized scores to visualize the normalization effect"""
    print("\nGenerating normalization comparison plots...")
    normalized_scores, original_scores = load_and_normalize_scores()
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Normalization Effect on Scores', fontsize=16)
    
    score_types = ['item', 'item_pair', 'position', 'position_pair']
    dist_types = {
        'item': 'robust', 
        'item_pair': 'log10',
        'position': 'linear', 
        'position_pair': 'linear'
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
        if score_type in ['item', 'item_pair']:  # For single letters or pairs
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
        
        # Optional: logarithmic scale for original scores if heavy-tailed
        if method == 'log10' and np.min(sorted_orig) > 0:
            ax.set_yscale('log')
    
    plt.tight_layout()
    output_path = PLOT_DIR / 'normalization_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved normalization comparison plot to: {output_path.absolute()}")
    plt.close()

def plot_score_heatmaps():
    """Plot heatmaps of both original and normalized pair scores"""
    print("\nGenerating score heatmap plots...")
    normalized_scores, original_scores = load_and_normalize_scores()
    
    # Create figure with 2 rows, 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Pair Scores Heatmaps: Original vs Normalized', fontsize=16)
    
    # Plot item pair scores - reshape to square matrix if possible
    item_pair_data = original_scores['item_pair']['data']
    item_pair_labels = original_scores['item_pair']['label']
    
    # Get unique letters from bigrams
    unique_letters = set()
    for bigram in item_pair_labels:
        if len(bigram) == 2:  # Ensure it's a valid bigram
            unique_letters.update(bigram)
    unique_letters = sorted(list(unique_letters))
    n_letters = len(unique_letters)
    
    if n_letters > 0:
        # Create letter-to-index mapping
        letter_to_idx = {letter: i for i, letter in enumerate(unique_letters)}
        
        # Initialize matrices
        orig_matrix = np.full((n_letters, n_letters), np.nan)
        norm_matrix = np.full((n_letters, n_letters), np.nan)

        # Fill matrices
        n_filled = 0
        for bigram, orig_score, norm_score in zip(item_pair_labels, item_pair_data, normalized_scores['item_pair']):
            if len(bigram) == 2:  # Ensure it's a valid bigram
                i, j = letter_to_idx[bigram[0]], letter_to_idx[bigram[1]]
                orig_matrix[i, j] = orig_score
                norm_matrix[i, j] = norm_score
                n_filled += 1
        
        print(f"Created item pair matrices ({n_letters}x{n_letters}) with {n_filled} filled cells")
        
        # Plot original matrix
        axes[0, 0].set_title('Original Item Pair Scores')
        im1 = sns.heatmap(orig_matrix, ax=axes[0, 0], cmap='viridis', 
                           xticklabels=unique_letters, yticklabels=unique_letters)
        
        # Plot normalized matrix
        axes[1, 0].set_title(f'Normalized Item Pair Scores')
        im2 = sns.heatmap(norm_matrix, ax=axes[1, 0], cmap='viridis',
                           xticklabels=unique_letters, yticklabels=unique_letters)
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
    output_path = PLOT_DIR / 'pair_scores_heatmaps_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved pair scores heatmaps to: {output_path.absolute()}")
    plt.close()

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 12})
    
    # Plot frequencies
    plot_letter_frequencies()
    plot_bigram_frequencies()
    plot_bigram_frequencies(n_bigrams=100, filename_suffix='_top100')
    
    # Plot score distributions and normalization comparison
    plot_score_distributions()
    plot_normalization_comparison()
    plot_score_heatmaps()
    
    print("\nAll plots generated successfully!")
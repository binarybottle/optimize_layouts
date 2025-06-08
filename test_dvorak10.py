#!/usr/bin/env python3
"""
Test correlations between Dvorak-10 criteria and typing speed.

This script analyzes:
1. Sentence typing times (criteria 1-2: two-hand criteria)
2. Bigram interkey intervals (criteria 3-10: same-hand criteria)

Usage:
    python test_dvorak10_correlations.py
"""

import csv
import sys
import os
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.stats.multitest import multipletests

# Import the Dvorak10Scorer from the provided script
try:
    from score_dvorak10 import Dvorak10Scorer
except ImportError:
    print("Error: Could not import score_layout_dvorak10.py")
    print("Make sure the file is in the same directory as this script.")
    sys.exit(1)

# Standard QWERTY layout mapping for testing
QWERTY_ITEMS = "abcdefghijklmnopqrstuvwxyz;,./"
QWERTY_POSITIONS = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./"

# Global file handle for text output
output_file = None

def print_and_log(*args, **kwargs):
    """Print to console and write to log file."""
    print(*args, **kwargs)
    if output_file:
        print(*args, **kwargs, file=output_file)
        output_file.flush()

def create_qwerty_mapping():
    """Create standard QWERTY layout mapping."""
    return dict(zip(QWERTY_ITEMS.lower(), QWERTY_POSITIONS.upper()))

def read_bigram_times(filename, min_threshold=50, max_threshold=2000, use_percentile_filter=False):
    """Read bigram times from CSV file with optional filtering.
    
    Args:
        filename: CSV file path
        min_threshold: Minimum interkey interval (ms) - filters out impossibly fast keystrokes
        max_threshold: Maximum interkey interval (ms) - filters out long pauses
        use_percentile_filter: If True, use 5th-95th percentile instead of absolute thresholds
    """
    bigrams = []
    times = []
    
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                bigram = row['bigram'].lower().strip()
                time = float(row['interkey_interval'])
                
                # Only include bigrams with characters in our layout
                if len(bigram) == 2 and all(c in QWERTY_ITEMS.lower() for c in bigram):
                    bigrams.append(bigram)
                    times.append(time)
    
    except FileNotFoundError:
        print_and_log(f"Error: {filename} not found")
        return [], []
    except Exception as e:
        print_and_log(f"Error reading {filename}: {e}")
        return [], []
    
    if not times:
        return bigrams, times
    
    # Apply filtering
    original_count = len(times)
    
    if use_percentile_filter:
        # Use percentile-based filtering (more adaptive)
        p5 = np.percentile(times, 5)
        p95 = np.percentile(times, 95)
        filtered_indices = [i for i, t in enumerate(times) if p5 <= t <= p95]
        min_used, max_used = p5, p95
        filter_method = f"5th-95th percentile ({p5:.1f}-{p95:.1f}ms)"
    else:
        # Use absolute thresholds
        filtered_indices = [i for i, t in enumerate(times) if min_threshold <= t <= max_threshold]
        min_used, max_used = min_threshold, max_threshold
        filter_method = f"absolute thresholds ({min_threshold}-{max_threshold}ms)"
    
    # Apply filtering
    filtered_bigrams = [bigrams[i] for i in filtered_indices]
    filtered_times = [times[i] for i in filtered_indices]
    
    removed_count = original_count - len(filtered_times)
    if removed_count > 0:
        print_and_log(f"Filtered {removed_count}/{original_count} bigrams using {filter_method}")
        print_and_log(f"  Kept {len(filtered_times)} bigrams ({len(filtered_times)/original_count*100:.1f}%)")
        print_and_log(f"  Time range: {min(filtered_times):.1f} - {max(filtered_times):.1f}ms")
    
    return filtered_bigrams, filtered_times

def read_word_times(filename, max_threshold=None, use_percentile_filter=False):
    """Read word times from CSV file with optional filtering.
    
    Args:
        filename: CSV file path  
        max_threshold: Maximum word time (ms) - filters out very long words
        use_percentile_filter: If True, use 5th-95th percentile instead of absolute threshold
    """
    words = []
    times = []
    
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                word = row['word'].strip()
                time = float(row['time'])
                
                words.append(word)
                times.append(time)
    
    except FileNotFoundError:
        print_and_log(f"Error: {filename} not found")
        return [], []
    except Exception as e:
        print_and_log(f"Error reading {filename}: {e}")
        return [], []
    
    if not times or max_threshold is None:
        return words, times
    
    # Apply filtering 
    original_count = len(times)
    
    if use_percentile_filter:
        # Use percentile-based filtering
        p95 = np.percentile(times, 95)
        filtered_indices = [i for i, t in enumerate(times) if t <= p95]
        max_used = p95
        filter_method = f"95th percentile ({p95:.1f}ms)"
    else:
        # Use absolute threshold
        filtered_indices = [i for i, t in enumerate(times) if t <= max_threshold]
        max_used = max_threshold
        filter_method = f"absolute threshold ({max_threshold}ms)"
    
    # Apply filtering
    filtered_words = [words[i] for i in filtered_indices]
    filtered_times = [times[i] for i in filtered_indices]
    
    removed_count = original_count - len(filtered_times)
    if removed_count > 0:
        print_and_log(f"Filtered {removed_count}/{original_count} words using {filter_method}")
        print_and_log(f"  Kept {len(filtered_times)} words ({len(filtered_times)/original_count*100:.1f}%)")
        print_and_log(f"  Time range: {min(filtered_times):.1f} - {max(filtered_times):.1f}ms")
    
    return filtered_words, filtered_times
    """Read sentence times from CSV file with optional filtering.
    
    Args:
        filename: CSV file path  
        max_threshold: Maximum sentence time (ms) - filters out very long sentences
        use_percentile_filter: If True, use 5th-95th percentile instead of absolute threshold
    """
    sentences = []
    times = []
    
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                sentence = row['sentence'].strip()
                time = float(row['time'])
                
                sentences.append(sentence)
                times.append(time)
    
    except FileNotFoundError:
        print_and_log(f"Error: {filename} not found")
        return [], []
    except Exception as e:
        print_and_log(f"Error reading {filename}: {e}")
        return [], []
    
    if not times or max_threshold is None:
        return sentences, times
    
    # Apply filtering 
    original_count = len(times)
    
    if use_percentile_filter:
        # Use percentile-based filtering
        p95 = np.percentile(times, 95)
        filtered_indices = [i for i, t in enumerate(times) if t <= p95]
        max_used = p95
        filter_method = f"95th percentile ({p95:.1f}ms)"
    else:
        # Use absolute threshold
        filtered_indices = [i for i, t in enumerate(times) if t <= max_threshold]
        max_used = max_threshold
        filter_method = f"absolute threshold ({max_threshold}ms)"
    
    # Apply filtering
    filtered_sentences = [sentences[i] for i in filtered_indices]
    filtered_times = [times[i] for i in filtered_indices]
    
    removed_count = original_count - len(filtered_times)
    if removed_count > 0:
        print_and_log(f"Filtered {removed_count}/{original_count} sentences using {filter_method}")
        print_and_log(f"  Kept {len(filtered_times)} sentences ({len(filtered_times)/original_count*100:.1f}%)")
        print_and_log(f"  Time range: {min(filtered_times):.1f} - {max(filtered_times):.1f}ms")
    
    return filtered_sentences, filtered_times

def plot_combined_boxplots(criterion_data, output_dir="plots"):
    """Create combined box-and-whisker plots for all same-hand criteria."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Better criterion names for display
    criterion_labels = {
        '3_different_fingers': 'Two Fingers',
        '4_non_adjacent_fingers': 'Skip Fingers', 
        '5_home_block': 'Home Block',
        '6_dont_skip_home': "Don't Skip Home",
        '7_same_row': 'Same Row',
        '8_include_home': 'Include Home',
        '9_roll_inward': 'Roll Inward',
        '10_strong_fingers': 'Strong Fingers'
    }
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    all_box_data = []
    all_labels = []
    all_positions = []
    position = 0
    
    for criterion, data in criterion_data.items():
        if criterion not in criterion_labels:
            continue
            
        scores = np.array(data['scores'])
        times = np.array(data['times'])
        
        # Separate times by score (0 or 1)
        times_0 = times[scores == 0]
        times_1 = times[scores == 1]
        
        if len(times_0) > 0:
            all_box_data.append(times_0)
            all_labels.append(f"{criterion_labels[criterion]} 0")
            all_positions.append(position)
            position += 1
        
        if len(times_1) > 0:
            all_box_data.append(times_1)
            all_labels.append(f"{criterion_labels[criterion]} 1")
            all_positions.append(position)
            position += 1
        
        # Add space between criteria
        position += 0.5
    
    # Create box plot
    box_plot = ax.boxplot(all_box_data, positions=all_positions, patch_artist=True)
    
    # Color boxes alternately (score 0 = light blue, score 1 = orange)
    colors = ['lightblue', 'orange'] * (len(all_box_data) // 2 + 1)
    for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.set_ylabel('Interkey Interval (ms)')
    ax.set_title('Distribution of Interkey Intervals by Criterion Score\n(Blue = Score 0, Orange = Score 1)')
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits - choose one of these options:
    
    # Option 1: Fixed limits (good if you know your typical range)
    # ax.set_ylim(0, 1000)  # 0 to 1000ms
    
    # Option 2: Automatic based on data percentiles (removes extreme outliers from view)
    all_times = []
    for data in criterion_data.values():
        all_times.extend(data['times'])
    if all_times:
        p5 = np.percentile(all_times, 5)
        p95 = np.percentile(all_times, 95)
        ax.set_ylim(0, p95 * 1.1)  # Show from 0 to 110% of 95th percentile
    
    # Option 3: More conservative percentile view
    # if all_times:
    #     p1 = np.percentile(all_times, 1) 
    #     p99 = np.percentile(all_times, 99)
    #     ax.set_ylim(max(0, p1 * 0.9), p99 * 1.1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/combined_boxplots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return "combined_boxplots.png"

def plot_combined_frequencies(criterion_data, output_dir="plots"):
    """Create combined frequency bar plot for all same-hand criteria."""
    Path(output_dir).mkdir(exist_ok=True)
    
    criterion_labels = {
        '3_different_fingers': 'Two Fingers',
        '4_non_adjacent_fingers': 'Skip Fingers', 
        '5_home_block': 'Home Block',
        '6_dont_skip_home': "Don't Skip Home",
        '7_same_row': 'Same Row',
        '8_include_home': 'Include Home',
        '9_roll_inward': 'Roll Inward',
        '10_strong_fingers': 'Strong Fingers'
    }
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    criteria_names = []
    counts_0 = []
    counts_1 = []
    
    for criterion, data in criterion_data.items():
        if criterion not in criterion_labels:
            continue
            
        scores = np.array(data['scores'])
        
        count_0 = np.sum(scores == 0)
        count_1 = np.sum(scores == 1)
        
        criteria_names.append(criterion_labels[criterion])
        counts_0.append(count_0)
        counts_1.append(count_1)
    
    x = np.arange(len(criteria_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, counts_0, width, label='Score 0', color='lightblue', alpha=0.7)
    bars2 = ax.bar(x + width/2, counts_1, width, label='Score 1', color='orange', alpha=0.7)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{int(height)}', ha='center', va='bottom')
    
    ax.set_xlabel('Criterion')
    ax.set_ylabel('Frequency')
    ax.set_title('Frequency Distribution of Criterion Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(criteria_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/combined_frequencies.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return "combined_frequencies.png"

def create_pairwise_correlation_matrix(all_criterion_data, output_dir="plots"):
    """Create pairwise correlation matrix between all criteria."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Collect all criteria and their scores
    criteria_info = {}
    
    # Add same-hand criteria
    same_hand_labels = {
        '3_different_fingers': 'Two Fingers',
        '4_non_adjacent_fingers': 'Skip Fingers', 
        '5_home_block': 'Home Block',
        '6_dont_skip_home': "Don't Skip Home",
        '7_same_row': 'Same Row',
        '8_include_home': 'Include Home',
        '9_roll_inward': 'Roll Inward',
        '10_strong_fingers': 'Strong Fingers'
    }
    
    for criterion, label in same_hand_labels.items():
        if criterion in all_criterion_data:
            criteria_info[label] = all_criterion_data[criterion]['scores']
    
    # Add two-hand criteria if available
    two_hand_labels = {
        '1_hand_balance': 'Hand Balance',
        '2_hand_alternation': 'Hand Alternation'
    }
    
    for criterion, label in two_hand_labels.items():
        if criterion in all_criterion_data:
            criteria_info[label] = all_criterion_data[criterion]['scores']
    
    if len(criteria_info) < 2:
        print_and_log("Not enough criteria for pairwise correlation matrix")
        return None
    
    # Create correlation matrix
    criteria_names = list(criteria_info.keys())
    n_criteria = len(criteria_names)
    correlation_matrix = np.zeros((n_criteria, n_criteria))
    
    print_and_log(f"\nPairwise Correlations Between All Criteria:")
    print_and_log("=" * 80)
    
    for i, crit1 in enumerate(criteria_names):
        for j, crit2 in enumerate(criteria_names):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                scores1 = criteria_info[crit1]
                scores2 = criteria_info[crit2]
                
                # Ensure same length
                min_len = min(len(scores1), len(scores2))
                if min_len > 0:
                    r, p = spearmanr(scores1[:min_len], scores2[:min_len])
                    correlation_matrix[i, j] = r
                    
                    # Print significant correlations
                    if i < j and abs(r) > 0.3:  # Only print upper triangle and moderate+ correlations
                        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                        print_and_log(f"{crit1} ↔ {crit2}: r = {r:.3f}{sig} (p = {p:.3f})")
                else:
                    correlation_matrix[i, j] = 0.0
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                vmin=-1, vmax=1,
                xticklabels=criteria_names,
                yticklabels=criteria_names,
                fmt='.3f',
                ax=ax)
    
    plt.title('Pairwise Correlations Between All Criteria')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pairwise_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return correlation_matrix, criteria_names

def apply_multiple_comparisons_correction(results, alpha=0.05):
    """Apply multiple comparisons correction to p-values."""
    if not results:
        return results
    
    print_and_log(f"\nMultiple Comparisons Correction")
    print_and_log("=" * 50)
    
    # Extract p-values
    p_values = [data['spearman_p'] for data in results.values()]
    criteria_names = [data['name'] for data in results.values()]
    
    # Apply FDR correction (less conservative than Bonferroni)
    rejected, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    
    print_and_log(f"Original α = {alpha}")
    print_and_log(f"Number of tests: {len(p_values)}")
    print_and_log(f"FDR-corrected significant results:")
    
    # Update results with corrected p-values
    corrected_results = {}
    any_significant = False
    
    for i, (criterion, data) in enumerate(results.items()):
        corrected_data = data.copy()
        corrected_data['spearman_p_corrected'] = p_adjusted[i]
        corrected_data['significant_corrected'] = rejected[i]
        corrected_results[criterion] = corrected_data
        
        if rejected[i]:
            any_significant = True
            r = data['spearman_r']
            direction = "Better criterion → Faster typing" if r < 0 else "Better criterion → Slower typing"
            print_and_log(f"  ✓ {data['name']}: r={r:.3f}, p_adj={p_adjusted[i]:.3f} ({direction})")
    
    if not any_significant:
        print_and_log("  None remain significant after correction")
    
    return corrected_results

def interpret_effect_sizes(results):
    """Interpret practical significance of correlations."""
    print_and_log("\nPractical Significance Interpretation")
    print_and_log("=" * 50)
    print_and_log("Cohen's guidelines for correlation effect sizes:")
    print_and_log("  Small effect:    |r| = 0.10-0.30")
    print_and_log("  Medium effect:   |r| = 0.30-0.50") 
    print_and_log("  Large effect:    |r| = 0.50+")
    print_and_log("\nWith your large sample size, focus on effect size, not p-values!")
    print_and_log("-" * 50)
    
    if not results:
        return
    
    # Categorize effects
    small_effects = []
    medium_effects = []
    large_effects = []
    
    for criterion, data in results.items():
        r = abs(data['spearman_r'])
        name = data['name']
        direction = "↓" if data['spearman_r'] < 0 else "↑"
        
        if r >= 0.50:
            large_effects.append((name, data['spearman_r'], direction))
        elif r >= 0.30:
            medium_effects.append((name, data['spearman_r'], direction))
        elif r >= 0.10:
            small_effects.append((name, data['spearman_r'], direction))
    
    # Print categorized results
    if large_effects:
        print_and_log("LARGE EFFECTS (|r| ≥ 0.50) - Practically Important:")
        for name, r, direction in sorted(large_effects, key=lambda x: abs(x[1]), reverse=True):
            print_and_log(f"  {direction} {name}: r = {r:.3f}")
    
    if medium_effects:
        print_and_log("\nMEDIUM EFFECTS (0.30 ≤ |r| < 0.50) - Moderately Important:")
        for name, r, direction in sorted(medium_effects, key=lambda x: abs(x[1]), reverse=True):
            print_and_log(f"  {direction} {name}: r = {r:.3f}")
    
    if small_effects:
        print_and_log("\nSMALL EFFECTS (0.10 ≤ |r| < 0.30) - Minor Importance:")
        for name, r, direction in sorted(small_effects, key=lambda x: abs(x[1]), reverse=True):
            print_and_log(f"  {direction} {name}: r = {r:.3f}")

def analyze_bigram_correlations(bigrams, times):
    """Analyze correlations between same-hand criteria (3-10) and bigram times."""
    layout_mapping = create_qwerty_mapping()
    
    # Criteria 3-10 (same-hand criteria)
    criteria_names = {
        '3_different_fingers': 'Two Fingers',
        '4_non_adjacent_fingers': 'Skip Fingers', 
        '5_home_block': 'Home Block',
        '6_dont_skip_home': "Don't Skip Home",
        '7_same_row': 'Same Row',
        '8_include_home': 'Include Home',
        '9_roll_inward': 'Roll Inward',
        '10_strong_fingers': 'Strong Fingers'
    }
    
    results = {}
    
    print_and_log("Analyzing bigram correlations...")
    print_and_log(f"Total bigrams: {len(bigrams)}")
    
    # Collect scores for each criterion
    criterion_scores = {criterion: [] for criterion in criteria_names.keys()}
    valid_times = []
    valid_bigrams = []
    
    for bigram, time in zip(bigrams, times):
        try:
            scorer = Dvorak10Scorer(layout_mapping, bigram)
            scores = scorer.calculate_all_scores()
            
            # Only include if we have same-hand digraphs to analyze
            has_same_hand = any(
                scores[criterion]['details'].get('total_same_hand', 0) > 0
                for criterion in criteria_names.keys()
            )
            
            if has_same_hand:
                valid_bigrams.append(bigram)
                valid_times.append(time)
                
                for criterion in criteria_names.keys():
                    score = scores[criterion]['score']
                    criterion_scores[criterion].append(score)
        
        except Exception as e:
            print_and_log(f"Error processing bigram '{bigram}': {e}")
            continue
    
    print_and_log(f"Valid same-hand bigrams: {len(valid_bigrams)}")
    
    # Calculate correlations
    for criterion, scores in criterion_scores.items():
        if len(scores) >= 3:  # Need at least 3 points for correlation
            try:
                # Pearson correlation
                pearson_r, pearson_p = pearsonr(scores, valid_times)
                
                # Spearman correlation (rank-based, more robust)
                spearman_r, spearman_p = spearmanr(scores, valid_times)
                
                results[criterion] = {
                    'name': criteria_names[criterion],
                    'n_samples': len(scores),
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'scores': scores.copy(),  # Store for plotting
                    'times': valid_times.copy()
                }
                
            except Exception as e:
                print_and_log(f"Error calculating correlation for {criterion}: {e}")
                continue
    
    # Apply multiple comparisons correction
    results = apply_multiple_comparisons_correction(results)
    
    # Interpret effect sizes
    interpret_effect_sizes(results)
    
    # Create combined plots
    plot_combined_boxplots(results)
    plot_combined_frequencies(results)
    
    return results

def analyze_word_correlations(words, times):
    """Analyze correlations between two-hand criteria (1-2) and word times."""
    layout_mapping = create_qwerty_mapping()
    
    # Criteria 1-2 (two-hand criteria)
    criteria_names = {
        '1_hand_balance': 'Hand Balance',
        '2_hand_alternation': 'Hand Alternation'
    }
    
    results = {}
    
    print_and_log("Analyzing word correlations...")
    print_and_log(f"Total words: {len(words)}")
    
    # Collect scores for each criterion
    criterion_scores = {criterion: [] for criterion in criteria_names.keys()}
    valid_times = []
    valid_words = []
    
    for word, time in zip(words, times):
        try:
            # Filter word to only include characters in our layout
            filtered_word = ''.join(c for c in word.lower() if c in layout_mapping)
            
            if len(filtered_word.strip()) < 2:
                continue
                
            scorer = Dvorak10Scorer(layout_mapping, filtered_word)
            scores = scorer.calculate_all_scores()
            
            valid_words.append(word[:20] + "..." if len(word) > 20 else word)
            valid_times.append(time)
            
            for criterion in criteria_names.keys():
                score = scores[criterion]['score']
                criterion_scores[criterion].append(score)
        
        except Exception as e:
            print_and_log(f"Error processing word '{word}': {e}")
            continue
    
    print_and_log(f"Valid words: {len(valid_words)}")
    
    # Calculate correlations
    for criterion, scores in criterion_scores.items():
        if len(scores) >= 3:  # Need at least 3 points for correlation
            try:
                # Pearson correlation
                pearson_r, pearson_p = pearsonr(scores, valid_times)
                
                # Spearman correlation (rank-based, more robust)
                spearman_r, spearman_p = spearmanr(scores, valid_times)
                
                results[criterion] = {
                    'name': criteria_names[criterion],
                    'n_samples': len(scores),
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'scores': scores.copy(),  # Store for plotting
                    'times': valid_times.copy()
                }
                
            except Exception as e:
                print_and_log(f"Error calculating correlation for {criterion}: {e}")
                continue
    
    # Apply multiple comparisons correction
    results = apply_multiple_comparisons_correction(results)
    
    # Interpret effect sizes
    interpret_effect_sizes(results)
    
    return results

def print_correlation_results(results, title, time_description):
    """Print correlation results in a formatted table."""
    print_and_log(f"\n{title}")
    print_and_log("=" * 80)
    print_and_log(f"Correlation with {time_description}")
    print_and_log("Note: Negative correlation = higher score → faster typing (good)")
    print_and_log("      Positive correlation = higher score → slower typing (bad)")
    print_and_log("-" * 80)
    
    if not results:
        print_and_log("No valid correlations found.")
        return
    
    # Header
    print_and_log(f"{'Criterion':<15} {'N':<4} {'Pearson r':<10} {'p-val':<8} {'Spearman r':<11} {'p-val':<8} {'Mean±SD':<12}")
    print_and_log("-" * 80)
    
    # Sort by Spearman correlation (more robust)
    sorted_results = sorted(results.items(), key=lambda x: abs(x[1]['spearman_r']), reverse=True)
    
    for criterion, data in sorted_results:
        name = data['name']
        n = data['n_samples']
        pr = data['pearson_r']
        pp = data['pearson_p']
        sr = data['spearman_r']
        sp = data['spearman_p']
        mean = data['mean_score']
        std = data['std_score']
        
        # Significance indicators
        p_sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else ""
        s_sig = "***" if sp < 0.001 else "**" if sp < 0.01 else "*" if sp < 0.05 else ""
        
        print_and_log(f"{name:<15} {n:<4} {pr:>7.3f}{p_sig:<3} {pp:<8.3f} {sr:>7.3f}{s_sig:<4} {sp:<8.3f} {mean:.2f}±{std:.2f}")

def print_summary(bigram_results, word_results):
    """Print summary of findings."""
    print_and_log("\n" + "=" * 80)
    print_and_log("SUMMARY OF FINDINGS")
    print_and_log("=" * 80)
    
    print_and_log("\nSignificant correlations (p < 0.05):")
    
    significant_found = False
    
    # Check bigram results
    for criterion, data in bigram_results.items():
        if data['spearman_p'] < 0.05:
            significant_found = True
            direction = "GOOD" if data['spearman_r'] < 0 else "BAD"
            print_and_log(f"  {data['name']} (bigrams): r={data['spearman_r']:.3f}, p={data['spearman_p']:.3f} [{direction}]")
    
    # Check word results  
    for criterion, data in word_results.items():
        if data['spearman_p'] < 0.05:
            significant_found = True
            direction = "GOOD" if data['spearman_r'] < 0 else "BAD"
            print_and_log(f"  {data['name']} (words): r={data['spearman_r']:.3f}, p={data['spearman_p']:.3f} [{direction}]")
    
    if not significant_found:
        print_and_log("  No statistically significant correlations found.")
    
    print_and_log("\nInterpretation:")
    print_and_log("- GOOD: Higher criterion score → faster typing (validates Dvorak's principle)")
    print_and_log("- BAD: Higher criterion score → slower typing (contradicts Dvorak's principle)")
    print_and_log("- Significance levels: * p<0.05, ** p<0.01, *** p<0.001")

def main():
    """Main analysis function."""
    global output_file
    
    # Open output file
    output_file = open('dvorak_analysis_results.txt', 'w', encoding='utf-8')
    
    try:
        print_and_log("Dvorak-10 Criteria Correlation Analysis")
        print_and_log("=" * 50)
        
        # Read data files
        print_and_log("Reading data files...")
        bigram_times_file = '../process_keystroke_data/output/bigram_times.csv'
        word_times_file = '../process_keystroke_data/output/word_times.csv'  # Changed from sentence_times.csv
        
        # FILTERING PARAMETERS - Adjust these as needed
        # For bigram interkey intervals:
        MIN_INTERVAL = 50     # ms - filters out impossibly fast keystrokes  
        MAX_INTERVAL = 2000   # ms - filters out long pauses/hesitations
        USE_PERCENTILE_BIGRAMS = False  # Set to True to use 5th-95th percentile instead
        
        # For word times:
        MAX_WORD_TIME = None    # ms - set to filter very long words (None = no filtering)
        USE_PERCENTILE_WORDS = False  # Set to True to use 95th percentile instead
        
        # Actually read the data files
        bigrams, bigram_times = read_bigram_times(
            bigram_times_file, 
            min_threshold=MIN_INTERVAL,
            max_threshold=MAX_INTERVAL, 
            use_percentile_filter=USE_PERCENTILE_BIGRAMS
        )
        
        words, word_times = read_word_times(
            word_times_file,
            max_threshold=MAX_WORD_TIME,
            use_percentile_filter=USE_PERCENTILE_WORDS
        )
        
        if not bigrams and not words:
            print_and_log("Error: No valid data found in CSV files.")
            sys.exit(1)
        
        results = {}
        
        # Create output directory for plots
        Path("plots").mkdir(exist_ok=True)
        print_and_log("Creating plots in 'plots/' directory...")
        
        # Analyze bigram correlations (criteria 3-10)
        if bigrams:
            bigram_results = analyze_bigram_correlations(bigrams, bigram_times)
            results['bigrams'] = bigram_results
            print_correlation_results(bigram_results, "BIGRAM ANALYSIS (Same-Hand Criteria)", 
                                    "interkey intervals (ms)")
        
        # Analyze word correlations (criteria 1-2)  
        if words:
            word_results = analyze_word_correlations(words, word_times)
            results['words'] = word_results
            print_correlation_results(word_results, "WORD ANALYSIS (Two-Hand Criteria)",
                                    "word typing times (ms)")
        
        # Create pairwise correlation matrix
        all_criterion_data = {}
        if 'bigrams' in results:
            all_criterion_data.update(results['bigrams'])
        if 'words' in results:
            all_criterion_data.update(results['words'])
        
        if all_criterion_data:
            create_pairwise_correlation_matrix(all_criterion_data)
        
        # Print summary
        bigram_res = results.get('bigrams', {})
        word_res = results.get('words', {})
        print_summary(bigram_res, word_res)
        
        print_and_log(f"\n" + "=" * 60)
        print_and_log("ANALYSIS COMPLETE")
        print_and_log("=" * 60)
        print_and_log(f"Key outputs saved:")
        print_and_log(f"- Text output: dvorak_analysis_results.txt")
        print_and_log(f"- Combined box plots: plots/combined_boxplots.png")
        print_and_log(f"- Combined frequencies: plots/combined_frequencies.png")
        print_and_log(f"- Pairwise correlations: plots/pairwise_correlations.png")
        
    finally:
        if output_file:
            output_file.close()

if __name__ == "__main__":
    main()
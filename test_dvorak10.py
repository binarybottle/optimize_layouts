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
# Assuming score_layout_dvorak10.py is in the same directory
try:
    from score_dvorak10 import Dvorak10Scorer
except ImportError:
    print("Error: Could not import score_layout_dvorak10.py")
    print("Make sure the file is in the same directory as this script.")
    sys.exit(1)

bigram_times_file = '../process_keystroke_data/output/bigram_times.csv'
sentence_times_file = '../process_keystroke_data/output/sentence_times.csv'

# Standard QWERTY layout mapping for testing
QWERTY_ITEMS = "abcdefghijklmnopqrstuvwxyz;,./"
QWERTY_POSITIONS = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./"

def create_qwerty_mapping():
    """Create standard QWERTY layout mapping."""
    return dict(zip(QWERTY_ITEMS.lower(), QWERTY_POSITIONS.upper()))

def read_bigram_times(filename):
    """Read bigram times from CSV file."""
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
        print(f"Error: {filename} not found")
        return [], []
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return [], []
    
    return bigrams, times

def read_sentence_times(filename):
    """Read sentence times from CSV file."""
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
        print(f"Error: {filename} not found")
        return [], []
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return [], []
    
    return sentences, times

def test_multicollinearity(scores_dict, criterion_names):
    """Test multicollinearity between Dvorak criteria."""
    print("\nMulticollinearity Analysis")
    print("=" * 50)
    
    # Create correlation matrix between criteria
    criteria_list = list(scores_dict.keys())
    n_criteria = len(criteria_list)
    
    if n_criteria < 2:
        print("Need at least 2 criteria for multicollinearity testing.")
        return None
    
    # Build correlation matrix
    correlation_matrix = np.zeros((n_criteria, n_criteria))
    criterion_labels = []
    
    for i, crit1 in enumerate(criteria_list):
        criterion_labels.append(criterion_names.get(crit1, crit1))
        for j, crit2 in enumerate(criteria_list):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                # Get scores for both criteria
                scores1 = scores_dict[crit1]
                scores2 = scores_dict[crit2]
                
                # Ensure same length (should be, but safety check)
                min_len = min(len(scores1), len(scores2))
                if min_len > 0:
                    r, _ = pearsonr(scores1[:min_len], scores2[:min_len])
                    correlation_matrix[i, j] = r
                else:
                    correlation_matrix[i, j] = 0.0
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                vmin=-1, vmax=1,
                xticklabels=criterion_labels,
                yticklabels=criterion_labels,
                fmt='.3f')
    
    plt.title('Multicollinearity Between Dvorak Criteria\n(High correlations indicate redundancy)')
    plt.tight_layout()
    plt.savefig('plots/multicollinearity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Report problematic correlations
    print("High correlations between criteria (|r| > 0.7):")
    high_corr_found = False
    
    for i in range(n_criteria):
        for j in range(i+1, n_criteria):
            r = correlation_matrix[i, j]
            if abs(r) > 0.7:
                high_corr_found = True
                print(f"  {criterion_labels[i]} ↔ {criterion_labels[j]}: r = {r:.3f}")
    
    if not high_corr_found:
        print("  None found (good!)")
    
    print("\nModerate correlations (0.5 < |r| < 0.7):")
    mod_corr_found = False
    
    for i in range(n_criteria):
        for j in range(i+1, n_criteria):
            r = correlation_matrix[i, j]
            if 0.5 < abs(r) <= 0.7:
                mod_corr_found = True
                print(f"  {criterion_labels[i]} ↔ {criterion_labels[j]}: r = {r:.3f}")
    
    if not mod_corr_found:
        print("  None found")
    
    return correlation_matrix

def interpret_effect_sizes(results):
    """Interpret practical significance of correlations."""
    print("\nPractical Significance Interpretation")
    print("=" * 50)
    print("Cohen's guidelines for correlation effect sizes:")
    print("  Small effect:    |r| = 0.10-0.30")
    print("  Medium effect:   |r| = 0.30-0.50") 
    print("  Large effect:    |r| = 0.50+")
    print("\nWith your large sample size, focus on effect size, not p-values!")
    print("-" * 50)
    
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
        print("LARGE EFFECTS (|r| ≥ 0.50) - Practically Important:")
        for name, r, direction in sorted(large_effects, key=lambda x: abs(x[1]), reverse=True):
            print(f"  {direction} {name}: r = {r:.3f}")
    
    if medium_effects:
        print("\nMEDIUM EFFECTS (0.30 ≤ |r| < 0.50) - Moderately Important:")
        for name, r, direction in sorted(medium_effects, key=lambda x: abs(x[1]), reverse=True):
            print(f"  {direction} {name}: r = {r:.3f}")
    
    if small_effects:
        print("\nSMALL EFFECTS (0.10 ≤ |r| < 0.30) - Minor Importance:")
        for name, r, direction in sorted(small_effects, key=lambda x: abs(x[1]), reverse=True):
            print(f"  {direction} {name}: r = {r:.3f}")
    
    # Calculate practical impact
    print(f"\nPractical Impact Estimation:")
    print(f"(Based on correlation with large sample size)")
    
    for criterion, data in results.items():
        if abs(data['spearman_r']) >= 0.30:  # Only show medium+ effects
            r = data['spearman_r']
            name = data['name']
            
            # Estimate practical impact (very rough approximation)
            # For a 1 SD change in criterion score (typically ~0.2-0.3 for normalized scores)
            # How much change in typing time?
            sd_criterion = data.get('std_score', 0.2)  # Fallback estimate
            
            print(f"  {name}: 1 SD improvement ({sd_criterion:.2f}) → {r*sd_criterion:.1%} change in typing time")

def apply_multiple_comparisons_correction(results, alpha=0.05):
    """Apply multiple comparisons correction to p-values."""
    if not results:
        return results
    
    print(f"\nMultiple Comparisons Correction")
    print("=" * 50)
    
    # Extract p-values
    p_values = [data['spearman_p'] for data in results.values()]
    criteria_names = [data['name'] for data in results.values()]
    
    # Apply FDR correction (less conservative than Bonferroni)
    rejected, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    
    print(f"Original α = {alpha}")
    print(f"Number of tests: {len(p_values)}")
    print(f"FDR-corrected significant results:")
    
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
            print(f"  ✓ {data['name']}: r={r:.3f}, p_adj={p_adjusted[i]:.3f} ({direction})")
    
    if not any_significant:
        print("  None remain significant after correction")
    
    return corrected_results

def plot_correlation_scatter(scores, times, criterion_name, time_description, output_dir="plots"):
    """Create scatter plot for a single criterion vs times."""
    Path(output_dir).mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(scores, times, alpha=0.6, s=50)
    
    # Add trend line
    z = np.polyfit(scores, times, 1)
    p = np.poly1d(z)
    plt.plot(scores, p(scores), "r--", alpha=0.8, linewidth=2)
    
    # Calculate correlation
    r, p_val = spearmanr(scores, times)
    
    plt.xlabel(f'{criterion_name} Score')
    plt.ylabel(f'{time_description}')
    plt.title(f'{criterion_name} vs {time_description}\nSpearman r = {r:.3f}, p = {p_val:.3f}')
    
    # Add interpretation text
    interpretation = "Better criterion → Faster typing" if r < 0 else "Better criterion → Slower typing"
    color = "green" if r < 0 else "red"
    plt.text(0.05, 0.95, interpretation, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    filename = f"{criterion_name.lower().replace(' ', '_')}_scatter.png"
    plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def plot_correlation_heatmap(results_dict, title, output_dir="plots"):
    """Create heatmap of correlation coefficients."""
    Path(output_dir).mkdir(exist_ok=True)
    
    if not results_dict:
        return None
    
    criteria = list(results_dict.keys())
    names = [results_dict[c]['name'] for c in criteria]
    correlations = [results_dict[c]['spearman_r'] for c in criteria]
    p_values = [results_dict[c]['spearman_p'] for c in criteria]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap data
    heatmap_data = np.array(correlations).reshape(-1, 1)
    
    # Custom colormap: green for negative (good), red for positive (bad)
    cmap = plt.cm.RdYlGn_r
    
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xticks([0])
    ax.set_xticklabels(['Correlation'])
    
    # Add correlation values and significance
    for i, (r, p) in enumerate(zip(correlations, p_values)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        text_color = 'white' if abs(r) > 0.5 else 'black'
        ax.text(0, i, f'{r:.3f}{sig}', ha='center', va='center', 
                color=text_color, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Spearman Correlation')
    plt.title(f'{title}\n(Green = Good, Red = Bad)')
    plt.tight_layout()
    
    filename = f"{title.lower().replace(' ', '_')}_heatmap.png"
    plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def plot_score_distributions(all_scores, criterion_name, output_dir="plots"):
    """Plot distribution of criterion scores."""
    Path(output_dir).mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(all_scores, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(all_scores), color='red', linestyle='--', linewidth=2, 
                label=f'Mean = {np.mean(all_scores):.3f}')
    plt.axvline(np.median(all_scores), color='green', linestyle='--', linewidth=2,
                label=f'Median = {np.median(all_scores):.3f}')
    
    plt.xlabel(f'{criterion_name} Score')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {criterion_name} Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"{criterion_name.lower().replace(' ', '_')}_distribution.png"
    plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def plot_summary_comparison(bigram_results, sentence_results, output_dir="plots"):
    """Create summary comparison plot of all correlations."""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Bigram results
    if bigram_results:
        criteria = list(bigram_results.keys())
        names = [bigram_results[c]['name'] for c in criteria]
        correlations = [bigram_results[c]['spearman_r'] for c in criteria]
        p_values = [bigram_results[c]['spearman_p'] for c in criteria]
        
        # Color bars based on significance and direction
        colors = []
        for r, p in zip(correlations, p_values):
            if p < 0.05:
                colors.append('green' if r < 0 else 'red')
            else:
                colors.append('gray')
        
        bars1 = ax1.barh(range(len(names)), correlations, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names)
        ax1.set_xlabel('Spearman Correlation')
        ax1.set_title('Same-Hand Criteria (Bigrams)\nGreen=Good, Red=Bad, Gray=Not Significant')
        ax1.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Add correlation values on bars
        for i, (bar, r, p) in enumerate(zip(bars1, correlations, p_values)):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax1.text(r + (0.01 if r >= 0 else -0.01), i, f'{r:.3f}{sig}', 
                    ha='left' if r >= 0 else 'right', va='center')
    
    # Sentence results
    if sentence_results:
        criteria = list(sentence_results.keys())
        names = [sentence_results[c]['name'] for c in criteria]
        correlations = [sentence_results[c]['spearman_r'] for c in criteria]
        p_values = [sentence_results[c]['spearman_p'] for c in criteria]
        
        colors = []
        for r, p in zip(correlations, p_values):
            if p < 0.05:
                colors.append('green' if r < 0 else 'red')
            else:
                colors.append('gray')
        
        bars2 = ax2.barh(range(len(names)), correlations, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names)
        ax2.set_xlabel('Spearman Correlation')
        ax2.set_title('Two-Hand Criteria (Sentences)\nGreen=Good, Red=Bad, Gray=Not Significant')
        ax2.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        for i, (bar, r, p) in enumerate(zip(bars2, correlations, p_values)):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax2.text(r + (0.01 if r >= 0 else -0.01), i, f'{r:.3f}{sig}', 
                    ha='left' if r >= 0 else 'right', va='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return "summary_correlations.png"

def analyze_bigram_correlations(bigrams, times):
    """Analyze correlations between same-hand criteria (3-10) and bigram times."""
    layout_mapping = create_qwerty_mapping()
    
    # Criteria 3-10 (same-hand criteria)
    criteria_names = {
        '3_different_fingers': 'Different Fingers',
        '4_non_adjacent_fingers': 'Non-Adjacent Fingers', 
        '5_home_block': 'Home Block',
        '6_dont_skip_home': 'Don\'t Skip Home',
        '7_same_row': 'Same Row',
        '8_include_home': 'Include Home',
        '9_roll_inward': 'Roll Inward',
        '10_strong_fingers': 'Strong Fingers'
    }
    
    results = {}
    
    print("Analyzing bigram correlations...")
    print(f"Total bigrams: {len(bigrams)}")
    
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
            print(f"Error processing bigram '{bigram}': {e}")
            continue
    
    print(f"Valid same-hand bigrams: {len(valid_bigrams)}")
    
    # Test multicollinearity between criteria
    print(f"\nTesting multicollinearity between same-hand criteria...")
    test_multicollinearity(criterion_scores, criteria_names)
    
    # Calculate correlations and create plots
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
                
                # Create individual scatter plot
                plot_correlation_scatter(scores, valid_times, criteria_names[criterion], 
                                       "Interkey Interval (ms)")
                
                # Create score distribution plot
                plot_score_distributions(scores, criteria_names[criterion])
                
            except Exception as e:
                print(f"Error calculating correlation for {criterion}: {e}")
                continue
    
    # Apply multiple comparisons correction
    results = apply_multiple_comparisons_correction(results)
    
    # Interpret effect sizes
    interpret_effect_sizes(results)
    
    # Create heatmap
    if results:
        plot_correlation_heatmap(results, "Same-Hand Criteria (Bigrams)")
    
    return results

def analyze_sentence_correlations(sentences, times):
    """Analyze correlations between two-hand criteria (1-2) and sentence times."""
    layout_mapping = create_qwerty_mapping()
    
    # Criteria 1-2 (two-hand criteria)
    criteria_names = {
        '1_hand_balance': 'Hand Balance',
        '2_hand_alternation': 'Hand Alternation'
    }
    
    results = {}
    
    print("Analyzing sentence correlations...")
    print(f"Total sentences: {len(sentences)}")
    
    # Collect scores for each criterion
    criterion_scores = {criterion: [] for criterion in criteria_names.keys()}
    valid_times = []
    valid_sentences = []
    
    for sentence, time in zip(sentences, times):
        try:
            # Filter sentence to only include characters in our layout
            filtered_sentence = ''.join(c for c in sentence.lower() if c in layout_mapping or c.isspace())
            
            if len(filtered_sentence.strip()) < 2:
                continue
                
            scorer = Dvorak10Scorer(layout_mapping, filtered_sentence)
            scores = scorer.calculate_all_scores()
            
            valid_sentences.append(sentence[:50] + "..." if len(sentence) > 50 else sentence)
            valid_times.append(time)
            
            for criterion in criteria_names.keys():
                score = scores[criterion]['score']
                criterion_scores[criterion].append(score)
        
        except Exception as e:
            print(f"Error processing sentence: {e}")
            continue
    
    print(f"Valid sentences: {len(valid_sentences)}")
    
    # Test multicollinearity between criteria (though only 2 criteria)
    if len(criteria_names) > 1:
        print(f"\nTesting correlation between two-hand criteria...")
        test_multicollinearity(criterion_scores, criteria_names)
    
    # Calculate correlations and create plots
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
                
                # Create individual scatter plot
                plot_correlation_scatter(scores, valid_times, criteria_names[criterion], 
                                       "Sentence Typing Time (ms)")
                
                # Create score distribution plot
                plot_score_distributions(scores, criteria_names[criterion])
                
            except Exception as e:
                print(f"Error calculating correlation for {criterion}: {e}")
                continue
    
    # Apply multiple comparisons correction
    results = apply_multiple_comparisons_correction(results)
    
    # Interpret effect sizes
    interpret_effect_sizes(results)
    
    # Create heatmap
    if results:
        plot_correlation_heatmap(results, "Two-Hand Criteria (Sentences)")
    
    return results

def print_correlation_results(results, title, time_description):
    """Print correlation results in a formatted table."""
    print(f"\n{title}")
    print("=" * 80)
    print(f"Correlation with {time_description}")
    print("Note: Negative correlation = higher score → faster typing (good)")
    print("      Positive correlation = higher score → slower typing (bad)")
    print("-" * 80)
    
    if not results:
        print("No valid correlations found.")
        return
    
    # Header
    print(f"{'Criterion':<25} {'N':<4} {'Pearson r':<10} {'p-val':<8} {'Spearman r':<11} {'p-val':<8} {'Mean±SD':<12}")
    print("-" * 80)
    
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
        
        print(f"{name:<25} {n:<4} {pr:>7.3f}{p_sig:<3} {pp:<8.3f} {sr:>7.3f}{s_sig:<4} {sp:<8.3f} {mean:.2f}±{std:.2f}")

def print_summary(bigram_results, sentence_results):
    """Print summary of findings."""
    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)
    
    print("\nSignificant correlations (p < 0.05):")
    
    significant_found = False
    
    # Check bigram results
    for criterion, data in bigram_results.items():
        if data['spearman_p'] < 0.05:
            significant_found = True
            direction = "GOOD" if data['spearman_r'] < 0 else "BAD"
            print(f"  {data['name']} (bigrams): r={data['spearman_r']:.3f}, p={data['spearman_p']:.3f} [{direction}]")
    
    # Check sentence results  
    for criterion, data in sentence_results.items():
        if data['spearman_p'] < 0.05:
            significant_found = True
            direction = "GOOD" if data['spearman_r'] < 0 else "BAD"
            print(f"  {data['name']} (sentences): r={data['spearman_r']:.3f}, p={data['spearman_p']:.3f} [{direction}]")
    
    if not significant_found:
        print("  No statistically significant correlations found.")
    
    print("\nInterpretation:")
    print("- GOOD: Higher criterion score → faster typing (validates Dvorak's principle)")
    print("- BAD: Higher criterion score → slower typing (contradicts Dvorak's principle)")
    print("- Significance levels: * p<0.05, ** p<0.01, *** p<0.001")

def main():
    """Main analysis function."""
    print("Dvorak-10 Criteria Correlation Analysis")
    print("=" * 50)
    
    # Read data files
    print("Reading data files...")
    bigram_times_file = '../process_keystroke_data/output/bigram_times.csv'
    sentence_times_file = '../process_keystroke_data/output/sentence_times.csv'
    
    # Actually read the data files
    bigrams, bigram_times = read_bigram_times(bigram_times_file)
    sentences, sentence_times = read_sentence_times(sentence_times_file)
    
    if not bigrams and not sentences:
        print("Error: No valid data found in CSV files.")
        sys.exit(1)
    
    results = {}
    
    # Create output directory for plots
    Path("plots").mkdir(exist_ok=True)
    print("Creating plots in 'plots/' directory...")
    
    # Analyze bigram correlations (criteria 3-10)
    if bigrams:
        bigram_results = analyze_bigram_correlations(bigrams, bigram_times)
        results['bigrams'] = bigram_results
        print_correlation_results(bigram_results, "BIGRAM ANALYSIS (Same-Hand Criteria 3-10)", 
                                "interkey intervals (ms)")
    
    # Analyze sentence correlations (criteria 1-2)  
    if sentences:
        sentence_results = analyze_sentence_correlations(sentences, sentence_times)
        results['sentences'] = sentence_results
        print_correlation_results(sentence_results, "SENTENCE ANALYSIS (Two-Hand Criteria 1-2)",
                                "sentence typing times (ms)")
    
    # Create summary comparison plot
    if 'bigrams' in results or 'sentences' in results:
        bigram_res = results.get('bigrams', {})
        sentence_res = results.get('sentences', {})
        plot_summary_comparison(bigram_res, sentence_res)
        print(f"\nSummary plot created: plots/summary_correlations.png")
    
    # Print summary
    if 'bigrams' in results and 'sentences' in results:
        print_summary(results['bigrams'], results['sentences'])
    elif 'bigrams' in results:
        print_summary(results['bigrams'], {})
    elif 'sentences' in results:
        print_summary({}, results['sentences'])
    
    print(f"\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Key outputs saved to 'plots/' directory:")
    print(f"- Individual criterion scatter plots (score vs time)")
    print(f"- Score distribution histograms")
    print(f"- Multicollinearity heatmaps") 
    print(f"- Summary correlation comparison")
    print(f"\nKey findings:")
    print(f"- Effect sizes categorized by Cohen's guidelines")
    print(f"- Multiple comparisons correction applied")
    print(f"- Multicollinearity between criteria assessed")
    print(f"- Focus on practical significance given large sample size")

if __name__ == "__main__":
    main()
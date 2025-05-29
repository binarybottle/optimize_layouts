import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

def analyze_letter_bigram_frequencies():
    """
    Analyze the cumulative bigram frequency contribution of letters,
    starting from most frequent to least frequent letters.
    """
    
    # Read the CSV files
    print("Loading data...")
    letter_freq = pd.read_csv('input/letter_frequencies_english.csv')
    bigram_freq = pd.read_csv('input/letter_pair_frequencies_english.csv')
    
    # Sort letters from most frequent to least frequent
    letter_freq_sorted = letter_freq.sort_values('score', ascending=False).reset_index(drop=True)
    
    total_bigram_freq = bigram_freq['score'].sum()
    total_letter_freq = letter_freq['score'].sum()
    
    print(f"Loaded {len(letter_freq)} letters and {len(bigram_freq)} bigrams")
    print(f"Total letter frequency: {total_letter_freq:,}")
    print(f"Total bigram frequency: {total_bigram_freq:,}")
    print("\nLetters ordered from most to least frequent:")
    print(letter_freq_sorted[['item', 'score']].head(10))
    
    # Create a mapping of bigrams to their frequencies
    bigram_dict = dict(zip(bigram_freq['item_pair'], bigram_freq['score']))
    
    # For each letter, find all bigrams that contain it
    def get_bigrams_containing_letter(letter, bigram_dict):
        """Return all bigrams and their frequencies that contain the given letter"""
        relevant_bigrams = {}
        for bigram, freq in bigram_dict.items():
            if letter in bigram:
                relevant_bigrams[bigram] = freq
        return relevant_bigrams
    
    # Calculate cumulative bigram frequencies
    results = []
    cumulative_bigrams = set()
    cumulative_freq = 0
    
    print("\nCalculating cumulative bigram frequencies...")
    print("Format: Letter: +X new bigrams, cumulative: Y (Z.Z%)")
    print("-" * 65)
    
    for idx, row in letter_freq_sorted.iterrows():
        letter = row['item']
        letter_freq_val = row['score']
        
        # Find bigrams containing this letter
        letter_bigrams = get_bigrams_containing_letter(letter, bigram_dict)
        
        # Add new bigrams to cumulative set
        new_bigrams = set(letter_bigrams.keys()) - cumulative_bigrams
        new_bigram_freq = sum(letter_bigrams[bg] for bg in new_bigrams)
        
        cumulative_bigrams.update(letter_bigrams.keys())
        cumulative_freq += new_bigram_freq
        
        # Calculate total frequency of all bigrams containing this letter
        total_letter_bigram_freq = sum(letter_bigrams.values())
        
        cumulative_percentage = (cumulative_freq / total_bigram_freq) * 100
        letter_percentage = (letter_freq_val / total_letter_freq) * 100
        bigram_contribution_percentage = (total_letter_bigram_freq / total_bigram_freq) * 100
        
        results.append({
            'letter': letter,
            'letter_frequency': letter_freq_val,
            'letter_frequency_percentage': letter_percentage,
            'letters_included_so_far': idx + 1,
            'new_bigrams_added': len(new_bigrams),
            'new_bigram_frequency': new_bigram_freq,
            'new_bigram_frequency_percentage': (new_bigram_freq / total_bigram_freq) * 100,
            'total_bigrams_so_far': len(cumulative_bigrams),
            'cumulative_bigram_frequency': cumulative_freq,
            'cumulative_percentage': cumulative_percentage,
            'letter_bigram_frequency': total_letter_bigram_freq,
            'letter_bigram_percentage': bigram_contribution_percentage
        })
        
        # Print progress with both absolute and percentage values
        print(f"'{letter}': +{len(new_bigrams):2d} new bigrams (+{new_bigram_freq:>12,}, "
              f"+{new_bigram_freq/total_bigram_freq*100:4.1f}%), "
              f"cumulative: {cumulative_freq:>15,} ({cumulative_percentage:5.1f}%)")
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Export to CSV
    output_filename = 'cumulative_bigram_analysis.csv'
    results_df.to_csv(output_filename, index=False)
    print(f"\nResults exported to {output_filename}")
    
    # Analyze letters with least bigram contribution
    analyze_least_contributing_letters(results_df, total_bigram_freq)
    
    # Create visualizations
    create_visualizations(results_df, letter_freq_sorted, bigram_freq)
    
    return results_df

def analyze_least_contributing_letters(results_df, total_bigram_freq):
    """Analyze which letters contribute least to bigram frequencies"""
    
    print("\n" + "="*80)
    print("LETTERS WITH LEAST BIGRAM CONTRIBUTION")
    print("="*80)
    
    # Sort by letter bigram frequency contribution (ascending)
    least_contributing = results_df.sort_values('letter_bigram_frequency').head(10)
    
    print("Top 10 letters contributing LEAST to bigram frequencies:")
    print("-" * 75)
    print(f"{'Letter':<6} {'Bigram Freq':<15} {'% of Total':<10} {'# Bigrams':<10}")
    print("-" * 75)
    
    for _, row in least_contributing.iterrows():
        print(f"'{row['letter']}'     {row['letter_bigram_frequency']:>12,}   "
            f"{row['letter_bigram_percentage']:>7.2f}%    "
            f"{row['new_bigrams_added']:>8d}")
            
    print(f"\nLowest contributing letters analysis:")
    print("-" * 50)
    bottom_3 = least_contributing.head(3)
    for _, row in bottom_3.iterrows():
        efficiency = row['letter_bigram_frequency'] / row['letter_frequency'] if row['letter_frequency'] > 0 else 0
        print(f"'{row['letter']}': {row['letter_bigram_frequency']:,} bigram freq "
              f"({row['letter_bigram_percentage']:.2f}%) from {row['letter_frequency']:,} letter freq "
              f"({row['letter_frequency_percentage']:.2f}%) - efficiency ratio: {efficiency:.2f}")

def create_visualizations(results_df, letter_freq, bigram_freq):
    """Create comprehensive visualizations of the analysis"""
    
    # Set up the plotting style with better spacing
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with better spacing and larger size
    fig = plt.figure(figsize=(18, 14))
    
    # Adjust subplot spacing to prevent overlap
    plt.subplots_adjust(hspace=0.45, wspace=0.35, top=0.92, bottom=0.12, left=0.10, right=0.95)
    
    # 1. Cumulative bigram frequency progression
    ax1 = plt.subplot(3, 3, 1)
    plt.plot(results_df['letters_included_so_far'], 
             results_df['cumulative_bigram_frequency'] / 1e9, 
             'b-', linewidth=2.5, marker='o', markersize=5)
    plt.xlabel('Letters Included (Most → Least Frequent)', fontsize=9)
    plt.ylabel('Cumulative Bigram Frequency (Billions)', fontsize=9)
    plt.title('Cumulative Bigram Frequency', fontsize=10, pad=10)
    plt.grid(True, alpha=0.3)
    
    # 2. Percentage of total bigram frequency covered
    ax2 = plt.subplot(3, 3, 2)
    plt.plot(results_df['letters_included_so_far'], 
             results_df['cumulative_percentage'], 
             'g-', linewidth=2.5, marker='s', markersize=5)
    plt.xlabel('Letters Included (Most → Least Frequent)', fontsize=9)
    plt.ylabel('Percentage of Total Bigram Frequency (%)', fontsize=9)
    plt.title('Percentage Coverage of Total Bigram Frequency', fontsize=10, pad=15)
    plt.grid(True, alpha=0.3)
    
    # 3. New bigrams added at each step
    ax3 = plt.subplot(3, 3, 3)
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
    bars = plt.bar(results_df['letters_included_so_far'], 
                   results_df['new_bigrams_added'],
                   alpha=0.8, color=colors)
    plt.xlabel('Letters Included (Most → Least Frequent)', fontsize=9)
    plt.ylabel('New Bigrams Added', fontsize=9)
    plt.title('New Bigrams Added at Each Step', fontsize=10, pad=15)
    plt.grid(True, alpha=0.3)
    
    # 4. Letter frequency vs bigram contribution
    ax4 = plt.subplot(3, 3, 4)
    scatter = plt.scatter(results_df['letter_frequency'] / 1e9, 
                         results_df['letter_bigram_frequency'] / 1e9,
                         alpha=0.7, s=80, c=range(len(results_df)), cmap='plasma')
    
    # Add letter labels for all points but with better positioning
    for i, row in results_df.iterrows():
        plt.annotate(row['letter'], 
                    (row['letter_frequency'] / 1e9, row['letter_bigram_frequency'] / 1e9),
                    xytext=(3, 3), textcoords='offset points', fontsize=9, alpha=0.8)
    
    plt.xlabel('Letter Frequency (Billions)', fontsize=9)
    plt.ylabel('Total Bigram Frequency for Letter (Billions)', fontsize=9)
    plt.title('Letter Frequency vs Bigram Contribution', fontsize=10, pad=15)
    plt.grid(True, alpha=0.3)
    
    # 5. Marginal contribution (new bigram frequency added)
    ax5 = plt.subplot(3, 3, 5)
    colors = plt.cm.plasma(np.linspace(0, 1, len(results_df)))
    bars = plt.bar(range(len(results_df)), 
                   results_df['new_bigram_frequency'] / 1e6,
                   alpha=0.8, color=colors)
    plt.xlabel('Letter Addition Order (Most → Least Frequent)', fontsize=9)
    plt.ylabel('New Bigram Frequency Added (Millions)', fontsize=9)
    plt.title('Marginal Bigram Frequency Contribution', fontsize=10, pad=15)
    
    # Add letter labels for x-axis (every 3rd)
    tick_positions = range(0, len(results_df), 5)
    tick_labels = [results_df.iloc[i]['letter'] for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # 6. Cumulative bigram count
    ax6 = plt.subplot(3, 3, 6)
    plt.plot(results_df['letters_included_so_far'], 
             results_df['total_bigrams_so_far'], 
             'r-', linewidth=2.5, marker='^', markersize=5)
    plt.xlabel('Letters Included (Most → Least Frequent)', fontsize=9)
    plt.ylabel('Total Unique Bigrams Covered', fontsize=9)
    plt.title('Cumulative Count of Unique Bigrams', fontsize=10, pad=15)
    plt.grid(True, alpha=0.3)
    
    # 7. Letter contribution percentage
    ax7 = plt.subplot(3, 3, 7)
    plt.bar(range(len(results_df)), 
            results_df['letter_bigram_percentage'],
            alpha=0.7, color='teal')
    plt.xlabel('Letter Order (Most → Least Frequent)', fontsize=9)
    plt.ylabel('Bigram Contribution Percentage (%)', fontsize=9)
    plt.title('Individual Letter Bigram Contribution Percentage', fontsize=10, pad=15)
    
    # Add letter labels for x-axis (every 3rd)
    tick_positions = range(0, len(results_df), 5)
    tick_labels = [results_df.iloc[i]['letter'] for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # 8. Efficiency: Bigram contribution vs Letter frequency
    ax8 = plt.subplot(3, 3, 8)
    efficiency = results_df['letter_bigram_frequency'] / results_df['letter_frequency']
    bars = plt.bar(range(len(results_df)), efficiency, alpha=0.7, color='coral')
    plt.xlabel('Letter Order (Most → Least Frequent)', fontsize=9)
    plt.ylabel('Bigram/Letter Frequency Ratio', fontsize=9)
    plt.title('Letter Efficiency in Bigram Generation', fontsize=10, pad=15)
    
    # Add letter labels for x-axis (every 3rd)
    plt.xticks(tick_positions, tick_labels, fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # 9. Least contributing letters highlight
    ax9 = plt.subplot(3, 3, 9)
    least_contrib = results_df.nsmallest(10, 'letter_bigram_frequency')
    plt.bar(range(len(least_contrib)), 
            least_contrib['letter_bigram_frequency'] / 1e6,
            alpha=0.8, color='red')
    plt.xlabel('Letters (Least Contributing)', fontsize=9)
    plt.ylabel('Bigram Frequency Contribution (Millions)', fontsize=9)
    plt.title('10 Least Contributing Letters to Bigrams', fontsize=10, pad=15)
    
    # Add letter labels
    plt.xticks(range(len(least_contrib)), least_contrib['letter'], fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('letter_bigram_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # Create a detailed summary table
    create_summary_table(results_df)

def create_summary_table(results_df):
    """Create and display a summary table with key insights"""
    
    print("\n" + "="*90)
    print("COMPREHENSIVE SUMMARY ANALYSIS")
    print("="*90)
    
    total_bigram_freq = results_df['cumulative_bigram_frequency'].iloc[-1]
    total_letter_freq = results_df['letter_frequency'].sum()
    
    # Key milestones with absolute values
    milestones = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    print(f"\nTotal bigram frequency covered: {total_bigram_freq:,} (100.0%)")
    print(f"Total letter frequency: {total_letter_freq:,}")
    print(f"Total unique bigrams covered: {results_df['total_bigrams_so_far'].iloc[-1]}")
    
    print(f"\nMilestones (Letters needed to reach X% of total bigram frequency):")
    print("-" * 80)
    print(f"{'Milestone':<10} {'Letters':<8} {'Through':<8} {'Absolute Frequency':<20} {'Percentage'}")
    print("-" * 80)
    
    for milestone in milestones:
        target_freq = total_bigram_freq * milestone
        # Find first row where cumulative frequency >= target
        try:
            milestone_row = results_df[results_df['cumulative_bigram_frequency'] >= target_freq].iloc[0]
            print(f"{milestone*100:>6.1f}%    {milestone_row['letters_included_so_far']:>2d}       "
                  f"'{milestone_row['letter']}'      {milestone_row['cumulative_bigram_frequency']:>15,}   "
                  f"{milestone_row['cumulative_percentage']:>6.1f}%")
        except IndexError:
            print(f"{milestone*100:>6.1f}%    --       --      Not reached in dataset")
    
    print(f"\nTop 10 letters by bigram frequency contribution (absolute + percentage):")
    print("-" * 85)
    print(f"{'Letter':<6} {'Bigram Contribution':<20} {'% of Total':<12} {'Letter Freq':<15} {'% Letter'}")
    print("-" * 85)
    top_contributors = results_df.nlargest(10, 'letter_bigram_frequency')
    for _, row in top_contributors.iterrows():
        print(f"'{row['letter']}'     {row['letter_bigram_frequency']:>15,}   "
              f"{row['letter_bigram_percentage']:>8.2f}%    "
              f"{row['letter_frequency']:>12,}   {row['letter_frequency_percentage']:>6.2f}%")
    
    print(f"\nLeast contributing letters (absolute + percentage):")
    print("-" * 85)
    print(f"{'Letter':<6} {'Bigram Contribution':<20} {'% of Total':<12} {'New Bigrams':<12} {'Efficiency'}")
    print("-" * 85)
    bottom_10 = results_df.nsmallest(10, 'letter_bigram_frequency')
    for _, row in bottom_10.iterrows():
        efficiency = row['letter_bigram_frequency'] / row['letter_frequency'] if row['letter_frequency'] > 0 else 0
        print(f"'{row['letter']}'     {row['letter_bigram_frequency']:>15,}   "
              f"{row['letter_bigram_percentage']:>8.2f}%    "
              f"{row['new_bigrams_added']:>8d}      {efficiency:>8.2f}")
    
    # Additional insights
    print(f"\nAdditional Insights:")
    print("-" * 50)
    
    # Most efficient letters (highest bigram/letter frequency ratio)
    efficiency_series = results_df['letter_bigram_frequency'] / results_df['letter_frequency']
    most_efficient = results_df.loc[efficiency_series.idxmax()]
    least_efficient = results_df.loc[efficiency_series.idxmin()]
    
    print(f"Most efficient letter (bigram/letter ratio): '{most_efficient['letter']}' "
          f"(ratio: {efficiency_series.max():.2f})")
    print(f"Least efficient letter: '{least_efficient['letter']}' "
          f"(ratio: {efficiency_series.min():.2f})")
    
    # Letters that add the most new bigrams
    max_new_bigrams = results_df.loc[results_df['new_bigrams_added'].idxmax()]
    print(f"Letter adding most new bigrams: '{max_new_bigrams['letter']}' "
          f"({max_new_bigrams['new_bigrams_added']} new bigrams)")
    
    # Cumulative coverage milestones
    print(f"\nFirst 5 letters cover: {results_df.iloc[4]['cumulative_percentage']:.1f}% "
          f"({results_df.iloc[4]['cumulative_bigram_frequency']:,})")
    print(f"First 10 letters cover: {results_df.iloc[9]['cumulative_percentage']:.1f}% "
          f"({results_df.iloc[9]['cumulative_bigram_frequency']:,})")

if __name__ == "__main__":
    print("Letter-Bigram Frequency Analysis")
    print("Analyzing from Most Frequent to Least Frequent Letters")
    print("=" * 70)
    
    try:
        results_df = analyze_letter_bigram_frequencies()
        
        print(f"\nAnalysis complete!")
        print(f"Results saved to 'cumulative_bigram_analysis.csv'")
        print(f"Visualizations saved to 'letter_bigram_analysis.png'")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required CSV files. Please ensure both files are in the current directory:")
        print(f"- letter_frequencies_english.csv")
        print(f"- letter_pair_frequencies_english.csv")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
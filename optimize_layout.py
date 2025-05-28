# optimize_layout.py
"""
Layout optimization software.

This script consolidates all optimization logic to support both 
Single-Objective Optimization (SOO) and Multi-Objective Optimization (MOO).
"""

import argparse
import time
import pandas as pd
from typing import Dict, Tuple
from pathlib import Path

# Import our consolidated modules
from config import Config, load_config, print_config_summary
from scoring import LayoutScorer, prepare_scoring_arrays
from search import single_objective_search, multi_objective_search
from display import (print_optimization_header, print_search_space_info, 
                    print_soo_results, print_moo_results, visualize_keyboard_layout,
                    save_soo_results_to_csv, save_moo_results_to_csv)
from validation import run_validation_suite

#-----------------------------------------------------------------------------
# Data loading
#-----------------------------------------------------------------------------
def load_normalized_scores(config: Config) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Load normalized scores from CSV files.
    
    Args:
        config: Configuration object containing file paths
        
    Returns:
        Tuple of (item_scores, item_pair_scores, position_scores, position_pair_scores)
    """
    def load_score_dict(filepath: str, key_col: str, score_col: str = 'score') -> Dict:
        """Helper to load score dictionary from CSV."""
        df = pd.read_csv(filepath)
        return {row[key_col].lower(): float(row[score_col]) for _, row in df.iterrows()}
    
    def load_pair_score_dict(filepath: str, pair_col: str, score_col: str = 'score') -> Dict:
        """Helper to load pair score dictionary from CSV."""
        df = pd.read_csv(filepath)
        result = {}
        for _, row in df.iterrows():
            pair_str = str(row[pair_col])
            if len(pair_str) == 2:
                key = (pair_str[0].lower(), pair_str[1].lower())
                result[key] = float(row[score_col])
        return result
    
    # Load all score dictionaries
    item_scores = load_score_dict(config.paths.item_scores_file, 'item')
    item_pair_scores = load_pair_score_dict(config.paths.item_pair_scores_file, 'item_pair')
    position_scores = load_score_dict(config.paths.position_scores_file, 'position')  
    position_pair_scores = load_pair_score_dict(config.paths.position_pair_scores_file, 'position_pair')
    
    print(f"Loaded {len(item_scores)} item scores")
    print(f"Loaded {len(item_pair_scores)} item pair scores")
    print(f"Loaded {len(position_scores)} position scores")
    print(f"Loaded {len(position_pair_scores)} position pair scores")
    
    return item_scores, item_pair_scores, position_scores, position_pair_scores

#-----------------------------------------------------------------------------
# Optimization functions
#-----------------------------------------------------------------------------
def run_single_objective_optimization(config: Config, n_solutions: int = 5, verbose: bool = False) -> None:
    """
    Run single-objective optimization and display results with complete layout scores.
    
    Args:
        config: Configuration object
        n_solutions: Number of top solutions to find
        verbose: Whether to show detailed scoring breakdown
    """
    print_optimization_header("SOO", config)
    print_config_summary(config) 
    print_search_space_info(config)
    
    # Display the number of solutions being searched for
    print(f"  Target solutions: {n_solutions}")

    # Show initial layout
    if config.visualization.print_keyboard:
        print("\nInitial Layout:")
        visualize_keyboard_layout(
            mapping=None,
            title="Positions to Optimize", 
            config=config,
            items_to_display=config.optimization.items_assigned,
            positions_to_display=config.optimization.positions_assigned
        )
    
    # Load data and create scorer
    print("\nLoading scoring data...")
    normalized_scores = load_normalized_scores(config)
    
    print("Preparing optimization arrays...")
    arrays = prepare_scoring_arrays(
        items_to_assign=list(config.optimization.items_to_assign),
        positions_to_assign=list(config.optimization.positions_to_assign),
        norm_item_scores=normalized_scores[0],
        norm_item_pair_scores=normalized_scores[1],
        norm_position_scores=normalized_scores[2],
        norm_position_pair_scores=normalized_scores[3],
        items_assigned=list(config.optimization.items_assigned) if config.optimization.items_assigned else None,
        positions_assigned=list(config.optimization.positions_assigned) if config.optimization.positions_assigned else None
    )
    
    scorer = LayoutScorer(arrays, mode='combined')
    print(f"Scorer initialized: {scorer.mode_name}")
    
    # Run optimization
    print(f"\nSearching for top {n_solutions} solutions...")
    start_time = time.time()
    
    results, nodes_processed, nodes_pruned = single_objective_search(
        config, scorer, n_solutions
    )
    
    elapsed_time = time.time() - start_time
    
    # Display results with complete scores
    if results:
        print_soo_results(results, config, scorer, normalized_scores, n_solutions, verbose)
        
        # Save results with complete scores
        csv_path = save_soo_results_to_csv(results, config, normalized_scores)
        print(f"\nResults saved to: {csv_path}")
    else:
        print("\nNo solutions found!")
    
    # Final summary
    print(f"\nOptimization Summary:")
    print(f"  Solutions found: {len(results)}")
    print(f"  Nodes processed: {nodes_processed:,}")
    print(f"  Nodes pruned: {nodes_pruned:,}")
    print(f"  Total time: {elapsed_time:.2f}s")

def run_multi_objective_optimization(config: Config, max_solutions: int = None, 
                                   time_limit: float = None) -> None:
    """
    Run multi-objective optimization and display results with complete layout scores.
    
    Args:
        config: Configuration object
        max_solutions: Maximum number of Pareto solutions to find
        time_limit: Time limit in seconds
    """
    print_optimization_header("MOO", config)
    print_config_summary(config)
    print_search_space_info(config)

    # Show initial layout
    if config.visualization.print_keyboard:
        print("\nInitial Layout:")
        visualize_keyboard_layout(
            mapping=None,
            title="Positions to Optimize", 
            config=config,
            items_to_display=config.optimization.items_assigned,
            positions_to_display=config.optimization.positions_assigned
        )

    # Load data and create MOO scorer
    print("\nLoading scoring data...")
    normalized_scores = load_normalized_scores(config)
    
    print("Preparing multi-objective arrays...")
    arrays = prepare_scoring_arrays(
        items_to_assign=list(config.optimization.items_to_assign),
        positions_to_assign=list(config.optimization.positions_to_assign),
        norm_item_scores=normalized_scores[0],
        norm_item_pair_scores=normalized_scores[1],
        norm_position_scores=normalized_scores[2],
        norm_position_pair_scores=normalized_scores[3],
        items_assigned=list(config.optimization.items_assigned) if config.optimization.items_assigned else None,
        positions_assigned=list(config.optimization.positions_assigned) if config.optimization.positions_assigned else None
    )
    
    scorer = LayoutScorer(arrays, mode='multi_objective')
    print(f"Multi-objective scorer initialized")
    
    # Run MOO search
    print(f"\nSearching for Pareto-optimal solutions...")
    if max_solutions:
        print(f"  Maximum solutions: {max_solutions}")
    if time_limit:
        print(f"  Time limit: {time_limit}s")
    
    start_time = time.time()
    
    pareto_front, nodes_processed, nodes_pruned = multi_objective_search(
        config, scorer, max_solutions, time_limit
    )
    
    elapsed_time = time.time() - start_time
    
    # Display results with complete scores
    if pareto_front:
        objective_names = ['Item Score', 'Item-Pair Score']
        print_moo_results(pareto_front, config, normalized_scores, objective_names)
        
        # Save results with complete scores
        csv_path = save_moo_results_to_csv(pareto_front, config, normalized_scores, objective_names)
        print(f"\nResults saved to: {csv_path}")
    else:
        print("\nNo Pareto solutions found!")
    
    # Final summary
    print(f"\nMulti-Objective Summary:")
    print(f"  Pareto solutions: {len(pareto_front)}")
    print(f"  Nodes processed: {nodes_processed:,}")
    print(f"  Total time: {elapsed_time:.2f}s")

#-----------------------------------------------------------------------------
# Command-line interface
#-----------------------------------------------------------------------------
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Optimize keyboard layout using consolidated scoring system.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-objective optimization with specified number of solutions (default)
  python optimize_layout.py --config config.yaml --n-solutions 10
  
  # Multi-objective optimization
  python optimize_layout.py --config config.yaml --moo
  
  # With validation and detailed output  
  python optimize_layout.py --config config.yaml --validate --verbose
  
  # Multi-objective with limits
  python optimize_layout.py --config config.yaml --moo --max-solutions 50 --time-limit 300
        """
    )
    
    # Basic options
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed scoring breakdown')
    
    # Optimization mode
    parser.add_argument('--moo', '--multi-objective', action='store_true',
                       help='Use multi-objective optimization (Pareto front)')
    
    # SOO-specific options
    parser.add_argument('--n-solutions', type=int, default=5,
                       help='Number of top solutions to find (SOO only, default: 5)')

    # MOO-specific options
    parser.add_argument('--max-solutions', type=int, default=None,
                       help='Maximum Pareto solutions to find (MOO only)')
    parser.add_argument('--time-limit', type=float, default=None,
                       help='Time limit in seconds (MOO only)')
    
    # Validation options
    parser.add_argument('--validate', action='store_true',
                       help='Run validation suite before optimization')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    try:
        # Load and validate configuration
        print("Loading configuration...")
        config = load_config(args.config)
        
        # Run validation if requested
        if args.validate:
            print("\n" + "="*60)
            print("RUNNING VALIDATION SUITE")
            print("="*60)
            
            validation_passed = run_validation_suite(config)
            
            print("\nProceeding to optimization...\n")
        
        # Run appropriate optimization mode
        if args.moo:
            run_multi_objective_optimization(config, args.max_solutions, args.time_limit)
        else:
            run_single_objective_optimization(config, args.n_solutions, args.verbose)
            
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that all required files exist and paths are correct.")
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please check your configuration file for errors.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease report this error with the traceback above.")

if __name__ == "__main__":
    main()
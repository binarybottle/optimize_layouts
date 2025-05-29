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
from moo_pruning import MOOPruner, create_moo_pruner

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
                                   time_limit: float = None, enable_pruning: bool = True) -> None:
    """
    Run multi-objective optimization and display results with complete layout scores.
    
    Args:
        config: Configuration object
        max_solutions: Maximum number of Pareto solutions to find
        time_limit: Time limit in seconds
        enable_pruning: Whether to use pruning optimization
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
    
    # Create pruner if enabled
    pruner = None
    if enable_pruning:
        items_list = list(config.optimization.items_to_assign)
        positions_list = list(config.optimization.positions_to_assign)
        pruner = create_moo_pruner(normalized_scores, items_list, positions_list)
        print(f"🚀 MOO Pruning enabled!")
        print(f"   Max item score: {max(pruner.max_item_scores.values()):.6f}")
        print(f"   Max pair score: {pruner.max_item_pair_score:.6f}")
    
    # Run MOO search
    print(f"\nSearching for {'pruned ' if enable_pruning else ''}Pareto-optimal solutions...")
    if max_solutions:
        print(f"  Maximum solutions: {max_solutions}")
    if time_limit:
        print(f"  Time limit: {time_limit}s")
    
    start_time = time.time()
    
    # Pass pruner to multi_objective_search
    pareto_front, nodes_processed, nodes_pruned = multi_objective_search(
        config, scorer, max_solutions, time_limit, pruner, enable_pruning
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
    if enable_pruning and nodes_pruned > 0:
        print(f"  Nodes pruned: {nodes_pruned:,}")
        prune_rate = 100 * nodes_pruned / (nodes_processed + nodes_pruned)
        print(f"  Pruning efficiency: {prune_rate:.1f}%")
        speedup_estimate = (nodes_processed + nodes_pruned) / nodes_processed
        print(f"  Estimated speedup: {speedup_estimate:.1f}x")
    print(f"  Total time: {elapsed_time:.2f}s")
    if nodes_processed > 0:
        print(f"  Rate: {nodes_processed/elapsed_time:.0f} nodes/sec")

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
    parser = argparse.ArgumentParser(description='Optimize keyboard layout')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--moo', action='store_true', help='Run multi-objective optimization')
    parser.add_argument('--n-solutions', type=int, default=3, help='Number of solutions to find')
    parser.add_argument('--max-solutions', type=int, help='Maximum solutions for MOO')
    parser.add_argument('--time-limit', type=float, help='Time limit in seconds')
    parser.add_argument('--validate', action='store_true', help='Validate configuration')
    parser.add_argument('--enable-pruning', action='store_true', help='Enable pruning')
    
    args = parser.parse_args()
    
    # Process arguments
    config = load_config(args.config)
    
    if args.validate:
        validation_mode = "moo" if args.moo else "soo"
        print(f"🧪 Running {validation_mode.upper()} validation suite...")
        validation_passed = run_validation_suite(config, quick=False, mode=validation_mode)
        if not validation_passed:
            print("❌ Validation failed. Please fix issues before running optimization.")
            return
        print("✅ Validation passed! Proceeding with optimization...\n")

    if args.moo:
        run_multi_objective_optimization(
            config=config,
            max_solutions=args.max_solutions,
            time_limit=args.time_limit,
            enable_pruning=args.enable_pruning
        )
    else:
        # Regular single-objective optimization
        run_single_objective_optimization(config, args.n_solutions, args.verbose)

if __name__ == "__main__":
    main()

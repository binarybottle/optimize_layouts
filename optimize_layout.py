# optimize_layout.py
"""
Layout optimization software

Supports both Single-Objective Optimization (SOO) and Multi-Objective Optimization (MOO)
with branch-and-bound search to optimize the arrangement of items (and item-pairs)
in positions (and position-pairs). An example use case is optimizing keyboard layouts.

The version of MOO is limited to two (item/item-pair) objectives.
See optimize_layout_general.py for arbitrary number of item-pair objectives.

Usage:
    # Single-objective optimization (default)
    python optimize_layout.py --config config.yaml --n-solutions 100

    # Multi-objective optimization (Pareto front of candidate solutions)
    python optimize_layout.py --config config.yaml --moo

    # Detailed MOO analysis with comprehensive validation
    python optimize_layout.py --moo --max-solutions 50 --detailed --validate

"""

import argparse
import time
import pandas as pd
from typing import Dict, Tuple
from pathlib import Path

# Import our consolidated modules
from config import Config, load_config, print_config_summary
from scoring import LayoutScorer, prepare_scoring_arrays, load_normalized_scores
from search import single_objective_search, multi_objective_search
from display import (print_optimization_header, print_search_space_info, 
                    print_soo_results, print_moo_results, visualize_keyboard_layout,
                    save_soo_results_to_csv, save_moo_results_to_csv)
from validation import run_validation_suite
from moo_analysis import analyze_moo_problem_quality, quick_moo_check

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
                                   time_limit: float = None, processes: int=None) -> None:
    """
    Multi-objective search with configurable process count.
    
    Args:
        config: Configuration object
        scorer: LayoutScorer in multi_objective mode
        max_solutions: Maximum solutions to find
        time_limit: Time limit in seconds  
        processes: Number of parallel processes (None = auto-detect)
    """
    import multiprocessing as mp

    # Auto-detect process count if not specified
    if processes is None:
        processes = mp.cpu_count()
    
    print(f"Using {processes} parallel processes")

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
    
    # multi_objective_search
    pareto_front, nodes_processed, nodes_pruned = multi_objective_search(
        config, scorer, max_solutions, time_limit, processes
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
    if nodes_processed > 0:
        print(f"  Rate: {nodes_processed/elapsed_time:.0f} nodes/sec")

def run_moo_analysis(config: Config, detailed: bool = True, 
                    test_size: int = None, sample_size: int = 200) -> float:
    """Run MOO problem quality analysis."""
    if not detailed:
        status = quick_moo_check(config)
        print(f"🎯 Quick MOO Assessment: {status}")
        return 0.7 if status == "EXCELLENT" else 0.5 if status == "FAIR" else 0.3
    
    result = analyze_moo_problem_quality(config, test_size, sample_size, verbose=True)
    return result.moo_suitability_score

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
    parser.add_argument('--analyze-moo', action='store_true',
                       help='Analyze MOO problem quality and characteristics')
    parser.add_argument('--detailed', action='store_true',
                       help='Run detailed analysis (use with --analyze-moo)')
    parser.add_argument('--test-size', type=int, default=None,
                       help='Number of items to test in analysis (default: auto-detect)')
    parser.add_argument('--sample-size', type=int, default=200,
                       help='Number of solutions to analyze (default: 200)')
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of parallel processes for MOO (default: auto-detect CPU count)')
    
    # Validation options
    parser.add_argument('--validate', action='store_true',
                       help='Run validation suite before optimization')
    
    return parser.parse_args()

def main():
    """Main entry point with integrated MOO analysis."""
    args = parse_arguments()  # Use the comprehensive argument parser
    
    # Load configuration
    config = load_config(args.config)
    
    # Run validation if requested
    if args.validate:
        validation_mode = "moo" if args.moo else "soo"
        print(f"🧪 Running {validation_mode.upper()} validation suite...")
        validation_passed = run_validation_suite(config, quick=False, mode=validation_mode)
        if not validation_passed:
            print("❌ Validation failed. Please fix issues before running optimization.")
            return
        print("✅ Validation passed!\n")
    
    # Run MOO analysis if requested
    if args.analyze_moo:
        suitability_score = run_moo_analysis(
            config=config,
            detailed=args.detailed,
            test_size=args.test_size,
            sample_size=args.sample_size
        )
        
        # If only doing analysis, exit here
        if not args.moo and args.n_solutions == 5:  # Default values = only analysis requested
            return
        
        # If analysis shows poor suitability, warn before optimization
        if suitability_score is not None and suitability_score < 0.3 and args.moo:
            print(f"\n⚠️  Warning: MOO analysis indicates poor suitability ({suitability_score:.2f})")
            print(f"   Consider using SOO instead, or improving scoring functions.")
            
            user_input = input("Continue with MOO anyway? (y/N): ")
            if user_input.lower() not in ['y', 'yes']:
                print("Optimization cancelled.")
                return
        
        print() 
    
    if args.moo:
        run_multi_objective_optimization(
            config=config,
            max_solutions=args.max_solutions,
            time_limit=args.time_limit,
            processes=args.processes)
    else:
        run_single_objective_optimization(config, args.n_solutions, args.verbose)
        
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Multi-Objective Layout Optimizer

Finds Pareto-optimal item-position layouts using frequency-weighted scoring 
across multiple objectives. 

Features:
    - Arbitrary number of objectives from position-pair scoring table
    - Direct score lookup with frequency weighting
    - Pareto-optimal solution discovery
    - Progress tracking and configurable search limits

Output:
    - Pareto front analysis printed to console
    - CSV file with complete results saved to output directory
    - Progress tracking during search
    - Objective statistics and ranges

Usage Examples:

    # Basic MOO with default settings in config.yaml
    python optimize_moo.py --config config.yaml

    # Basic MOO with six of the Engram-7 objectives
    python optimize_moo.py --config config.yaml \
        --objectives engram7_load,engram7_strength,engram7_position,engram7_vspan,engram7_hspan,engram7_sequence

    # With custom settings
    python optimize_moo.py --config config.yaml \
        --objectives engram7_load,engram7_strength,engram7_position,engram7_vspan,engram7_hspan,engram7_sequence \
        --position-pair-score-table input/position_pair_score_table.csv \
        --item-pair-score-table input/item_pair_score_table.csv \
        --weights 1.0,2.0,0.5,0.5,0.5,0.5 --maximize true,true,false,false,false,false \
        --max-solutions 50 --time-limit 1800

    # Validation run
    python optimize_moo.py --config config.yaml --validate --dry-run

"""

import argparse
from email import parser
from logging import config
import time
import sys
import pandas as pd
import datetime
from typing import List, Dict, Tuple
from pathlib import Path

# Local imports
from config import Config, load_config, print_config_summary
from moo_scoring import FrequencyWeightedMOOScorer, validate_frequency_scoring_consistency
from moo_search import moo_search, analyze_pareto_front, validate_pareto_front

def parse_objectives(objectives_str: str, weights_str: str = None, maximize_str: str = None) -> Tuple[List[str], List[float], List[bool]]:
    """
    Parse objectives configuration from command line arguments.
    
    Args:
        objectives_str: Comma-separated objective names
        weights_str: Optional comma-separated weights
        maximize_str: Optional comma-separated maximize flags
        
    Returns:
        Tuple of (objectives, weights, maximize_flags)
    """
    objectives = [obj.strip() for obj in objectives_str.split(',') if obj.strip()]
    
    if not objectives:
        raise ValueError("At least one objective must be specified")
    
    # Parse weights
    if weights_str:
        try:
            weights = [float(w.strip()) for w in weights_str.split(',') if w.strip()]
            if len(weights) != len(objectives):
                raise ValueError(f"Weights count ({len(weights)}) != objectives count ({len(objectives)})")
        except ValueError as e:
            raise ValueError(f"Invalid weights format: {e}")
    else:
        weights = [1.0] * len(objectives)
    
    # Parse maximize flags
    if maximize_str:
        maximize = []
        for flag in maximize_str.split(','):
            flag = flag.strip().lower()
            if flag in ['true', '1', 'yes', 'max', 'maximize']:
                maximize.append(True)
            elif flag in ['false', '0', 'no', 'min', 'minimize']:
                maximize.append(False)
            else:
                raise ValueError(f"Invalid maximize flag: {flag}")
        
        if len(maximize) != len(objectives):
            raise ValueError(f"Maximize flags count ({len(maximize)}) != objectives count ({len(objectives)})")
    else:
        maximize = [True] * len(objectives)
    
    return objectives, weights, maximize


def parse_inf_value(value, default_val):
    """Convert 'Inf' string to None, otherwise return the value."""
    if isinstance(value, str) and value.lower() == 'inf':
        return None
    return value or default_val


def validate_inputs(config: Config, objectives: List[str], position_pair_score_table: str,
                   item_pair_score_table_path: str) -> None:
    """
    Validate all input files and configuration.
    
    Args:
        config: Configuration object
        objectives: List of objective names
        position_pair_score_table: Path to position-pair scoring table
        item_pair_score_table: Path to frequency file
    """
    # Check position-pair scoring table exists and has required objectives
    if not Path(position_pair_score_table).exists():
        raise FileNotFoundError(f"Position-pair scoring table not found: {position_pair_score_table}")
    
    try:
        pp_df = pd.read_csv(position_pair_score_table, dtype={'key_pair': str})
        missing = [obj for obj in objectives if obj not in pp_df.columns]
        if missing:
            raise ValueError(f"Missing objectives in position-pair scoring table: {missing}")
        #print(f"Position-pair scoring table validation: {len(pp_df)} rows, objectives {objectives} found")
    except Exception as e:
        raise ValueError(f"Error validating position-pair scoring table: {e}")

    # Check item-pair scoring table (optional but warn if missing)
    if not Path(item_pair_score_table_path).exists():
        print(f"Warning: Item-pair scoring table not found: {item_pair_score_table_path}")
        print("Will use unweighted scoring (all letter pairs treated equally)")
    else:
        try:
            ip_df = pd.read_csv(item_pair_score_table_path)
            #print(f"Item-pair scoring table validation: {len(ip_df)} rows loaded")
        except Exception as e:
            print(f"Warning: Error reading item-pair scoring table: {e}")

    # Validate configuration
    opt = config.optimization
    n_items = len(opt.items_to_assign)
    n_positions = len(opt.positions_to_assign)
    
    if n_items > n_positions:
        raise ValueError(f"More items ({n_items}) than positions ({n_positions})")
    if n_items < 2:
        raise ValueError("Need at least 2 items for meaningful optimization")


def save_moo_results(pareto_front: List[Dict], config: Config, objectives: List[str]) -> str:
    """
    Save MOO results to CSV file with comprehensive information.
    
    Args:
        pareto_front: List of Pareto-optimal solutions
        config: Configuration object
        objectives: List of objective names
        
    Returns:
        Path to saved CSV file
    """
    if not pareto_front:
        print("No solutions to save")
        return ""
    
    # Prepare results data
    results_data = []
    for i, solution in enumerate(pareto_front, 1):
        mapping = solution['mapping']
        obj_scores = solution['objectives']
        
        # Create layout strings
        items_str = ''.join(sorted(mapping.keys()))
        positions_str = ''.join(mapping[item] for item in sorted(mapping.keys()))
        layout_display = f"{items_str} -> {positions_str}"
        
        # Build result row
        row = {
            'rank': i,
            'items': items_str,
            'positions': positions_str,
            'layout': layout_display
        }
        
        # Add objective scores
        for j, obj in enumerate(objectives):
            score = obj_scores[j] if j < len(obj_scores) else 0.0
            row[obj] = f"{score:.9f}"
        
        # Add combined score
        combined_score = sum(obj_scores) / len(obj_scores) if obj_scores else 0.0
        row['combined_score'] = f"{combined_score:.9f}"
        
        results_data.append(row)
    
    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = Path(config._config_path).stem + '_' if hasattr(config, '_config_path') else ''
    filename = f"moo_results_{config_name}{timestamp}.csv"
    filepath = Path(config.paths.layout_results_folder) / filename
    
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(filepath, index=False)
    
    return str(filepath)


def print_results_summary(pareto_front: List[Dict], objectives: List[str], 
                         search_stats, config: Config) -> None:
    """
    Print comprehensive summary of optimization results.
    
    Args:
        pareto_front: Pareto-optimal solutions found
        objectives: List of objective names  
        search_stats: Search statistics object
        config: Configuration object
    """
    print(f"\n" + "="*80)
    print("MULTI-OBJECTIVE OPTIMIZATION RESULTS")
    print("="*80)
    
    print(f"\nSearch Summary:")
    print(f"  Time elapsed: {search_stats.elapsed_time:.2f}s")
    print(f"  Nodes processed: {search_stats.nodes_processed:,}")
    print(f"  Solutions evaluated: {search_stats.solutions_found:,}")
    print(f"  Pareto front size: {len(pareto_front)}")
    
    if search_stats.nodes_processed > 0:
        rate = search_stats.nodes_processed / search_stats.elapsed_time
        efficiency = search_stats.solutions_found / search_stats.nodes_processed * 100
        print(f"  Search rate: {rate:.0f} nodes/sec")
        print(f"  Solution efficiency: {efficiency:.2f}% nodes yielded solutions")
    
    # Analyze and display Pareto front
    analyze_pareto_front(pareto_front, objectives)
    
    # Validate Pareto front
    is_valid = validate_pareto_front(pareto_front)
    print(f"\nPareto Front Validation: {'PASS' if is_valid else 'FAIL'}")
    
    if not is_valid:
        print("Warning: Pareto front contains dominated solutions")


def run_moo_optimization(config: Config, objectives: List[str], position_pair_score_table: str,
                        weights: List[float], maximize: List[bool],
                        item_pair_score_table: str, max_solutions: int = None, 
                        time_limit: float = None,
                        verbose=False) -> Tuple[List[Dict], object]:
    """
    Run multi-objective optimization with given parameters.
    
    Args:
        config: Configuration object
        objectives: List of objective names
        position_pair_score_table: Path to position-pair scoring table
        weights: Weights for each objective
        maximize: Maximize flags for each objective
        item_pair_score_table: Path to item-pair scoring table
        max_solutions: Maximum solutions to find
        time_limit: Time limit in seconds
        
    Returns:
        Tuple of (pareto_front, search_stats)
    """
    if verbose:
        print("Initializing Multi-Objective Optimizer...")
    
    # Extract items and positions
    items = list(config.optimization.items_to_assign)
    positions = list(config.optimization.positions_to_assign)
    
    if verbose:
        print(f"Creating frequency-weighted scorer...")
    
    # Create scorer
    scorer = FrequencyWeightedMOOScorer(
        objectives=objectives,
        position_pair_score_table=position_pair_score_table,
        items=items,
        positions=positions,
        weights=weights,
        maximize=maximize,
        item_pair_score_table=item_pair_score_table
    )
    
    if verbose:
        print(f"Scorer initialization complete")
    
    # Print objective statistics
    if verbose:
        stats = scorer.get_objective_stats()
        if stats:
            print(f"\nObjective Score Ranges:")
            for obj, stat in stats.items():
                print(f"  {obj}: [{stat['min']:.3f}, {stat['max']:.3f}] mean={stat['mean']:.3f} (n={stat['count']})")
    
    # Run search
    if verbose:
        print(f"\nStarting search...")

    pareto_front, search_stats = moo_search(
        config=config,
        scorer=scorer,
        max_solutions=max_solutions,
        time_limit=time_limit,
        progress_bar=True
    )
    
    return pareto_front, search_stats


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Multi-objective layout optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage Examples:')[1].split('Output:')[0] if 'Usage Examples:' in __doc__ else ""
    )
    
    # Required arguments
    parser.add_argument('--config', required=True, 
                       help='Configuration YAML file')
    parser.add_argument('--objectives', required=False,
                       help='Comma-separated objectives from position-pair scoring table (default from config)')
    
    # Optional objective configuration
    parser.add_argument('--weights', 
                       help='Comma-separated weights for objectives (default: all 1.0)')
    parser.add_argument('--maximize',
                       help='Comma-separated true/false for each objective (default: all true)')

    # Optional input file overrides
    parser.add_argument('--position-pair-score-table',
                        help='Override position-pair scoring table path from config')
    parser.add_argument('--item-pair-score-table', 
                        help='Override item-pair scoring table path from config')

    # Search limits
    parser.add_argument('--max-solutions', type=int, default=None,
                       help='Maximum Pareto solutions (default: from config)')
    parser.add_argument('--time-limit', type=float, default=None,
                       help='Time limit in seconds (default: from config)')
    
    # Utility options
    parser.add_argument('--validate', action='store_true',
                       help='Validate scorer consistency before optimization')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configuration and exit')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    return parser


def main() -> int:
    """Main entry point for MOO layout optimization."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        if args.verbose:
            print_config_summary(config)
        
        # Parse objectives - use config defaults if not specified
        if args.objectives:
            objectives, weights, maximize = parse_objectives(args.objectives, args.weights, args.maximize)
        else:
            # Use config defaults
            objectives = config.moo.default_objectives
            weights = config.moo.default_weights or [1.0] * len(objectives)
            maximize = config.moo.default_maximize or [True] * len(objectives)
            if not objectives:
                raise ValueError("No objectives specified and no default_objectives in config")

        # Use config paths unless overridden
        position_pair_score_table = args.position_pair_score_table or config.paths.position_pair_score_table
        item_pair_score_table = args.item_pair_score_table or config.paths.item_pair_score_table

        if args.verbose:
            print(f"Multi-Objective Configuration:")
            print(f"  Objectives ({len(objectives)}):")
            for i, obj in enumerate(objectives):
                direction = "maximize" if maximize[i] else "minimize"
                print(f"    {i+1}. {obj} (weight: {weights[i]:.2f}, {direction})")
            print(f"  Position-pair scoring table: {position_pair_score_table}")
            print(f"  Item-pair scoring table: {item_pair_score_table}")

        # Validate inputs
        if args.verbose:
            print(f"\nValidating inputs...")
        validate_inputs(config, objectives, position_pair_score_table, item_pair_score_table)
        
        if args.dry_run:
            print(f"\nDry run - configuration validation successful!")
            return 0
        
        # Optional consistency validation
        if args.validate:
            print(f"\nRunning scorer consistency validation...")
            try:
                test_items = config.optimization.items_to_assign[:4]  # Test with first 4 items
                test_positions = config.optimization.positions_to_assign[:4]
                
                validation_scores = validate_frequency_scoring_consistency(
                    items=test_items,
                    positions=test_positions,
                    objectives=objectives[:2],  # Test with first 2 objectives
                    position_pair_score_table=position_pair_score_table,
                    item_pair_score_table=item_pair_score_table,
                    verbose=True
                )
                print(f"Validation passed!")
                
            except Exception as e:
                print(f"Validation failed: {e}")
                return 1
        
        # Handle Inf values for limits
        max_solutions = args.max_solutions or parse_inf_value(config.moo.default_max_solutions, 100000)
        time_limit = args.time_limit or parse_inf_value(config.moo.default_time_limit, 100000.0)

        # Run optimization
        pareto_front, search_stats = run_moo_optimization(
            config=config,
            objectives=objectives,
            position_pair_score_table=position_pair_score_table,
            weights=weights,
            maximize=maximize,
            item_pair_score_table=item_pair_score_table,
            max_solutions=max_solutions,
            time_limit=time_limit,
            verbose=args.verbose
        )

        # Display results
        print_results_summary(pareto_front, objectives, search_stats, config)
        
        # Save results
        if pareto_front:
            csv_path = save_moo_results(pareto_front, config, objectives)
            print(f"\nResults saved to: {csv_path}")
        else:
            print(f"\nNo solutions found!")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\nOptimization interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
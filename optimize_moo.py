#!/usr/bin/env python3
"""
Multi-Objective Layout Optimizer

Finds Pareto-optimal item-position layouts using weighted scoring 
across multiple objectives. 

Features:
    - Arbitrary number of objectives from position-pair scoring table
    - Direct score lookup with weighting
    - Pareto-optimal solution discovery
    - Progress tracking and configurable search limits

Output:
    - Pareto front analysis printed to console
    - CSV file with complete results saved to output directory
    - Progress tracking during search
    - Objective statistics and ranges

Usage Examples:

    # Basic MOO with default settings in config.yaml (including exhaustive search)
    python optimize_moo.py --config config.yaml

   # Branch-and-bound
    poetry run python optimize_moo.py --config config.yaml --objectives engram6_strength,engram6_curl,engram6_rows,engram6_columns,engram6_order,engram6_3key_order --search-mode branch-bound

    # With custom settings
    python optimize_moo.py --config config.yaml \
        --objectives engram6_strength,engram6_curl,engram6_rows,engram6_columns,engram6_order,engram6_3key_order
        --position-pair-score-table input/position_pair_score_table.csv \
        --item-pair-score-table input/item_pair_score_table.csv \
        --weights 1.0,2.0,0.5,0.5,0.5,0.5 --maximize true,true,false,false,false,false \
        --max-solutions 50 --time-limit 1800

    # Validation run
    python optimize_moo.py --config config.yaml --validate --dry-run

"""

import argparse
from html import parser
import time
import sys
import pandas as pd
import datetime
from typing import List, Dict, Tuple
from pathlib import Path

# Local imports
from config import Config, load_config, print_config_summary
from moo_scoring import WeightedMOOScorer, validate_item_pair_scoring_consistency
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
                   item_pair_score_table_path: str, position_triple_score_table: str = None) -> None:
    """
    Validate all input files and configuration.
    
    Args:
        config: Configuration object
        objectives: List of objective names
        position_pair_score_table: Path to position-pair scoring table
        item_pair_score_table: Path to item-pair scoring table
        position_triple_score_table: Path to position-triple scoring table (optional)
    """
    # Load position-pair scoring table and check which objectives are available
    if not Path(position_pair_score_table).exists():
        raise FileNotFoundError(f"Position-pair scoring table not found: {position_pair_score_table}")
    
    try:
        pp_df = pd.read_csv(position_pair_score_table, dtype={'key_pair': str})
        bigram_objectives = [obj for obj in objectives if obj in pp_df.columns]
        print(f"Position-pair scoring table validation: {len(pp_df)} rows, {len(bigram_objectives)} bigram objectives found")
    except Exception as e:
        raise ValueError(f"Error reading position-pair scoring table: {e}")

    # Load position-triple scoring table if provided and check which objectives are available
    trigram_objectives = []
    if position_triple_score_table and Path(position_triple_score_table).exists():
        try:
            pt_df = pd.read_csv(position_triple_score_table, dtype={'position_triple': str})
            trigram_objectives = [obj for obj in objectives if obj in pt_df.columns]
            print(f"Position-triple scoring table validation: {len(pt_df)} rows, {len(trigram_objectives)} trigram objectives found")
        except Exception as e:
            print(f"Warning: Error reading position-triple scoring table: {e}")
    elif position_triple_score_table:
        print(f"Warning: Position-triple scoring table not found: {position_triple_score_table}")
    
    # Check that all objectives are found in either table
    all_found_objectives = set(bigram_objectives + trigram_objectives)
    missing_objectives = [obj for obj in objectives if obj not in all_found_objectives]
    
    if missing_objectives:
        raise ValueError(f"Missing objectives not found in any scoring table: {missing_objectives}")

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
    

def save_moo_results(pareto_front: List[Dict], config: Config, objectives: List[str], 
                     weights: List[float] = None, maximize: List[bool] = None) -> str:
    """
    Save MOO results to CSV file with comprehensive information including configuration metadata.
    """
    if not pareto_front:
        print("No solutions to save")
        return ""
    
    # Get configuration metadata
    opt = config.optimization
    config_metadata = {
        'config_items_to_assign': ''.join(opt.items_to_assign),
        'config_positions_to_assign': ''.join(opt.positions_to_assign),
        'config_items_assigned': ''.join(opt.items_assigned) if opt.items_assigned else '',
        'config_positions_assigned': ''.join(opt.positions_assigned) if opt.positions_assigned else '',
        'config_items_constrained': ''.join(opt.items_to_constrain_set) if opt.items_to_constrain_set else '',
        'config_positions_constrained': ''.join(opt.positions_to_constrain_set) if opt.positions_to_constrain_set else '',
        'objectives_used': ','.join(objectives),
        'weights_used': ','.join(map(str, weights)) if weights else '',
        'maximize_used': ','.join(map(str, maximize)) if maximize else ''
    }
    
    # Prepare results data
    results_data = []
    for i, solution in enumerate(pareto_front, 1):
        mapping = solution['mapping']
        obj_scores = solution['objectives']
        
        # Build the order that matches what's shown in console output
        opt = config.optimization
        items_assigned = list(opt.items_assigned) if opt.items_assigned else []
        items_to_assign = list(opt.items_to_assign) if opt.items_to_assign else []

        # Use the same order as the complete problem space: assigned + to_assign
        expected_order = items_assigned + items_to_assign  # etao + insrhldcum = etaoinsrhldcum

        items_str = ''.join(expected_order)
        positions_str = ''.join(mapping[item] for item in expected_order)
        layout_display = f"{items_str} -> {positions_str}"
        
        # Build result row with configuration metadata
        row = {
            'rank': i,
            'items': items_str,           # Now shows complete layout
            'positions': positions_str,   # Now shows complete positions
            'layout': layout_display,     # Now shows complete mapping
            **config_metadata
        }
        
        # Objective scores
        for j, obj in enumerate(objectives):
            score = obj_scores[j] if j < len(obj_scores) else 0.0
            row[obj] = f"{score:.9f}"
        
        # Combined score
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
    """Print comprehensive summary of optimization results."""
    print(f"\n" + "="*80)
    print("MULTI-OBJECTIVE OPTIMIZATION RESULTS")
    print("="*80)
    
    print(f"\nSearch Summary:")
    print(f"  Time elapsed: {search_stats.elapsed_time:.2f}s")
    print(f"  Nodes processed: {search_stats.nodes_processed:,}")
    if hasattr(search_stats, 'nodes_pruned') and search_stats.nodes_pruned > 0:
        prune_rate = search_stats.nodes_pruned / search_stats.nodes_processed * 100
        print(f"  Nodes pruned: {search_stats.nodes_pruned:,} ({prune_rate:.1f}%)")
    print(f"  Solutions evaluated: {search_stats.solutions_found:,}")
    print(f"  Pareto front size: {len(pareto_front)}")
    
    if search_stats.nodes_processed > 0:
        rate = search_stats.nodes_processed / search_stats.elapsed_time
        efficiency = search_stats.solutions_found / search_stats.nodes_processed * 100
        print(f"  Search rate: {rate:.0f} nodes/sec")
        print(f"  Solution efficiency: {efficiency:.2f}% nodes yielded solutions")
    
    # Use the new analysis function
    analyze_pareto_front(pareto_front, objectives)
    
    # Validate Pareto front
    is_valid = validate_pareto_front(pareto_front)
    print(f"\nPareto Front Validation: {'PASS' if is_valid else 'FAIL'}")
    
    if not is_valid:
        print("Warning: Pareto front contains dominated solutions")


def run_moo_optimization(config: Config, objectives: List[str], position_pair_score_table: str,
                        weights: List[float], maximize: List[bool],
                        item_pair_score_table: str, 
                        position_triple_score_table: str = None,
                        item_triple_score_table: str = None,
                        max_solutions: int = None, 
                        time_limit: float = None,
                        search_mode: str = 'branch-bound',
                        verbose=False) -> Tuple[List[Dict], object]:
    """
    Run multi-objective optimization with selectable search algorithm.
    """
    if verbose:
        print("Initializing Multi-Objective Optimizer...")
    
    # Handle the config semantics correctly
    opt = config.optimization
    
    # Extract the separate sets from config
    items_assigned = list(opt.items_assigned) if opt.items_assigned else []
    items_to_assign = list(opt.items_to_assign) if opt.items_to_assign else []
    positions_assigned = list(opt.positions_assigned) if opt.positions_assigned else []
    positions_to_assign = list(opt.positions_to_assign) if opt.positions_to_assign else []
    
    # Combine to create the full problem space
    all_items = items_assigned + items_to_assign
    all_positions = positions_assigned + positions_to_assign
    
    if verbose:
        print(f"\nCombining problem space:")
        print(f"  Pre-assigned items: {items_assigned}")
        print(f"  Items to optimize: {items_to_assign}")
        print(f"  -> Complete item set: {all_items}")
        print(f"  Pre-assigned positions: {positions_assigned}")
        print(f"  Positions available: {positions_to_assign}")
        print(f"  -> Complete position set: {all_positions}")
    
    # Use the combined sets for the scorer (this is what was missing!)
    items = all_items
    positions = all_positions
    
    if verbose:
        print(f"Creating weighted scorer...")
    
    # Create scorer with the COMPLETE problem space
    scorer = WeightedMOOScorer(
        objectives=objectives,
        position_pair_score_table=position_pair_score_table,
        items=items,           # Now includes pre-assigned + to-be-optimized
        positions=positions,   # Now includes pre-assigned + available
        weights=weights,
        maximize=maximize,
        item_pair_score_table=item_pair_score_table,
        position_triple_score_table=position_triple_score_table,
        item_triple_score_table=item_triple_score_table,
        verbose=verbose
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
    
    # Verify the pre-assignments will work
    if items_assigned and positions_assigned:
        print(f"\nPre-assignment verification:")
        print(f"  Pre-assigned: {items_assigned} -> {positions_assigned}")
        
        # Check that all pre-assigned items exist in the full item set
        missing_items = [item for item in items_assigned if item not in items]
        if missing_items:
            print(f"  ERROR: Pre-assigned items not in full item set: {missing_items}")
        
        # Check that all pre-assigned positions exist in the full position set  
        missing_positions = [pos for pos in positions_assigned if pos.upper() not in [p.upper() for p in positions]]
        if missing_positions:
            print(f"  ERROR: Pre-assigned positions not in full position set: {missing_positions}")
        
        if not missing_items and not missing_positions:
            print(f"  ✓ Pre-assignments are valid")
            
            # Calculate actual search space
            remaining_items = len(items_to_assign)
            remaining_positions = len(positions_to_assign)
            from math import factorial
            if remaining_positions >= remaining_items:
                search_space = factorial(remaining_positions) // factorial(remaining_positions - remaining_items)
                print(f"  ✓ Search space: {remaining_items} items in {remaining_positions} positions = {search_space:,} permutations")
    
    # Run search with selected algorithm
    pareto_front, search_stats = moo_search(
        config=config,
        scorer=scorer,
        max_solutions=max_solutions,
        time_limit=time_limit,
        progress_bar=True,
        verbose=verbose,
        search_mode=search_mode  # PASS THE SEARCH MODE
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
    
    # Search method options
    parser.add_argument('--search-mode', choices=['branch-bound', 'exhaustive'], 
                       default='exhaustive',
                       help='Search algorithm: branch-bound (default, faster) or exhaustive (guaranteed complete)')
    parser.add_argument('--search-all', action='store_true',
                       help='Force exhaustive search (same as --search-mode exhaustive)')

    # Search limits
    parser.add_argument('--max-solutions', type=int, default=None,
                       help='Maximum Pareto solutions (default: from config)')
    parser.add_argument('--time-limit', type=float, default=None,
                       help='Time limit in seconds (default: from config)')
    
    # Optional objective configuration
    parser.add_argument('--weights', 
                       help='Comma-separated weights for objectives (default: all 1.0)')
    parser.add_argument('--maximize',
                       help='Comma-separated true/false for each objective (default: all true)')

    # Optional input file overrides
    parser.add_argument('--item-pair-score-table', 
                        help='Override item-pair scoring table path from config')
    parser.add_argument('--position-pair-score-table',
                        help='Override position-pair scoring table path from config')
    parser.add_argument('--item-triple-score-table',
                        help='Override item-triple scoring table path from config')
    parser.add_argument('--position-triple-score-table',
                        help='Override position-triple scoring table path from config')

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
        position_triple_score_table = args.position_triple_score_table or config.paths.position_triple_score_table  # Add this line

        if args.verbose:
            print(f"Multi-Objective Configuration:")
            print(f"  Objectives ({len(objectives)}):")
            for i, obj in enumerate(objectives):
                direction = "maximize" if maximize[i] else "minimize"
                print(f"    {i+1}. {obj} (weight: {weights[i]:.2f}, {direction})")
            print(f"  Position-pair scoring table: {position_pair_score_table}")
            print(f"  Item-pair scoring table: {item_pair_score_table}")
            print(f"  Position-triple scoring table: {position_triple_score_table}")  # Add this line

        # Validate inputs
        if args.verbose:
            print(f"\nValidating inputs...")
            validate_inputs(config, objectives, position_pair_score_table, item_pair_score_table, position_triple_score_table)        

        if args.dry_run:
            print(f"\nDry run - configuration validation successful!")
            return 0
        
        # Optional consistency validation
        if args.validate:
            print(f"\nRunning scorer consistency validation...")
            try:
                test_items = config.optimization.items_to_assign[:4]  # Test with first 4 items
                test_positions = config.optimization.positions_to_assign[:4]
                
                validation_scores = validate_item_pair_scoring_consistency(
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

        item_triple_score_table = args.item_triple_score_table or config.paths.item_triple_score_table

        # Determine search mode
        search_mode = args.search_mode
        if args.search_all:
            search_mode = 'exhaustive'
            print("Using exhaustive search (--search-all specified)")
        
        # Handle Inf values for limits
        max_solutions = args.max_solutions or parse_inf_value(config.moo.default_max_solutions, 100000)
        time_limit = args.time_limit or parse_inf_value(config.moo.default_time_limit, 100000.0)

        # Run optimization with search mode
        pareto_front, search_stats = run_moo_optimization(
            config=config,
            objectives=objectives,
            position_pair_score_table=position_pair_score_table,
            weights=weights,
            maximize=maximize,
            item_pair_score_table=item_pair_score_table,
            position_triple_score_table=position_triple_score_table,
            item_triple_score_table=item_triple_score_table,
            max_solutions=max_solutions,
            time_limit=time_limit,
            search_mode=search_mode,  # ADD THIS
            verbose=args.verbose
        )

        # Display results (use the new analyze_pareto_front function)
        print_results_summary(pareto_front, objectives, search_stats, config)
        
        # Save results
        if pareto_front:
            csv_path = save_moo_results(pareto_front, config, objectives, weights, maximize)
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
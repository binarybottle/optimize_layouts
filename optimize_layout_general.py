#!/usr/bin/env python3
"""
General Multi-Objective Layout Optimizer with Branch-and-Bound

Supports arbitrary number of objectives from keypair score tables using the existing
search infrastructure. Integrates cleanly with the existing scoring and search systems.
Saves Pareto-optimal solutions (typically 10-100 solutions)

Usage (with 6 of 7 Engram-7 metrics):
    # Branch-and-bound MOO (saves Pareto front)
    python optimize_layout_general.py --config config.yaml \
        --objectives engram7_load_normalized,engram7_strength_normalized,engram7_position_normalized,engram7_vspan_normalized,engram7_hspan_normalized,engram7_sequence_normalized \
        --keypair-table input/keypair_scores_detailed.csv

    # Brute force (for small problems or validation)
    python optimize_layout_general.py --config config.yaml \
        --objectives engram7_load_normalized,engram7_strength_normalized,engram7_position_normalized,engram7_vspan_normalized,engram7_hspan_normalized,engram7_sequence_normalized \
        --keypair-table input/keypair_scores_detailed.csv \
        --brute-force --max-solutions 100 --time-limit 3600

"""

import argparse
import time
import numpy as np
import pandas as pd
import datetime
from typing import List, Dict, Tuple
from pathlib import Path
from itertools import permutations
from numba import jit
import multiprocessing as mp
from math import factorial

from config import Config, load_config
from display import print_optimization_header, print_search_space_info
from scoring import ScoringArrays, LayoutScorer, ScoreComponents, load_normalized_scores, prepare_scoring_arrays
from search import single_objective_search, multi_objective_search, get_next_item_jit, validate_constraints_jit
from validation import run_validation_suite

#-----------------------------------------------------------------------------
# General MOO Scoring Classes
#-----------------------------------------------------------------------------
class TransformedScorer:
    """Standalone class for transformed scoring to ensure proper functionality."""
    
    def __init__(self, base_scorer, weight, maximize_flag):
        self.base_scorer = base_scorer
        self.weight = weight
        self.maximize = maximize_flag
        
    def score_layout(self, mapping, return_components=False):
        score = self.base_scorer.score_layout(mapping, return_components)
        if return_components:
            transformed_score = score[0] * self.weight
            if not self.maximize:
                transformed_score *= -1.0
            return (transformed_score,) + score[1:]
        else:
            transformed_score = score * self.weight
            if not self.maximize:
                transformed_score *= -1.0
            return transformed_score
    
    def get_components(self, mapping):
        return self.base_scorer.get_components(mapping)
        
    def clear_cache(self):
        return self.base_scorer.clear_cache()

class GeneralMOOScorer:
    """
    Multi-objective scorer optimized for single-process execution.
    Pre-creates all objective scorers for efficiency and reliability.
    """
    
    def __init__(self, config: Config, objectives: List[str], keypair_table_path: str,
                 items: List[str], positions: List[str], 
                 weights: List[float], maximize: List[bool]):
        
        self.objectives = objectives
        self.objective_weights = weights
        self.objective_maximize = maximize
        self.items = items
        self.positions = positions
        
        print(f"Creating GeneralMOOScorer: {len(objectives)} objectives, {len(items)} items, {len(positions)} positions")
        
        # Load normalized scores using existing infrastructure
        normalized_scores = load_normalized_scores(config)
        norm_item_scores, norm_item_pair_scores, norm_position_scores, norm_position_pair_scores = normalized_scores
        
        # Load keypair table for objective-specific modifications
        keypair_df = pd.read_csv(keypair_table_path, dtype={'key_pair': str})
        print(f"Loaded keypair table: {len(keypair_df)} rows")
        
        # Validate objectives exist in keypair table
        missing = [obj for obj in objectives if obj not in keypair_df.columns]
        if missing:
            raise ValueError(f"Missing objectives in keypair table: {missing}")
        
        # Pre-create all objective-specific scorers
        self.objective_scorers = {}
        total_memory = 0

        for i, obj in enumerate(objectives):
            print(f"  [{i+1}/{len(objectives)}] Computing {obj}...")
            
            # Create modified position pair scores for this objective
            modified_position_pair_scores = norm_position_pair_scores.copy()

            # Apply objective-specific modifications from keypair table
            objective_values = []
            for _, row in keypair_df.iterrows():
                key_pair = str(row['key_pair']).strip("'\"")
                if len(key_pair) == 2 and not pd.isna(row[obj]):
                    pos1, pos2 = key_pair[0].upper(), key_pair[1].upper()
                    
                    if pos1 in positions and pos2 in positions:
                        original_score = norm_position_pair_scores.get((pos1.lower(), pos2.lower()), 1.0)
                        objective_modifier = float(row[obj])
                        
                        blend_factor = 0.5
                        modified_score = blend_factor * original_score + (1 - blend_factor) * objective_modifier
                        modified_position_pair_scores[(pos1.lower(), pos2.lower())] = modified_score
                        
                        objective_values.append(objective_modifier)

            if objective_values:
                print(f"      Range: [{np.min(objective_values):.6f}, {np.max(objective_values):.6f}]")
            
            # Create scoring arrays for this objective
            arrays = prepare_scoring_arrays(
                items_to_assign=items,
                positions_to_assign=positions,
                norm_item_scores=norm_item_scores,
                norm_item_pair_scores=norm_item_pair_scores,
                norm_position_scores=norm_position_scores,
                norm_position_pair_scores=modified_position_pair_scores,
                items_assigned=list(config.optimization.items_assigned) if config.optimization.items_assigned else None,
                positions_assigned=list(config.optimization.positions_assigned) if config.optimization.positions_assigned else None
            )
            
            # Create base scorer
            scorer = LayoutScorer(arrays, mode='pair_only')
            
            # Apply weight and direction transformations if needed
            if weights[i] != 1.0 or not maximize[i]:
                scorer = TransformedScorer(scorer, weights[i], maximize[i])

            self.objective_scorers[obj] = scorer
            total_memory += arrays.item_scores.nbytes + arrays.item_pair_matrix.nbytes + arrays.position_matrix.nbytes
                
        print(f"Pre-computation complete: {total_memory / (1024*1024):.1f}MB total")
        
        # Store the first arrays object for compatibility with existing MOO search infrastructure
        self.arrays = list(self.objective_scorers.values())[0].arrays if self.objective_scorers else None
    
    def score_layout(self, mapping: np.ndarray, return_components: bool = False):
        """
        Score layout using all objectives.
        Returns list of scores, one for each objective.
        """
        scores = []
        
        for obj in self.objectives:
            score = self.objective_scorers[obj].score_layout(mapping)
            scores.append(score)
        
        if return_components:
            # For display purposes, include combined total
            combined_total = sum(scores) / len(scores) if scores else 0.0
            return scores + [combined_total]
        else:
            return scores
        
    def clear_cache(self):
        """Clear all scorer caches."""
        for scorer in self.objective_scorers.values():
            scorer.clear_cache()

#-----------------------------------------------------------------------------
# Search Wrapper Functions
#-----------------------------------------------------------------------------
def run_general_moo(config: Config, keypair_table: str, objectives: List[str],
                   weights: List[float], maximize: List[bool], **kwargs) -> List[Dict]:
    print("Using general MOO with single-process branch-and-bound search")
    print("Note: General MOO is optimized for single-process execution")
    
    items = list(config.optimization.items_to_assign)
    positions = list(config.optimization.positions_to_assign)
    
    scorer = GeneralMOOScorer(config, objectives, keypair_table, items, positions, weights, maximize)
    
    pareto_front, nodes_processed, nodes_pruned = multi_objective_search(
        config, scorer, 
        max_solutions=kwargs.get('max_solutions'),
        time_limit=kwargs.get('time_limit'),
        processes=1  # Force single-process
    )
    
    if pareto_front and len(objectives) > 0:
        objectives_matrix = np.array([sol['objectives'] for sol in pareto_front])
        print(f"\nObjective Statistics:")
        for i, obj in enumerate(objectives):
            if i < objectives_matrix.shape[1]:
                values = objectives_matrix[:, i]
                print(f"  {obj}: [{np.min(values):.6f}, {np.max(values):.6f}]")
    
    return pareto_front

def run_brute_force_general_moo(config: Config, keypair_table: str, objectives: List[str],
                               weights: List[float], maximize: List[bool], **kwargs) -> List[Dict]:
    """Run general MOO with brute force enumeration (for small problems or validation)."""
    
    max_solutions = kwargs.get('max_solutions')
    time_limit = kwargs.get('time_limit')
    
    print("Using BRUTE FORCE enumeration (all permutations)")
    print("WARNING: This will be very slow for problems larger than ~8 items!")
    
    items = list(config.optimization.items_to_assign)
    positions = list(config.optimization.positions_to_assign)
    n_items = len(items)
    n_positions = len(positions)
    
    scorer = GeneralMOOScorer(config, objectives, keypair_table, items, positions, weights, maximize)
    
    total_perms = 1
    for i in range(n_items):
        total_perms *= (n_positions - i)
    
    print(f"Total permutations to evaluate: {total_perms:,}")
    
    estimated_seconds = total_perms / 1000
    if estimated_seconds > 3600:
        print(f"Estimated time: {estimated_seconds/3600:.1f} hours")
        if total_perms > 10000000:
            response = input(f"\nThis will take a very long time. Continue? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Brute force search cancelled.")
                return []
    
    from search import pareto_dominates
    
    pareto_front = []
    pareto_objectives = []
    evaluated = 0
    start_time = time.time()
    
    print(f"\nStarting brute force enumeration...")
    
    try:
        for perm in permutations(range(n_positions), n_items):
            evaluated += 1
            
            if time_limit and (time.time() - start_time) > time_limit:
                print(f"\nTime limit reached at {evaluated:,} evaluations")
                break
            if max_solutions and len(pareto_front) >= max_solutions:
                print(f"\nSolution limit reached at {evaluated:,} evaluations")
                break
            
            mapping_array = np.array(perm, dtype=np.int32)
            objectives = scorer.score_layout(mapping_array)
            
            is_non_dominated = True
            dominated_indices = []
            
            for i, existing_obj in enumerate(pareto_objectives):
                if pareto_dominates(existing_obj, objectives):
                    is_non_dominated = False
                    break
                elif pareto_dominates(objectives, existing_obj):
                    dominated_indices.append(i)
            
            if is_non_dominated:
                for i in reversed(sorted(dominated_indices)):
                    del pareto_front[i]
                    del pareto_objectives[i]
                
                item_mapping = {items[i]: positions[perm[i]] for i in range(n_items)}
                solution = {
                    'mapping': item_mapping,
                    'objectives': objectives
                }
                
                pareto_front.append(solution)
                pareto_objectives.append(objectives)
            
            if evaluated % 10000 == 0:
                elapsed = time.time() - start_time
                rate = evaluated / elapsed if elapsed > 0 else 0
                percent = (evaluated / total_perms) * 100
                print(f"  Progress: {evaluated:,}/{total_perms:,} ({percent:.2f}%) - {rate:.0f} layouts/sec - {len(pareto_front)} Pareto solutions")
    
    except KeyboardInterrupt:
        print(f"\nInterrupted by user at {evaluated:,} evaluations")
    
    elapsed_time = time.time() - start_time
    
    print(f"\nBrute Force Results:")
    print(f"  Evaluated: {evaluated:,} of {total_perms:,} permutations ({evaluated/total_perms*100:.2f}%)")
    print(f"  Pareto solutions: {len(pareto_front)}")
    print(f"  Search time: {elapsed_time:.2f}s")
    if evaluated > 0:
        print(f"  Rate: {evaluated/elapsed_time:.0f} layouts/sec")
    
    return pareto_front

#-----------------------------------------------------------------------------
# Results Management
#-----------------------------------------------------------------------------
def save_general_results(pareto_front: List[Dict], config: Config, objectives: List[str], 
                        method: str = "general") -> str:
    """Save general MOO results to CSV."""
    
    results_data = []
    for i, solution in enumerate(pareto_front):
        mapping = solution['mapping']
        obj_scores = solution['objectives']
        
        items_str = ''.join(mapping.keys())
        positions_str = ''.join(mapping.values())
        
        row = {
            'rank': i + 1,
            'items': items_str,
            'positions': positions_str,
            'layout': f"{items_str} -> {positions_str}"
        }
        
        for j, obj in enumerate(objectives):
            row[f'{obj}'] = obj_scores[j] if j < len(obj_scores) else 0.0
        
        results_data.append(row)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = Path(config._config_path).stem if hasattr(config, '_config_path') else 'unknown'
    filename = f"{method}_moo_results_{config_name}_{timestamp}.csv"
    filepath = Path(config.paths.layout_results_folder) / filename
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results_data).to_csv(filepath, index=False)
    
    return str(filepath)

#-----------------------------------------------------------------------------
# Argument Parsing and Main
#-----------------------------------------------------------------------------
def parse_objectives(objectives_str: str, weights_str: str = None, maximize_str: str = None):
    """Parse objectives configuration from command line."""
    objectives = [obj.strip() for obj in objectives_str.split(',') if obj.strip()]
    
    if weights_str:
        weights = [float(w.strip()) for w in weights_str.split(',') if w.strip()]
        if len(weights) != len(objectives):
            raise ValueError(f"Weights count ({len(weights)}) != objectives count ({len(objectives)})")
    else:
        weights = [1.0] * len(objectives)
    
    if maximize_str:
        maximize = []
        for flag in maximize_str.split(','):
            flag = flag.strip().lower()
            maximize.append(flag in ['true', '1', 'yes', 'max', 'maximize'])
        if len(maximize) != len(objectives):
            raise ValueError(f"Maximize flags count ({len(maximize)}) != objectives count ({len(objectives)})")
    else:
        maximize = [True] * len(objectives)
    
    return objectives, weights, maximize

def main():
    parser = argparse.ArgumentParser(
        description="Multi-objective optimization for layout problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Branch-and-bound with arbitrary objectives (recommended)
  python optimize_layout_general.py --config config.yaml \\
    --objectives engram8_columns_normalized,engram8_curl_normalized,engram8_home_normalized \\
    --keypair-table data/keypair_scores.csv
  # With search limits
  python optimize_layout_general.py --config config.yaml \\
    --objectives engram8_columns_normalized,engram8_curl_normalized \\
    --keypair-table data/keypair_scores.csv \\
    --max-solutions 50 --time-limit 3600
  # Brute force (for small problems or validation)
  python optimize_layout_general.py --config config.yaml \\
    --objectives engram8_columns_normalized,engram8_curl_normalized \\
    --keypair-table data/keypair_scores.csv \\
    --brute-force --max-solutions 100
        """
    )
    
    # Required arguments
    parser.add_argument('--config', required=True, help='Configuration YAML file')
    parser.add_argument('--objectives', required=True, help='Comma-separated objectives from keypair table')
    parser.add_argument('--keypair-table', required=True, help='CSV file with keypair data')
    
    # Optional objective configuration
    parser.add_argument('--weights', help='Comma-separated weights for objectives (default: all 1.0)')
    parser.add_argument('--maximize', help='Comma-separated true/false for each objective (default: all true)')
    
    # Search method
    parser.add_argument('--brute-force', action='store_true', 
                       help='Use brute force enumeration instead of branch-and-bound')
    
    # Search limits
    parser.add_argument('--max-solutions', type=int, help='Maximum Pareto solutions')
    parser.add_argument('--time-limit', type=float, help='Time limit in seconds')
    parser.add_argument('--processes', type=int, default=1, help='Number of parallel processes (default: 1)')
    
    # Utility options
    parser.add_argument('--validate', action='store_true', help='Run validation before optimization')
    parser.add_argument('--dry-run', action='store_true', help='Validate configuration only')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Parse objectives
        objectives, weights, maximize = parse_objectives(args.objectives, args.weights, args.maximize)
        
        print(f"Configuration loaded:")
        print(f"  Items to assign: {config.optimization.items_to_assign}")
        print(f"  Positions to assign: {config.optimization.positions_to_assign}")
        print(f"  Keypair table: {args.keypair_table}")
        print(f"  Search method: {'Brute Force' if args.brute_force else 'Branch-and-Bound'}")
        print(f"  Objectives ({len(objectives)}):")
        for i, obj in enumerate(objectives):
            direction = "maximize" if maximize[i] else "minimize"
            print(f"    {i+1}. {obj} (weight: {weights[i]}, {direction})")
        
        # Validate files exist
        if not Path(args.keypair_table).exists():
            raise FileNotFoundError(f"Keypair table not found: {args.keypair_table}")
        
        if args.dry_run:
            print("\nDry run - configuration validation successful!")
            return 0
        
        # Run validation if requested
        if args.validate:
            print("\nRunning validation...")
            if not run_validation_suite(config, quick=False, mode="moo"):
                print("Validation failed!")
                return 1
            print("Validation passed!")
        
        # Show search space analysis
        print_optimization_header("General MOO", config)
        print_search_space_info(config)
        
        # Prepare search parameters
        search_kwargs = {
            'max_solutions': args.max_solutions,
            'time_limit': args.time_limit,
            'processes': args.processes
        }

        # Regular MOO or brute force
        if args.brute_force:
            results = run_brute_force_general_moo(
                config, args.keypair_table, objectives, weights, maximize, **search_kwargs
            )
            method = "brute_force"
        else:
            results = run_general_moo(
                config, args.keypair_table, objectives, weights, maximize, **search_kwargs
            )
            method = "branch_and_bound"
        
        # Save results
        if results:
            csv_path = save_general_results(results, config, objectives, method)
            print(f"\nResults saved to: {csv_path}")
        else:
            print("No solutions found!")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()
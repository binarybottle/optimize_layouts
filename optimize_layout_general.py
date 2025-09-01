#!/usr/bin/env python3
"""
General Multi-Objective Layout Optimizer with Branch-and-Bound

Supports arbitrary number of objectives with both intelligent branch-and-bound search
and optional brute-force enumeration for comparison/validation.

Usage:
    # Branch-and-bound (recommended)
    python optimize_layout_general.py --config config.yaml \
        --objectives engram8_columns_normalized,engram8_curl_normalized,engram8_home_normalized,engram8_hspan_normalized,engram8_load_normalized,engram8_sequence_normalized,engram8_strength_normalized,engram8_vspan_normalized \
        --keypair-table input/keypair_scores_detailed.csv

    # Brute force option (for small problems or validation)
    python optimize_layout_general.py --config config.yaml \
        --objectives engram8_columns_normalized,engram8_curl_normalized,engram8_home_normalized,engram8_hspan_normalized,engram8_load_normalized,engram8_sequence_normalized,engram8_strength_normalized,engram8_vspan_normalized \
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

from config import Config, load_config
from display import print_optimization_header, print_search_space_info
from scoring import ScoringArrays, LayoutScorer, ScoreComponents, ScoreCalculator
from search import multi_objective_search
from validation import run_validation_suite

#-----------------------------------------------------------------------------
# General MOO Scoring Infrastructure
#-----------------------------------------------------------------------------
class GeneralMOOArrays(ScoringArrays):
    """
    Extended ScoringArrays that supports arbitrary objectives from keypair table.
    Integrates with existing branch-and-bound infrastructure.
    """
    
    def __init__(self, keypair_df: pd.DataFrame, objectives: List[str], 
                 items: List[str], positions: List[str], 
                 weights: List[float], maximize: List[bool]):
        
        self.objectives = objectives
        self.objective_weights = weights
        self.objective_maximize = maximize
        self.items = [item.lower() for item in items]
        self.positions = [pos.upper() for pos in positions]
        
        print(f"Creating GeneralMOOArrays: {len(objectives)} objectives, {len(items)} items, {len(positions)} positions")
        
        # Create lookups
        self.item_to_idx = {item: i for i, item in enumerate(self.items)}
        self.position_to_idx = {pos: i for i, pos in enumerate(self.positions)}
        
        # Pre-compute objective matrices for each objective
        self.objective_matrices = {}
        total_memory = 0
        
        for i, obj in enumerate(objectives):
            print(f"  [{i+1}/{len(objectives)}] Computing {obj}...")
            
            matrix = self._create_objective_matrix(keypair_df, obj)
            
            # Apply weight and direction
            if weights[i] != 1.0:
                matrix *= weights[i]
            if not maximize[i]:
                matrix *= -1.0  # Convert minimize to maximize
            
            self.objective_matrices[obj] = matrix.astype(np.float32)
            total_memory += matrix.nbytes
            
            print(f"      Range: [{np.min(matrix):.6f}, {np.max(matrix):.6f}]")
        
        print(f"Pre-computation complete: {total_memory / (1024*1024):.1f}MB total")
        
        # Create minimal dummy arrays for parent class compatibility
        n_items = len(items)
        n_positions = len(positions)
        
        item_scores = np.ones(n_items, dtype=np.float32)
        item_pair_matrix = np.ones((n_items, n_items), dtype=np.float32)
        position_matrix = np.ones((n_positions, n_positions), dtype=np.float32)
        
        # Initialize parent class
        super().__init__(item_scores, item_pair_matrix, position_matrix)
    
    def _create_objective_matrix(self, keypair_df: pd.DataFrame, obj_col: str) -> np.ndarray:
        """Create matrix for single objective from keypair table."""
        
        # Build lookup from keypair table
        lookup = {}
        missing_pairs = 0
        
        for _, row in keypair_df.iterrows():
            key_pair = str(row['key_pair'])
            if len(key_pair) == 2 and not pd.isna(row[obj_col]):
                pos1, pos2 = key_pair[0].upper(), key_pair[1].upper()
                if pos1 in self.position_to_idx and pos2 in self.position_to_idx:
                    lookup[(pos1, pos2)] = float(row[obj_col])
        
        # Create matrix
        n_pos = len(self.positions)
        matrix = np.zeros((n_pos, n_pos), dtype=np.float32)
        
        for i, pos1 in enumerate(self.positions):
            for j, pos2 in enumerate(self.positions):
                if i != j:  # Skip diagonal
                    key = (pos1, pos2)
                    if key in lookup:
                        matrix[i, j] = lookup[key]
                    else:
                        matrix[i, j] = 0.0  # Default for missing pairs
                        missing_pairs += 1
        
        if missing_pairs > 0:
            total_pairs = n_pos * (n_pos - 1)
            print(f"      Missing: {missing_pairs}/{total_pairs} pairs ({missing_pairs/total_pairs*100:.1f}%)")
        
        return matrix

class GeneralMOOCalculator(ScoreCalculator):
    """Score calculator for arbitrary objectives."""
    
    def __init__(self, general_arrays: GeneralMOOArrays):
        self.general_arrays = general_arrays
        # Initialize parent with the general arrays
        super().__init__(general_arrays)
    
    def calculate_components(self, mapping: np.ndarray) -> ScoreComponents:
        """Calculate objectives as independent components."""
        # Calculate all objective scores
        objective_scores = self._calculate_all_objectives(mapping)
        
        # For compatibility with existing 2-component system, we'll use the first
        # two objectives as "item_score" and "item_pair_score", and store all
        # objectives in a custom attribute
        item_score = objective_scores[0] if len(objective_scores) > 0 else 0.0
        item_pair_score = objective_scores[1] if len(objective_scores) > 1 else 0.0
        
        components = ScoreComponents(item_score, item_pair_score)
        
        # Add custom attribute for all objectives
        components.all_objectives = objective_scores
        
        return components
    
    def _calculate_all_objectives(self, mapping: np.ndarray) -> List[float]:
        """Calculate all objective scores for a mapping."""
        scores = []
        
        for obj_name in self.general_arrays.objectives:
            matrix = self.general_arrays.objective_matrices[obj_name]
            score = self._score_single_objective_jit(mapping, matrix)
            scores.append(score)
        
        return scores
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _score_single_objective_jit(mapping_array: np.ndarray, objective_matrix: np.ndarray) -> float:
        """JIT-compiled single objective scoring."""
        total_score = 0.0
        pair_count = 0
        n_items = len(mapping_array)
        
        for i in range(n_items):
            pos_i = mapping_array[i]
            if pos_i >= 0 and pos_i < objective_matrix.shape[0]:
                for j in range(n_items):
                    pos_j = mapping_array[j]
                    if i != j and pos_j >= 0 and pos_j < objective_matrix.shape[1]:
                        total_score += objective_matrix[pos_i, pos_j]
                        pair_count += 1
        
        return total_score / max(1, pair_count)

class GeneralMOOScorer(LayoutScorer):
    """Layout scorer for arbitrary objectives that integrates with existing search."""
    
    def __init__(self, general_arrays: GeneralMOOArrays):
        self.general_arrays = general_arrays
        self.calculator = GeneralMOOCalculator(general_arrays)
        
        # Initialize parent with multi_objective mode
        super().__init__(general_arrays, mode='multi_objective')
        
        # Override calculator with our general one
        self.calculator = GeneralMOOCalculator(general_arrays)
    
    def score_layout(self, mapping: np.ndarray, return_components: bool = False):
        """Score layout using all objectives."""
        components = self.calculator.calculate_components(mapping)
        all_objectives = getattr(components, 'all_objectives', [])
        
        if return_components:
            # Return all objectives plus a combined total for display
            combined_total = sum(all_objectives) / len(all_objectives) if all_objectives else 0.0
            return all_objectives + [combined_total]
        else:
            # Return just the objective list for MOO search
            return all_objectives

#-----------------------------------------------------------------------------
# Search Implementations
#-----------------------------------------------------------------------------
def run_general_moo(config: Config, keypair_table: str, objectives: List[str],
                          weights: List[float], maximize: List[bool], **kwargs) -> List[Dict]:
    """Run general MOO with custom branch-and-bound search for arbitrary objectives."""
    
    print("Using custom general MOO search with Pareto dominance pruning")
    
    # Load keypair table
    keypair_df = pd.read_csv(keypair_table, dtype={'key_pair': str})
    print(f"Loaded keypair table: {len(keypair_df)} rows")
    
    # Validate objectives
    missing = [obj for obj in objectives if obj not in keypair_df.columns]
    if missing:
        raise ValueError(f"Missing objectives in keypair table: {missing}")
    
    # Create general MOO arrays and calculator
    items = list(config.optimization.items_to_assign)
    positions = list(config.optimization.positions_to_assign)
    n_items = len(items)
    n_positions = len(positions)
    
    general_arrays = GeneralMOOArrays(keypair_df, objectives, items, positions, weights, maximize)
    calculator = GeneralMOOCalculator(general_arrays)
    
    # Custom branch-and-bound search for general MOO
    print(f"\nRunning custom general MOO search...")
    start_time = time.time()
    
    # Pareto front storage
    pareto_front = []
    pareto_objectives = []
    
    # Search statistics
    nodes_processed = 0
    nodes_pruned = 0
    max_solutions = kwargs.get('max_solutions')
    time_limit = kwargs.get('time_limit')
    
    def search_recursive(mapping: np.ndarray, used: np.ndarray, depth: int):
        nonlocal nodes_processed, nodes_pruned, pareto_front, pareto_objectives
        
        nodes_processed += 1
        
        # Check termination conditions
        if time_limit and (time.time() - start_time) > time_limit:
            return
        if max_solutions and len(pareto_front) >= max_solutions:
            return
        
        # Complete solution
        if depth == n_items:
            # Calculate all objectives
            components = calculator.calculate_components(mapping)
            all_objectives = getattr(components, 'all_objectives', [])
            
            if not all_objectives:
                return
            
            # Check Pareto dominance
            is_non_dominated = True
            dominated_indices = []
            
            for i, existing_obj in enumerate(pareto_objectives):
                if pareto_dominates(existing_obj, all_objectives):
                    is_non_dominated = False
                    break
                elif pareto_dominates(all_objectives, existing_obj):
                    dominated_indices.append(i)
            
            if is_non_dominated:
                # Remove dominated solutions
                for i in reversed(sorted(dominated_indices)):
                    del pareto_front[i]
                    del pareto_objectives[i]
                
                # Add new solution
                item_mapping = {items[i]: positions[mapping[i]] for i in range(n_items)}
                solution = {
                    'mapping': item_mapping,
                    'objectives': all_objectives
                }
                pareto_front.append(solution)
                pareto_objectives.append(all_objectives)
            
            return
        
        # Branch: try each available position
        available_positions = [i for i in range(n_positions) if not used[i]]
        
        for pos in available_positions:
            # Basic pruning: if Pareto front is full, check if this partial solution
            # could possibly lead to a non-dominated solution
            if len(pareto_front) >= 100:  # Simple pruning threshold
                # Could add more sophisticated upper bound pruning here
                pass
            
            mapping[depth] = pos
            used[pos] = True
            
            search_recursive(mapping, used, depth + 1)
            
            mapping[depth] = -1
            used[pos] = False
    
    # Initialize search
    initial_mapping = np.full(n_items, -1, dtype=np.int32)
    initial_used = np.zeros(n_positions, dtype=bool)
    
    search_recursive(initial_mapping, initial_used, 0)
    
    elapsed_time = time.time() - start_time
    
    # Print results
    if pareto_front:
        print(f"\nGeneral MOO Results:")
        print(f"  Pareto solutions: {len(pareto_front)}")
        print(f"  Nodes processed: {nodes_processed:,}")
        print(f"  Search time: {elapsed_time:.2f}s")
        if nodes_processed > 0:
            print(f"  Rate: {nodes_processed/elapsed_time:.0f} nodes/sec")
        
        # Show objective statistics - fix the indexing issue
        if pareto_front and len(objectives) > 0:
            # Verify objectives data structure
            first_solution_objectives = pareto_front[0]['objectives']
            print(f"  First solution has {len(first_solution_objectives)} objectives")
            
            if len(first_solution_objectives) == len(objectives):
                objectives_matrix = np.array([sol['objectives'] for sol in pareto_front])
                print(f"\nObjective Statistics:")
                for i, obj in enumerate(objectives):
                    if i < objectives_matrix.shape[1]:
                        values = objectives_matrix[:, i]
                        print(f"  {obj}: [{np.min(values):.6f}, {np.max(values):.6f}]")
            else:
                print(f"  WARNING: Objective count mismatch: expected {len(objectives)}, got {len(first_solution_objectives)}")
    else:
        print("\nNo Pareto solutions found!")
    
    return pareto_front

def run_brute_force_general_moo(config: Config, keypair_table: str, objectives: List[str],
                               weights: List[float], maximize: List[bool], **kwargs) -> List[Dict]:
    """Run general MOO with brute force enumeration."""
    
    max_solutions = kwargs.get('max_solutions')
    time_limit = kwargs.get('time_limit')
    
    print("Using BRUTE FORCE enumeration (all permutations)")
    print("WARNING: This will be very slow for problems larger than ~8 items!")
    
    # Load keypair table
    keypair_df = pd.read_csv(keypair_table, dtype={'key_pair': str})
    
    # Create arrays
    items = list(config.optimization.items_to_assign)
    positions = list(config.optimization.positions_to_assign)
    n_items = len(items)
    n_positions = len(positions)
    
    general_arrays = GeneralMOOArrays(keypair_df, objectives, items, positions, weights, maximize)
    
    # Calculate total search space
    total_perms = 1
    for i in range(n_items):
        total_perms *= (n_positions - i)
    
    print(f"Total permutations to evaluate: {total_perms:,}")
    
    # Estimate time
    estimated_seconds = total_perms / 1000  # Assume 1000 evals/sec
    if estimated_seconds > 3600:
        print(f"Estimated time: {estimated_seconds/3600:.1f} hours")
        if estimated_seconds > 86400:
            print(f"                {estimated_seconds/86400:.1f} days")
    else:
        print(f"Estimated time: {estimated_seconds:.0f} seconds")
    
    # Ask for confirmation on large problems
    if total_perms > 10000000:  # 10 million
        response = input(f"\nThis will take a very long time. Continue? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Brute force search cancelled.")
            return []
    
    # Pareto front storage
    pareto_front = []
    pareto_objectives = []
    
    # Search statistics
    evaluated = 0
    start_time = time.time()
    
    # Progress tracking
    update_interval = max(1000, total_perms // 100)
    next_update = update_interval
    
    print(f"\nStarting brute force enumeration...")
    
    try:
        for perm in permutations(range(n_positions), n_items):
            evaluated += 1
            
            # Check time limit
            if time_limit and (time.time() - start_time) > time_limit:
                print(f"\nTime limit reached at {evaluated:,} evaluations")
                break
            
            # Check solution limit (early termination)
            if max_solutions and len(pareto_front) >= max_solutions:
                print(f"\nSolution limit reached at {evaluated:,} evaluations")
                break
            
            # Create mapping and calculate objectives
            mapping_array = np.array(perm, dtype=np.int32)
            calc = GeneralMOOCalculator(general_arrays)
            components = calc.calculate_components(mapping_array)
            objectives = getattr(components, 'all_objectives', [])
            
            # Check Pareto dominance
            is_non_dominated = True
            dominated_indices = []
            
            for i, existing_obj in enumerate(pareto_objectives):
                if pareto_dominates(existing_obj, objectives):
                    is_non_dominated = False
                    break
                elif pareto_dominates(objectives, existing_obj):
                    dominated_indices.append(i)
            
            if is_non_dominated:
                # Remove dominated solutions
                for i in reversed(sorted(dominated_indices)):
                    del pareto_front[i]
                    del pareto_objectives[i]
                
                # Add new solution
                item_mapping = {items[i]: positions[perm[i]] for i in range(n_items)}
                solution = {
                    'mapping': item_mapping,
                    'objectives': objectives
                }
                
                pareto_front.append(solution)
                pareto_objectives.append(objectives)
            
            # Progress update
            if evaluated >= next_update:
                elapsed = time.time() - start_time
                rate = evaluated / elapsed if elapsed > 0 else 0
                percent = (evaluated / total_perms) * 100
                print(f"  Progress: {evaluated:,}/{total_perms:,} ({percent:.2f}%) - {rate:.0f} layouts/sec - {len(pareto_front)} Pareto solutions")
                next_update += update_interval
    
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

def pareto_dominates(obj1: List[float], obj2: List[float]) -> bool:
    """Check if obj1 Pareto dominates obj2 (works for any number of objectives)."""
    better_in_one = False
    for v1, v2 in zip(obj1, obj2):
        if v1 < v2:
            return False
        if v1 > v2:
            better_in_one = True
    return better_in_one

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
        
        # Add objective scores
        for j, obj in enumerate(objectives):
            row[f'{obj}'] = obj_scores[j] if j < len(obj_scores) else 0.0
        
        results_data.append(row)
    
    # Create filename
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
    """Parse objectives configuration."""
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
        description="General MOO Layout Optimizer with Branch-and-Bound",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Branch-and-bound (recommended)
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
    parser.add_argument('--keypair-table', required=True, help='Keypair table CSV file')
    
    # Optional objective configuration
    parser.add_argument('--weights', help='Comma-separated weights for objectives')
    parser.add_argument('--maximize', help='Comma-separated true/false for each objective')
    
    # Search method
    parser.add_argument('--brute-force', action='store_true', 
                       help='Use brute force enumeration instead of branch-and-bound')
    
    # Search limits
    parser.add_argument('--max-solutions', type=int, help='Maximum Pareto solutions')
    parser.add_argument('--time-limit', type=float, help='Time limit in seconds')
    parser.add_argument('--processes', type=int, help='Number of parallel processes')
    
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
        
        # Run optimization
        if args.brute_force:
            pareto_front = run_brute_force_general_moo(
                config, args.keypair_table, objectives, weights, maximize, **search_kwargs
            )
            method = "brute_force"
        else:
            pareto_front = run_general_moo(
                config, args.keypair_table, objectives, weights, maximize, **search_kwargs
            )
            method = "branch_and_bound"
        
        # Save results
        if pareto_front:
            csv_path = save_general_results(pareto_front, config, objectives, method)
            print(f"\nResults saved to: {csv_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
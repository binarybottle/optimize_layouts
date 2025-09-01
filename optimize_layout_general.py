#!/usr/bin/env python3
"""
Standalone Multi-Objective Layout Optimizer with Pre-computed Matrices

Optimized for extreme-memory cluster nodes. Pre-computes all objective matrices
at startup for maximum speed during search.

Usage:
    # Current system compatibility (2 objectives: item + item-pair)
    python optimize_layout_general.py --config config.yaml --mode current --keypair-table data/keypair_scores.csv
    
    # General MOO (arbitrary objectives from key-pair table)
    python optimize_layout_general.py --config config.yaml --mode general --keypair-table data/keypair_scores.csv --objectives comfort_score_normalized,time_total_normalized,engram8_score_normalized
    
    # With custom weights and directions
    python optimize_layout_general.py --config config.yaml --mode general --keypair-table data/keypair_scores.csv --objectives comfort_score_normalized,time_total_normalized --weights 1.0,2.0 --maximize true,false --max-solutions 50 --time-limit 3600

    # Keyboard optimization study command
    poetry run python3 optimize_layout_general.py --config config.yaml --mode general --objectives engram8_columns_normalized,engram8_curl_normalized,engram8_home_normalized,engram8_hspan_normalized,engram8_load_normalized,engram8_sequence_normalized,engram8_strength_normalized,engram8_vspan_normalized --keypair-table ../keyboard_layout_scorers/tables/keypair_scores_detailed.csv
    
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

# Import only what we need from existing modules
from config import Config, load_config
from display import print_optimization_header, print_search_space_info
from validation import run_validation_suite

#-----------------------------------------------------------------------------
# Simple Pre-computed Arrays
#-----------------------------------------------------------------------------
class SimpleObjectiveArrays:
    """Simple pre-computed matrices - standalone implementation."""
    
    def __init__(self, keypair_df: pd.DataFrame, objectives: List[str], 
                 positions: List[str], weights: List[float], maximize: List[bool]):
        
        self.objectives = objectives
        self.positions = [pos.upper() for pos in positions]
        self.n_positions = len(positions)
        self.weights = weights
        self.maximize = maximize
        
        print(f"Pre-computing {len(objectives)} matrices...")
        
        # Create position lookup
        self.position_to_idx = {pos: i for i, pos in enumerate(self.positions)}
        
        # Pre-compute matrices
        self.matrices = {}
        total_memory = 0
        
        for i, obj in enumerate(objectives):
            print(f"  [{i+1}/{len(objectives)}] {obj}...")
            
            matrix = self._create_matrix(keypair_df, obj)
            
            # Apply weight and direction
            matrix *= weights[i]
            if not maximize[i]:
                matrix *= -1.0
            
            self.matrices[obj] = matrix
            total_memory += matrix.nbytes
            
            print(f"      Range: [{np.min(matrix):.6f}, {np.max(matrix):.6f}]")
        
        print(f"Pre-computation complete: {total_memory / (1024*1024):.1f}MB total")
        
        # JIT warmup
        dummy_mapping = np.array([0, 1, 2], dtype=np.int32)
        dummy_matrix = np.random.random((self.n_positions, self.n_positions)).astype(np.float32)
        _score_objective_jit(dummy_mapping, dummy_matrix)
    
    def _create_matrix(self, keypair_df: pd.DataFrame, obj_col: str) -> np.ndarray:
        """Create matrix for single objective."""
        
        # Build lookup from key-pair table
        lookup = {}
        missing_pairs = 0
        
        for _, row in keypair_df.iterrows():
            key_pair = str(row['key_pair'])
            if len(key_pair) == 2 and not pd.isna(row[obj_col]):
                pos1, pos2 = key_pair[0].upper(), key_pair[1].upper()
                if pos1 in self.position_to_idx and pos2 in self.position_to_idx:
                    lookup[(pos1, pos2)] = float(row[obj_col])
        
        # Create matrix
        matrix = np.zeros((self.n_positions, self.n_positions), dtype=np.float32)
        
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
            total_pairs = self.n_positions * (self.n_positions - 1)
            print(f"      Missing: {missing_pairs}/{total_pairs} pairs ({missing_pairs/total_pairs*100:.1f}%)")
        
        return matrix
    
    def calculate_objectives(self, mapping_array: np.ndarray) -> List[float]:
        """Calculate all objectives for a layout."""
        return [_score_objective_jit(mapping_array, self.matrices[obj]) 
                for obj in self.objectives]

@jit(nopython=True, fastmath=True)
def _score_objective_jit(mapping_array: np.ndarray, objective_matrix: np.ndarray) -> float:
    """JIT-compiled objective scoring."""
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

#-----------------------------------------------------------------------------
# Simple Standalone Search (No Integration Issues)
#-----------------------------------------------------------------------------
def pareto_dominates(obj1: List[float], obj2: List[float]) -> bool:
    """Check if obj1 dominates obj2."""
    better_in_one = False
    for v1, v2 in zip(obj1, obj2):
        if v1 < v2:
            return False
        if v1 > v2:
            better_in_one = True
    return better_in_one

def simple_moo_search(arrays: SimpleObjectiveArrays, items: List[str], positions: List[str],
                     max_solutions: int = None, time_limit: float = None) -> List[Dict]:
    """Standalone MOO search - no integration with existing search."""
    
    n_items = len(items)
    n_positions = len(positions)
    
    print(f"Running standalone MOO search...")
    print(f"Problem: {n_items} items in {n_positions} positions")
    
    # Calculate total search space
    total_perms = 1
    for i in range(n_items):
        total_perms *= (n_positions - i)
    
    print(f"Search space: {total_perms:,} permutations")
    
    # Track Pareto front
    pareto_front = []
    pareto_objectives = []
    
    # Search statistics
    evaluated = 0
    start_time = time.time()
    
    # Progress tracking
    update_interval = max(1000, total_perms // 100)
    next_update = update_interval
    
    for perm in permutations(range(n_positions), n_items):
        evaluated += 1
        
        # Check time limit
        if time_limit and (time.time() - start_time) > time_limit:
            print(f"Time limit reached at {evaluated} evaluations")
            break
        
        # Check solution limit
        if max_solutions and len(pareto_front) >= max_solutions:
            print(f"Solution limit reached at {evaluated} evaluations")
            break
        
        # Create mapping and calculate objectives
        mapping_array = np.array(perm, dtype=np.int32)
        objectives = arrays.calculate_objectives(mapping_array)
        
        # Check if non-dominated
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
            rate = evaluated / elapsed
            print(f"  Progress: {evaluated:,}/{total_perms:,} ({evaluated/total_perms*100:.1f}%) - {rate:.0f} layouts/sec - {len(pareto_front)} Pareto solutions")
            next_update += update_interval
    
    elapsed_time = time.time() - start_time
    print(f"Search completed: {evaluated:,} layouts in {elapsed_time:.2f}s ({evaluated/elapsed_time:.0f} layouts/sec)")
    
    return pareto_front

#-----------------------------------------------------------------------------
# Current System Compatibility (Simplified)
#-----------------------------------------------------------------------------
def run_current_system(config: Config, **kwargs):
    """Run current system using existing infrastructure."""
    
    print_optimization_header("Current System", config)
    print_search_space_info(config)
    
    # Use existing infrastructure
    from scoring import load_normalized_scores, prepare_scoring_arrays, LayoutScorer
    from search import multi_objective_search
    
    normalized_scores = load_normalized_scores(config)
    arrays = prepare_scoring_arrays(
        items_to_assign=list(config.optimization.items_to_assign),
        positions_to_assign=list(config.optimization.positions_to_assign),
        norm_item_scores=normalized_scores[0],
        norm_item_pair_scores=normalized_scores[1],
        norm_position_scores=normalized_scores[2],
        norm_position_pair_scores=normalized_scores[3]
    )
    
    scorer = LayoutScorer(arrays, mode='multi_objective')
    
    pareto_front, nodes_processed, nodes_pruned = multi_objective_search(
        config, scorer, **kwargs
    )
    
    if pareto_front:
        from display import save_moo_results_to_csv
        objective_names = ['Item Score', 'Item-Pair Score']
        csv_path = save_moo_results_to_csv(pareto_front, config, normalized_scores, objective_names)
        print(f"Results saved to: {csv_path}")
    
    print(f"Current system summary: {len(pareto_front)} solutions, {nodes_processed:,} nodes")

def run_general_system(config: Config, keypair_table: str, objectives: List[str],
                      weights: List[float], maximize: List[bool], **kwargs):
    """Run general MOO using standalone search."""
    
    print_optimization_header("General MOO", config)
    print(f"Objectives: {objectives}")
    print_search_space_info(config)
    
    # Load key-pair table
    keypair_df = pd.read_csv(keypair_table, dtype={'key_pair': str})
    print(f"Loaded key-pair table: {len(keypair_df)} rows")
    
    # Validate objectives exist
    missing = [obj for obj in objectives if obj not in keypair_df.columns]
    if missing:
        print(f"Error: Missing objectives: {missing}")
        return
    
    # Create arrays and run standalone search
    items = list(config.optimization.items_to_assign)
    positions = list(config.optimization.positions_to_assign)
    
    arrays = SimpleObjectiveArrays(keypair_df, objectives, positions, weights, maximize)
    
    pareto_front = simple_moo_search(
        arrays, items, positions, 
        max_solutions=kwargs.get('max_solutions'),
        time_limit=kwargs.get('time_limit')
    )
    
    if pareto_front:
        # Print results
        print(f"\nGeneral MOO Results:")
        print(f"Pareto solutions: {len(pareto_front)}")
        
        if pareto_front:
            objectives_matrix = np.array([sol['objectives'] for sol in pareto_front])
            print(f"\nObjective Statistics:")
            for i, obj in enumerate(objectives):
                values = objectives_matrix[:, i]
                print(f"  {obj}: [{np.min(values):.6f}, {np.max(values):.6f}]")
        
        # Save results
        csv_path = save_general_results(pareto_front, config, objectives)
        print(f"Results saved to: {csv_path}")
    else:
        print("No Pareto solutions found!")

def save_general_results(pareto_front: List[Dict], config: Config, objectives: List[str]) -> str:
    """Save general MOO results."""
    
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
    
    # Save with simple filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = Path(config._config_path).stem if hasattr(config, '_config_path') else 'unknown'
    filename = f"general_moo_results_{config_name}_{timestamp}.csv"
    filepath = Path(config.paths.output.layout_results_folder) / filename
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results_data).to_csv(filepath, index=False)
    
    return str(filepath)

#-----------------------------------------------------------------------------
# Parsing Functions
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

#-----------------------------------------------------------------------------
# Main Interface
#-----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="General MOO Layout Optimizer")
    
    # Required
    parser.add_argument('--config', required=True, help='Config YAML file')
    parser.add_argument('--mode', choices=['current', 'general'], required=True, help='Mode')
    
    # General mode
    parser.add_argument('--keypair-table', help='Key-pair table CSV (required for general)')
    parser.add_argument('--objectives', help='Comma-separated objectives (required for general)')
    parser.add_argument('--weights', help='Comma-separated weights (optional)')
    parser.add_argument('--maximize', help='Comma-separated true/false (optional)')
    
    # Search parameters
    parser.add_argument('--max-solutions', type=int, help='Max Pareto solutions')
    parser.add_argument('--time-limit', type=float, help='Time limit seconds')
    parser.add_argument('--processes', type=int, help='Processes (ignored for now)')
    
    # Utility
    parser.add_argument('--validate', action='store_true', help='Run validation')
    parser.add_argument('--dry-run', action='store_true', help='Validate only')
    
    args = parser.parse_args()
    
    try:
        # Load config
        config = load_config(args.config)
        
        # Validate requirements
        if args.mode == 'general':
            if not args.keypair_table or not args.objectives:
                print("Error: --keypair-table and --objectives required for general mode")
                return 1
            
            if not Path(args.keypair_table).exists():
                print(f"Error: Key-pair table not found: {args.keypair_table}")
                return 1
            
            objectives, weights, maximize = parse_objectives(args.objectives, args.weights, args.maximize)
            print(f"Parsed objectives: {len(objectives)}")
            for i, obj in enumerate(objectives):
                direction = "maximize" if maximize[i] else "minimize"
                print(f"  {i+1}. {obj} (weight: {weights[i]}, {direction})")
        
        if args.dry_run:
            print("Configuration validated successfully")
            return 0
        
        # Run validation if requested
        if args.validate:
            print("Running validation...")
            if not run_validation_suite(config, quick=False, mode="moo"):
                print("Validation failed")
                return 1
            print("Validation passed!")
        
        # Prepare kwargs
        kwargs = {
            'max_solutions': args.max_solutions,
            'time_limit': args.time_limit,
            'processes': args.processes
        }
        
        # Run optimization
        if args.mode == 'current':
            run_current_system(config, **kwargs)
        else:
            run_general_system(config, args.keypair_table, objectives, weights, maximize, **kwargs)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
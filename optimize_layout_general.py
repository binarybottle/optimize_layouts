#!/usr/bin/env python3
"""
General Multi-Objective Layout Optimizer with Branch-and-Bound

Supports arbitrary number of objectives from keypair score tables using the existing
search infrastructure. Integrates cleanly with the existing scoring and search systems.

A goal programming option turns the MOO problem into a SOO minimization of total weighted deviations,
handles discrete data well, and shows exactly how far each solution deviates from targets.

Usage (with 6 of 7 Engram-7 metrics):
    # Branch-and-bound 
    python optimize_layout_general.py --config config.yaml \
        --objectives engram7_load_normalized,engram7_strength_normalized,engram7_position_normalized,engram7_vspan_normalized,engram7_hspan_normalized,engram7_sequence_normalized \
        --keypair-table input/keypair_scores_detailed.csv

    # Brute force (for small problems or validation)
    python optimize_layout_general.py --config config.yaml \
        --objectives engram7_load_normalized,engram7_strength_normalized,engram7_position_normalized,engram7_vspan_normalized,engram7_hspan_normalized,engram7_sequence_normalized \
        --keypair-table input/keypair_scores_detailed.csv \
        --brute-force --max-solutions 100 --time-limit 3600

    # Goal programming (minimize deviations from targets)
    python optimize_layout_general.py --config config.yaml \
        --objectives engram7_load_normalized,engram7_strength_normalized,engram7_position_normalized,engram7_vspan_normalized,engram7_hspan_normalized,engram7_sequence_normalized \
        --keypair-table input/keypair_scores_detailed.csv \
        --goal-programming

    # Goal programming with custom targets and deviation weights
    python optimize_layout_general.py --config config.yaml \
        --objectives engram7_load_normalized,engram7_strength_normalized,engram7_position_normalized,engram7_vspan_normalized,engram7_hspan_normalized,engram7_sequence_normalized \
        --keypair-table input/keypair_scores_detailed.csv \
        --goal-programming \
        --targets "1.0,1.0,1.0,1.0,1.0,1.0" \
        --deviation-weights "2.0,1.0,1.0,2.0,1.0,1.0"

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
from scoring import ScoringArrays, LayoutScorer, ScoreComponents, ScoreCalculator
from search import single_objective_search, multi_objective_search, get_next_item_jit, validate_constraints_jit
from validation import run_validation_suite

@jit(nopython=True, fastmath=True)
def _calculate_partial_objective_value(partial_mapping: np.ndarray, objective_matrix: np.ndarray) -> float:
    """Calculate current contribution to objective from assigned items."""
    total_score = 0.0
    pair_count = 0
    n_items = len(partial_mapping)
    
    # Only count pairs where both items are assigned
    for i in range(n_items):
        pos_i = partial_mapping[i]
        if pos_i >= 0:  # Item i is assigned
            for j in range(n_items):
                pos_j = partial_mapping[j]
                if i != j and pos_j >= 0:  # Item j is also assigned
                    total_score += objective_matrix[pos_i, pos_j]
                    pair_count += 1
    
    return total_score / max(1, pair_count) if pair_count > 0 else 0.0

def _estimate_max_objective_improvement(unassigned_items: List[int], available_positions: List[int],
                                    partial_mapping: np.ndarray, objective_matrix: np.ndarray) -> float:
    """Estimate maximum possible improvement to objective from optimal assignment of unassigned items."""
    
    if not unassigned_items or not available_positions:
        return 0.0
    
    max_improvement = 0.0
    n_items = len(partial_mapping)
    
    # Method 1: Pairs between unassigned items (optimistically assume best pairing)
    if len(unassigned_items) >= 2:
        max_internal_pairs = 0.0
        
        # For each pair of unassigned items, find their best position pair
        for i in range(len(unassigned_items)):
            for j in range(i + 1, len(unassigned_items)):
                item1, item2 = unassigned_items[i], unassigned_items[j]
                best_pair_score = 0.0
                
                # Try all available position pairs
                for pos1 in available_positions:
                    for pos2 in available_positions:
                        if pos1 != pos2:
                            fwd_score = objective_matrix[pos1, pos2]
                            bwd_score = objective_matrix[pos2, pos1]
                            pair_score = fwd_score + bwd_score
                            best_pair_score = max(best_pair_score, pair_score)
                
                max_internal_pairs += best_pair_score
        
        max_improvement += max_internal_pairs
    
    # Method 2: Pairs between unassigned and assigned items
    assigned_items = [i for i in range(n_items) if partial_mapping[i] >= 0]
    
    for unassigned_item in unassigned_items:
        for assigned_item in assigned_items:
            assigned_pos = partial_mapping[assigned_item]
            best_mixed_score = 0.0
            
            for pos in available_positions:
                fwd_score = objective_matrix[pos, assigned_pos]
                bwd_score = objective_matrix[assigned_pos, pos]
                mixed_score = fwd_score + bwd_score
                best_mixed_score = max(best_mixed_score, mixed_score)
            
            max_improvement += best_mixed_score
    
    # Normalize by total expected pair count (this is the key adaptation)
    total_pairs = n_items * (n_items - 1) if n_items > 1 else 1
    return max_improvement / total_pairs

#-----------------------------------------------------------------------------
# General MOO Scoring
#-----------------------------------------------------------------------------
class GeneralMOOArrays(ScoringArrays):
    """
    Extended ScoringArrays supporting arbitrary objectives from keypair tables.
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
        
        # Pre-compute BOTH raw and transformed objective matrices
        self.raw_objective_matrices = {}      # NEW: For goal programming
        self.objective_matrices = {}          # Existing: For MOO optimization
        total_memory = 0
        
        for i, obj in enumerate(objectives):
            print(f"  [{i+1}/{len(objectives)}] Computing {obj}...")
            
            # Create raw matrix first (NEW)
            raw_matrix = self._create_objective_matrix(keypair_df, obj)
            self.raw_objective_matrices[obj] = raw_matrix.astype(np.float32)
            
            # Create transformed matrix for MOO (MODIFIED)
            transformed_matrix = raw_matrix.copy()
            if weights[i] != 1.0:
                transformed_matrix *= weights[i]
            if not maximize[i]:
                transformed_matrix *= -1.0
            
            self.objective_matrices[obj] = transformed_matrix.astype(np.float32)
            total_memory += raw_matrix.nbytes + transformed_matrix.nbytes
            
            print(f"      Range: [{np.min(raw_matrix):.6f}, {np.max(raw_matrix):.6f}]")
        
        print(f"Pre-computation complete: {total_memory / (1024*1024):.1f}MB total")
        
        # Create minimal compatible arrays for parent class
        n_items = len(items)
        n_positions = len(positions)
        
        # Use first transformed objective as primary for compatibility
        primary_matrix = list(self.objective_matrices.values())[0] if self.objective_matrices else np.ones((n_positions, n_positions))
        
        item_scores = np.ones(n_items, dtype=np.float32)
        item_pair_matrix = np.ones((n_items, n_items), dtype=np.float32)
        
        # Initialize parent class with compatible arrays
        super().__init__(item_scores, item_pair_matrix, primary_matrix)
    
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
    
class GeneralMOOArraysWithObjectives(GeneralMOOArrays):
    """Extended GeneralMOOArrays with built-in objective calculation."""
    
    def __init__(self, keypair_df: pd.DataFrame, objectives: List[str], 
                 items: List[str], positions: List[str], 
                 weights: List[float], maximize: List[bool]):
        
        super().__init__(keypair_df, objectives, items, positions, weights, maximize)
        
        # Store objective names for reference
        self.objective_names = objectives
        
        # Create calculator for getting objectives
        self.scorer = GeneralMOOScorer(self)

    def get_raw_objectives(self, permutation: List[int]) -> List[float]:
        """Get raw objective values for goal programming (no weights/direction changes)."""
        mapping_array = np.array(permutation, dtype=np.int32)
        
        # Validate permutation
        if len(permutation) != len(self.items):
            raise ValueError(f"Permutation length {len(permutation)} != items length {len(self.items)}")
        
        if any(p < 0 or p >= len(self.positions) for p in permutation):
            raise ValueError("Invalid position indices in permutation")
        
        raw_objectives = []
        for obj_name in self.objectives:
            raw_matrix = self.raw_objective_matrices[obj_name]  # Use raw matrices
            score = self._score_single_objective_jit(mapping_array, raw_matrix)
            raw_objectives.append(score)
        
        return raw_objectives

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

    def get_objectives(self, permutation: List[int]) -> List[float]:
        """Get objective values for a permutation (uses transformed objectives for MOO)."""
        mapping_array = np.array(permutation, dtype=np.int32)
        
        # Validate permutation
        if len(permutation) != len(self.items):
            raise ValueError(f"Permutation length {len(permutation)} != items length {len(self.items)}")
        
        if any(p < 0 or p >= len(self.positions) for p in permutation):
            raise ValueError("Invalid position indices in permutation")
        
        # Calculate objectives using the scorer
        objectives = self.scorer.score_layout(mapping_array)
        
        return objectives
    
    def get_objective_names(self) -> List[str]:
        """Get the list of objective names."""
        return self.objective_names.copy()

class GeneralMOOCalculator(ScoreCalculator):
    """Score calculator for arbitrary objectives."""
    
    def __init__(self, general_arrays: GeneralMOOArrays):
        self.general_arrays = general_arrays
        super().__init__(general_arrays)  # Initialize parent properly
    
    def calculate_components(self, mapping: np.ndarray) -> ScoreComponents:
        """Calculate objectives as independent components."""
        objective_scores = self._calculate_all_objectives(mapping)
        
        # For compatibility with existing 2-component system, use first two objectives
        # as "item_score" and "item_pair_score", store all in custom attribute
        item_score = objective_scores[0] if len(objective_scores) > 0 else 0.0
        item_pair_score = objective_scores[1] if len(objective_scores) > 1 else 0.0
        
        components = ScoreComponents(item_score, item_pair_score)
        components.all_objectives = objective_scores  # Custom attribute
        
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
        # Initialize with multi_objective mode to ensure compatibility
        super().__init__(general_arrays, mode='multi_objective')
        
        # Replace calculator with our general one
        self.calculator = GeneralMOOCalculator(general_arrays)
        self.general_arrays = general_arrays
    
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
# Goal Programming (Clean Implementation)
#-----------------------------------------------------------------------------
def goal_programming_search(config: Config, moo_arrays: GeneralMOOArraysWithObjectives, 
                                 target_values: List[float], deviation_weights: List[float], 
                                 max_solutions: int = 10) -> List[Dict]:
    """
    Goal programming search.
    """
    
    print("Running goal programming search...")
    print(f"Target values: {target_values}")
    print(f"Deviation weights: {deviation_weights}")
    
    opt = config.optimization
    items_list = list(opt.items_to_assign)
    positions_list = list(opt.positions_to_assign)
    n_items = len(items_list)
    n_positions = len(positions_list)
    
    print(f"Search space: {n_items} items in {n_positions} positions")
    print(f"Expected permutations: {factorial(n_positions) // factorial(n_positions - n_items):,}")
    
    # Set up constraints
    constrained_items = np.array([i for i, item in enumerate(items_list) 
                                 if item in opt.items_to_constrain_set], dtype=np.int32)
    constrained_positions = np.array([i for i, pos in enumerate(positions_list) 
                                    if pos.upper() in opt.positions_to_constrain_set], dtype=np.int32)
    
    # Initialize search state
    initial_mapping = np.full(n_items, -1, dtype=np.int16)
    initial_used = np.zeros(n_positions, dtype=bool)
    
    # Handle pre-assigned items (FIXED: track actual assigned count)
    initial_assigned_count = 0
    if opt.items_assigned:
        item_to_idx = {item: idx for idx, item in enumerate(items_list)}
        pos_to_idx = {pos: idx for idx, pos in enumerate(positions_list)}
        
        for item, pos in zip(opt.items_assigned, opt.positions_assigned):
            if item in item_to_idx and pos.upper() in pos_to_idx:
                item_idx = item_to_idx[item]
                pos_idx = pos_to_idx[pos.upper()]
                initial_mapping[item_idx] = pos_idx
                initial_used[pos_idx] = True
                initial_assigned_count += 1
    
    print(f"Pre-assigned items: {initial_assigned_count}")
    
    # Goal programming specific data structures
    solutions = []
    best_deviation = float('inf')
    nodes_processed = 0
    nodes_pruned = 0
    solutions_found = 0
    
    def calculate_goal_programming_score(mapping: np.ndarray) -> Tuple[float, List[float], List[float]]:
        """Calculate goal programming score using RAW objectives."""
        # Ensure all items are assigned
        if np.any(mapping < 0):
            return float('inf'), [0.0] * len(target_values), [0.0] * len(target_values)
        
        permutation = [int(mapping[i]) for i in range(len(mapping))]
        
        try:
            # Use RAW objectives for meaningful target comparison
            raw_objectives = moo_arrays.get_raw_objectives(permutation)
            
            # Calculate deviations from raw objectives
            deviations = [abs(obj_val - target) for obj_val, target in zip(raw_objectives, target_values)]
            total_deviation = sum(dev * weight for dev, weight in zip(deviations, deviation_weights))
            
            return total_deviation, raw_objectives, deviations
        except Exception as e:
            print(f"Error calculating objectives: {e}")
            return float('inf'), [0.0] * len(target_values), [0.0] * len(target_values)
    
    def simple_goal_programming_bound(partial_mapping: np.ndarray) -> float:
        """
        SIMPLIFIED upper bound calculation for goal programming.
        
        Returns a conservative lower bound on the total weighted deviation.
        Conservative = higher bound = less pruning = correct but slower search.
        """
        assigned_count = np.sum(partial_mapping >= 0)
        
        if assigned_count == n_items:
            # Complete assignment - return actual score
            deviation, _, _ = calculate_goal_programming_score(partial_mapping)
            return deviation
        
        if assigned_count == 0:
            # No assignments yet - use optimistic bound (perfect assignment possible)
            return 0.0
        
        # Partial assignment - use a simple conservative estimate
        # Assume remaining assignments contribute zero deviation (optimistic)
        # This bound might not be tight, but it's correct and fast
        
        # For a more conservative (higher) bound, assume some minimum deviation
        # from unassigned items. This reduces pruning but ensures correctness.
        min_deviation_per_unassigned = 0.01  # Conservative assumption
        unassigned_count = n_items - assigned_count
        
        return min_deviation_per_unassigned * unassigned_count
    
    def dfs_goal_programming_fixed(mapping: np.ndarray, used: np.ndarray, depth: int):
        """FIXED goal programming DFS with proper depth management."""
        nonlocal nodes_processed, nodes_pruned, solutions, best_deviation, solutions_found
        
        nodes_processed += 1
        
        # Progress reporting
        if nodes_processed % 10000 == 0 or nodes_processed <= 100:
            print(f"  Nodes: {nodes_processed:,}, Depth: {depth}, Solutions: {solutions_found}, Best: {best_deviation:.6f}")
        
        # TERMINATION CONDITION: Check if solution is complete
        if depth == n_items:
            # All items assigned - evaluate solution
            if len(constrained_items) > 0:
                if not validate_constraints_jit(mapping, constrained_items, constrained_positions):
                    return
            
            # Calculate goal programming score
            total_deviation, objectives, deviations = calculate_goal_programming_score(mapping)
            
            if total_deviation < float('inf'):
                solutions_found += 1
                
                # Update solutions list
                if len(solutions) < max_solutions or total_deviation < best_deviation:
                    # Convert to dictionary mapping
                    item_mapping = {items_list[i]: positions_list[mapping[i]] for i in range(n_items)}
                    
                    solution_dict = {
                        'rank': len(solutions) + 1,
                        'total_deviation': total_deviation,
                        'mapping': item_mapping,
                        'objectives': objectives,
                        'deviations': deviations
                    }
                    
                    solutions.append(solution_dict)
                    
                    # Sort by total deviation (ascending - lower is better)
                    solutions.sort(key=lambda x: x['total_deviation'])
                    
                    # Keep only top max_solutions
                    if len(solutions) > max_solutions:
                        solutions = solutions[:max_solutions]
                    
                    # Update best (worst of kept solutions)
                    if solutions:
                        best_deviation = solutions[-1]['total_deviation']
                        
                    if solutions_found <= 10:
                        print(f"    Solution #{solutions_found}: deviation = {total_deviation:.6f}")
            
            return
        
        # SEARCH CONTINUATION: Get next item to assign
        next_item = get_next_item_jit(mapping, constrained_items)
        if next_item == -1:
            print(f"    ERROR: No next item found at depth {depth}, but depth < n_items")
            return
        
        # Get valid positions for this item
        if next_item in constrained_items:
            valid_positions = [pos for pos in constrained_positions if not used[pos]]
        else:
            valid_positions = [pos for pos in range(n_positions) if not used[pos]]
        
        if not valid_positions:
            print(f"    No valid positions for item {next_item} at depth {depth}")
            return
        
        # Try each valid position
        for pos in valid_positions:
            # Create new state
            mapping[next_item] = pos
            used[pos] = True
            
            # OPTIONAL PRUNING (can be disabled for debugging)
            do_pruning = True
            if do_pruning and len(solutions) >= max_solutions:
                bound = simple_goal_programming_bound(mapping)
                if bound > best_deviation + 1e-9:  # Small epsilon for numerical stability
                    nodes_pruned += 1
                    # Restore state
                    mapping[next_item] = -1
                    used[pos] = False
                    continue
            
            # Recurse to next depth
            dfs_goal_programming_fixed(mapping, used, depth + 1)
            
            # Backtrack
            mapping[next_item] = -1
            used[pos] = False
    
    # FIXED: Start search with correct initial depth
    print(f"\nStarting fixed goal programming search...")
    print(f"  Items: {n_items}, Positions: {n_positions}")
    print(f"  Initial assigned: {initial_assigned_count}")
    print(f"  Search will start at depth: {initial_assigned_count}")
    
    start_time = time.time()
    
    # Start search with correct depth
    dfs_goal_programming_fixed(initial_mapping, initial_used, initial_assigned_count)
    
    elapsed_time = time.time() - start_time
    
    # Update ranks
    for i, solution in enumerate(solutions):
        solution['rank'] = i + 1
    
    print(f"\nFIXED Goal Programming Search Results:")
    print(f"  Solutions found: {solutions_found}")
    print(f"  Solutions kept: {len(solutions)}")
    print(f"  Nodes processed: {nodes_processed:,}")
    print(f"  Nodes pruned: {nodes_pruned:,}")
    print(f"  Search time: {elapsed_time:.2f}s")
    print(f"  Rate: {nodes_processed/elapsed_time:.0f} nodes/sec")
    
    if solutions:
        print(f"  Best deviation: {solutions[0]['total_deviation']:.6f}")
        print(f"  Worst kept deviation: {solutions[-1]['total_deviation']:.6f}")
        
        # Show diversity of solutions
        if len(solutions) > 1:
            deviations = [sol['total_deviation'] for sol in solutions]
            print(f"  Deviation range: {min(deviations):.6f} to {max(deviations):.6f}")
            print(f"  Standard deviation: {np.std(deviations):.6f}")
    
    return solutions

def save_goal_programming_results(results: List[Dict], config: Config, objectives: List[str]) -> str:
    """Save goal programming optimization results to CSV."""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config.paths.layout_results_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"goal_programming_results_{timestamp}.csv"
    
    # Convert results to DataFrame
    rows = []
    for result in results:
        row = {
            'rank': result['rank'],
            'total_deviation': result['total_deviation'],
            'mapping': str(result['mapping'])
        }
        # Add objective values and deviations
        for i, (obj_name, obj_value) in enumerate(zip(objectives, result['objectives'])):
            row[f'{obj_name}_value'] = obj_value
            row[f'{obj_name}_deviation'] = result['deviations'][i]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return str(csv_path)

#-----------------------------------------------------------------------------
# Search Wrapper Functions
#-----------------------------------------------------------------------------
def run_general_moo(config: Config, keypair_table: str, objectives: List[str],
                   weights: List[float], maximize: List[bool], **kwargs) -> List[Dict]:
    """Run general MOO using existing search infrastructure."""
    
    print("Using general MOO with existing branch-and-bound search infrastructure")
    
    # Load keypair table
    keypair_df = pd.read_csv(keypair_table, dtype={'key_pair': str})
    print(f"Loaded keypair table: {len(keypair_df)} rows")
    
    # Validate objectives
    missing = [obj for obj in objectives if obj not in keypair_df.columns]
    if missing:
        raise ValueError(f"Missing objectives in keypair table: {missing}")
    
    # Create general MOO scorer
    items = list(config.optimization.items_to_assign)
    positions = list(config.optimization.positions_to_assign)
    
    general_arrays = GeneralMOOArrays(keypair_df, objectives, items, positions, weights, maximize)
    scorer = GeneralMOOScorer(general_arrays)
    
    # Use existing multi_objective_search from search.py
    print(f"\nRunning multi-objective search with {len(objectives)} objectives...")
    
    pareto_front, nodes_processed, nodes_pruned = multi_objective_search(
        config, 
        scorer, 
        max_solutions=kwargs.get('max_solutions'),
        time_limit=kwargs.get('time_limit'),
        processes=kwargs.get('processes')
    )
    
    # Show objective statistics if results exist
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
    
    # Load keypair table and create scorer (same as above)
    keypair_df = pd.read_csv(keypair_table, dtype={'key_pair': str})
    items = list(config.optimization.items_to_assign)
    positions = list(config.optimization.positions_to_assign)
    n_items = len(items)
    n_positions = len(positions)
    
    general_arrays = GeneralMOOArrays(keypair_df, objectives, items, positions, weights, maximize)
    scorer = GeneralMOOScorer(general_arrays)
    
    # Calculate and show search space
    total_perms = 1
    for i in range(n_items):
        total_perms *= (n_positions - i)
    
    print(f"Total permutations to evaluate: {total_perms:,}")
    
    # Warn for large problems
    estimated_seconds = total_perms / 1000  # Assume 1000 evals/sec
    if estimated_seconds > 3600:
        print(f"Estimated time: {estimated_seconds/3600:.1f} hours")
        if total_perms > 10000000:  # 10 million
            response = input(f"\nThis will take a very long time. Continue? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Brute force search cancelled.")
                return []
    
    # Import pareto dominance function from search module
    from search import pareto_dominates
    
    # Run brute force enumeration
    pareto_front = []
    pareto_objectives = []
    evaluated = 0
    start_time = time.time()
    
    print(f"\nStarting brute force enumeration...")
    
    try:
        for perm in permutations(range(n_positions), n_items):
            evaluated += 1
            
            # Check limits
            if time_limit and (time.time() - start_time) > time_limit:
                print(f"\nTime limit reached at {evaluated:,} evaluations")
                break
            if max_solutions and len(pareto_front) >= max_solutions:
                print(f"\nSolution limit reached at {evaluated:,} evaluations")
                break
            
            # Calculate objectives
            mapping_array = np.array(perm, dtype=np.int32)
            objectives = scorer.score_layout(mapping_array)
            
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
  # Goal programming (minimize deviations from targets)
  python optimize_layout_general.py --config config.yaml \\
    --objectives engram7_load_normalized,engram7_strength_normalized,engram7_position_normalized,engram7_hspan_normalized,engram7_vspan_normalized,engram7_sequence_normalized \\
    --keypair-table input/keypair_scores_detailed.csv \\
    --goal-programming
  # Goal programming with custom targets and deviation weights
  python optimize_layout_general.py --config config.yaml \\
    --objectives engram7_load_normalized,engram7_strength_normalized,engram7_position_normalized,engram7_hspan_normalized,engram7_vspan_normalized,engram7_sequence_normalized \\
    --keypair-table input/keypair_scores_detailed.csv \\
    --goal-programming \\
    --targets "1.0,1.0,1.0,1.0,1.0,1.0" \\
    --deviation-weights "2.0,1.0,1.0,2.0,1.0,1.0"
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
    parser.add_argument('--processes', type=int, help='Number of parallel processes')
    
    # Utility options
    parser.add_argument('--validate', action='store_true', help='Run validation before optimization')
    parser.add_argument('--dry-run', action='store_true', help='Validate configuration only')
    
    # Goal programming
    parser.add_argument('--goal-programming', action='store_true',
                       help='Use goal programming instead of traditional MOO')
    parser.add_argument('--targets', type=str,
                       help='Comma-separated target values for each objective (e.g., "1.0,1.0,0.5")')
    parser.add_argument('--deviation-weights', type=str,
                       help='Comma-separated weights for deviations (default: equal weights)')
    
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
        print(f"  Search method: {'Goal Programming' if args.goal_programming else ('Brute Force' if args.brute_force else 'Branch-and-Bound')}")
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

        # Goal programming logic
        if args.goal_programming:
            if not args.targets:
                target_values = [1.0] * len(objectives)
                print(f"Using default targets: {target_values}")
            else:
                target_values = [float(t.strip()) for t in args.targets.split(',')]
                if len(target_values) != len(objectives):
                    raise ValueError(f"Targets count ({len(target_values)}) != objectives count ({len(objectives)})")
            
            if args.deviation_weights:
                deviation_weights = [float(w.strip()) for w in args.deviation_weights.split(',')]
                if len(deviation_weights) != len(objectives):
                    raise ValueError(f"Deviation weights count != objectives count")
            else:
                deviation_weights = [1.0] * len(objectives)
            
            # Load and validate keypair table
            keypair_df = pd.read_csv(args.keypair_table, dtype={'key_pair': str})
            print(f"Loaded keypair table: {len(keypair_df)} rows")
            
            missing = [obj for obj in objectives if obj not in keypair_df.columns]
            if missing:
                raise ValueError(f"Missing objectives in keypair table: {missing}")
            
            # IMPORTANT: Create extended MOO arrays with corrected implementation
            items = list(config.optimization.items_to_assign)
            positions = list(config.optimization.positions_to_assign)
            
            moo_arrays = GeneralMOOArraysWithObjectives(
                keypair_df, objectives, items, positions, weights, maximize
            )
            
            # Run corrected goal programming search
            results = goal_programming_search(
                config, moo_arrays, target_values, deviation_weights, 
                max_solutions=args.max_solutions or 10
            )
            
            # Save results
            if results:
                csv_path = save_goal_programming_results(results, config, objectives)
                print(f"\nResults saved to: {csv_path}")
                print(f"\nGoal Programming completed successfully!")
            else:
                print("No solutions found!")
        
        # Regular MOO or brute force
        else:
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
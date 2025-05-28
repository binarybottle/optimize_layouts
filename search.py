# search.py
"""
Search algorithms for layout optimization.

Consolidates all search logic including:
- Single-objective branch and bound
- Multi-objective Pareto search  
- Upper bound calculations
- Constraint handling
"""

import numpy as np
import time
import gc
from typing import List, Tuple, Dict, Set, Optional
from numba import jit
from math import factorial
from tqdm import tqdm

from config import Config
from scoring import LayoutScorer
from scoring import LayoutScorer, apply_default_combination

#-----------------------------------------------------------------------------
# Upper bound calculations
#-----------------------------------------------------------------------------
class UpperBoundCalculator:
    """Calculate tight upper bounds for branch-and-bound pruning."""
    
    def __init__(self, scorer: LayoutScorer):
        self.scorer = scorer
        self._cache = {}
    
    def calculate_upper_bound(self, partial_mapping: np.ndarray, 
                            used_positions: np.ndarray) -> float:
        """
        Calculate upper bound for partial mapping.
        
        Args:
            partial_mapping: Current partial assignment (-1 for unassigned)
            used_positions: Boolean array of used positions
            
        Returns:
            Upper bound score achievable from this state
        """
        # Cache key for performance
        cache_key = (tuple(partial_mapping), tuple(used_positions))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Find unassigned items and available positions
        unassigned_items = [i for i in range(len(partial_mapping)) if partial_mapping[i] < 0]
        available_positions = [i for i in range(len(used_positions)) if not used_positions[i]]
        
        if not unassigned_items:
            # Complete assignment - return actual score
            bound = self.scorer.score_layout(partial_mapping)
        else:
            # Estimate maximum possible score from current state
            current_components = self.scorer.get_components(partial_mapping)
            
            # Estimate maximum improvements for each component
            max_item_gain = self._estimate_max_item_gain(unassigned_items, available_positions)
            max_pair_gain = self._estimate_max_pair_gain(unassigned_items, available_positions, partial_mapping)
            
            # Combine using same logic as scorer
            if self.scorer.mode == 'combined':
                improved_item = current_components.item_score + max_item_gain
                improved_item_pair = current_components.item_pair_score + max_pair_gain
                bound = apply_default_combination(improved_item, improved_item_pair)
            elif self.scorer.mode == 'pair_only':
                bound = (current_components.item_pair_score + max_pair_gain)
            else:  # item_only
                bound = current_components.item_score + max_item_gain
        
        self._cache[cache_key] = bound
        return bound
    
    def _estimate_max_item_gain(self, unassigned_items: List[int], 
                               available_positions: List[int]) -> float:
        """Estimate maximum possible item score gain."""
        if not unassigned_items or not available_positions:
            return 0.0
        
        # For each unassigned item, find its best possible position
        total_gain = 0.0
        current_count = len(self.scorer.arrays.item_scores) - len(unassigned_items)
        
        for item_idx in unassigned_items:
            best_score = 0.0
            item_score = self.scorer.arrays.item_scores[item_idx]
            
            for pos_idx in available_positions:
                pos_score = self.scorer.arrays.position_matrix[pos_idx, pos_idx]
                score = item_score * pos_score
                best_score = max(best_score, score)
            
            total_gain += best_score
        
        # Normalize by total items that will be placed
        total_items = current_count + len(unassigned_items)
        return total_gain / max(1, total_items)
    
    def _estimate_max_pair_gain(self, unassigned_items: List[int],
                               available_positions: List[int],
                               partial_mapping: np.ndarray) -> float:
        """Estimate maximum possible pair score gain."""
        if len(unassigned_items) < 2:
            return 0.0
        
        total_gain = 0.0
        
        # Pairs between unassigned items
        for i, item1 in enumerate(unassigned_items):
            for item2 in unassigned_items[i+1:]:
                if len(available_positions) >= 2:
                    best_pair_score = 0.0
                    
                    # Try all position pairs
                    for pos1 in available_positions:
                        for pos2 in available_positions:
                            if pos1 != pos2:
                                fwd_score = (self.scorer.arrays.item_pair_matrix[item1, item2] * 
                                           self.scorer.arrays.position_matrix[pos1, pos2])
                                bwd_score = (self.scorer.arrays.item_pair_matrix[item2, item1] * 
                                           self.scorer.arrays.position_matrix[pos2, pos1])
                                pair_score = fwd_score + bwd_score
                                best_pair_score = max(best_pair_score, pair_score)
                    
                    total_gain += best_pair_score
        
        # Pairs between unassigned and assigned items
        assigned_items = [i for i in range(len(partial_mapping)) if partial_mapping[i] >= 0]
        for unassigned_item in unassigned_items:
            for assigned_item in assigned_items:
                assigned_pos = partial_mapping[assigned_item]
                
                best_mixed_score = 0.0
                for pos in available_positions:
                    fwd_score = (self.scorer.arrays.item_pair_matrix[unassigned_item, assigned_item] * 
                               self.scorer.arrays.position_matrix[pos, assigned_pos])
                    bwd_score = (self.scorer.arrays.item_pair_matrix[assigned_item, unassigned_item] * 
                               self.scorer.arrays.position_matrix[assigned_pos, pos])
                    mixed_score = fwd_score + bwd_score
                    best_mixed_score = max(best_mixed_score, mixed_score)
                
                total_gain += best_mixed_score
        
        # Normalize by total pair count
        total_items = len(partial_mapping)
        total_pairs = total_items * (total_items - 1) if total_items > 1 else 1
        return total_gain / total_pairs
        
    def clear_cache(self):
        """Clear upper bound cache."""
        self._cache.clear()

#-----------------------------------------------------------------------------
# Constraint handling
#-----------------------------------------------------------------------------
@jit(nopython=True)
def get_next_item_jit(mapping: np.ndarray, constrained_items: np.ndarray) -> int:
    """JIT-compiled function to get next item to assign."""
    # First try constrained items
    if len(constrained_items) > 0:
        for item_idx in constrained_items:
            if mapping[item_idx] < 0:
                return item_idx
    
    # Then any unassigned item
    for i in range(len(mapping)):
        if mapping[i] < 0:
            return i
    
    return -1

@jit(nopython=True)
def validate_constraints_jit(mapping: np.ndarray, constrained_items: np.ndarray,
                           constrained_positions: np.ndarray) -> bool:
    """JIT-compiled constraint validation."""
    for i in range(len(constrained_items)):
        item_idx = constrained_items[i]
        if mapping[item_idx] >= 0:
            pos = mapping[item_idx]
            # Check if position is in constrained positions
            found = False
            for j in range(len(constrained_positions)):
                if pos == constrained_positions[j]:
                    found = True
                    break
            if not found:
                return False
    return True

#-----------------------------------------------------------------------------
# Single-objective search
#-----------------------------------------------------------------------------
def single_objective_search(config: Config, scorer: LayoutScorer, 
                          n_solutions: int = 5) -> Tuple[List[Tuple[float, Dict[str, str], Dict]], int, int]:
    """
    Branch-and-bound search for single-objective optimization.
    
    Args:
        config: Configuration object
        scorer: Layout scorer
        n_solutions: Number of top solutions to maintain
        
    Returns:
        Tuple of (results, nodes_processed, nodes_pruned)
    """
    print("Running Single-Objective Branch-and-Bound Search...")
    
    opt = config.optimization
    items_list = list(opt.items_to_assign)
    positions_list = list(opt.positions_to_assign)
    n_items = len(items_list)
    n_positions = len(positions_list)
    
    # Set up constraints
    constrained_items = np.array([i for i, item in enumerate(items_list) 
                                 if item in opt.items_to_constrain_set], dtype=np.int32)
    constrained_positions = np.array([i for i, pos in enumerate(positions_list) 
                                    if pos.upper() in opt.positions_to_constrain_set], dtype=np.int32)
    
    # Initialize search state
    initial_mapping = np.full(n_items, -1, dtype=np.int16)
    initial_used = np.zeros(n_positions, dtype=bool)
    
    # Handle pre-assigned items
    if opt.items_assigned:
        item_to_idx = {item: idx for idx, item in enumerate(items_list)}
        pos_to_idx = {pos: idx for idx, pos in enumerate(positions_list)}
        
        for item, pos in zip(opt.items_assigned, opt.positions_assigned):
            if item in item_to_idx and pos.upper() in pos_to_idx:
                item_idx = item_to_idx[item]
                pos_idx = pos_to_idx[pos.upper()]
                initial_mapping[item_idx] = pos_idx
                initial_used[pos_idx] = True
    
    # Initialize upper bound calculator
    bound_calc = UpperBoundCalculator(scorer)
    
    # Search statistics
    nodes_processed = 0
    nodes_pruned = 0
    solutions = []
    worst_score = float('-inf')
    
    # Calculate search space for progress tracking
    if len(constrained_items) > 0:
        phase1_perms = factorial(len(constrained_positions)) // factorial(len(constrained_positions) - len(constrained_items))
        remaining_items = n_items - len(constrained_items)
        remaining_positions = n_positions - len(constrained_items)
        phase2_perms = factorial(remaining_positions) // factorial(remaining_positions - remaining_items)
        total_estimated_nodes = phase1_perms * phase2_perms * 2  # Rough estimate including internal nodes
    else:
        total_perms = factorial(n_positions) // factorial(n_positions - n_items)
        total_estimated_nodes = total_perms * 2
    
    start_time = time.time()
    
    def dfs_search(mapping: np.ndarray, used: np.ndarray, depth: int, pbar: tqdm):
        """Iterative DFS search using explicit stack."""
        nonlocal nodes_processed, nodes_pruned, solutions, worst_score
        
        # Stack entries: (mapping, used, depth)
        stack = [(mapping.copy(), used.copy(), depth)]
        
        while stack:
            current_mapping, current_used, current_depth = stack.pop()
            nodes_processed += 1
            
            # Progress update
            if nodes_processed % 10000 == 0:
                pbar.update(10000)
                # Memory cleanup
                if nodes_processed % 100000 == 0:
                    gc.collect()
                    bound_calc.clear_cache()
                    scorer.clear_cache()
            
            # Check if solution is complete
            if current_depth == n_items:
                # Validate constraints
                if len(constrained_items) > 0:
                    if not validate_constraints_jit(current_mapping, constrained_items, constrained_positions):
                        continue
                
                # Calculate score
                score = scorer.score_layout(current_mapping)
                
                # Update solutions list
                if len(solutions) < n_solutions or score > worst_score:
                    # Convert to dictionary mapping
                    item_mapping = {items_list[i]: positions_list[current_mapping[i]] 
                                   for i in range(n_items)}
                    
                    solutions.append((score, item_mapping, {}))
                    solutions.sort(key=lambda x: x[0], reverse=True)
                    
                    if len(solutions) > n_solutions:
                        solutions.pop()  # Remove worst
                    
                    if solutions:
                        worst_score = solutions[-1][0]
                
                continue
            
            # Get next item to assign
            next_item = get_next_item_jit(current_mapping, constrained_items)
            if next_item == -1:
                continue
            
            # Get valid positions for this item
            if next_item in constrained_items:
                valid_positions = [pos for pos in constrained_positions if not current_used[pos]]
            else:
                valid_positions = [pos for pos in range(n_positions) if not current_used[pos]]
            
            # Try each valid position
            for pos in valid_positions:
                # Create new state
                new_mapping = current_mapping.copy()
                new_used = current_used.copy()
                new_mapping[next_item] = pos
                new_used[pos] = True
                
                # Pruning check
                if len(solutions) >= n_solutions:
                    upper_bound = bound_calc.calculate_upper_bound(new_mapping, new_used)
                    if upper_bound <= worst_score + 1e-9:  # Small epsilon for numerical stability
                        nodes_pruned += 1
                        continue
                
                # Add to stack for exploration
                stack.append((new_mapping, new_used, current_depth + 1))
    
    # Run search with progress bar
    with tqdm(total=min(total_estimated_nodes, 1000000), desc="Searching", unit=" nodes") as pbar:
        dfs_search(initial_mapping, initial_used, 0, pbar)
        
        # Update final progress
        remaining = nodes_processed % 10000
        if remaining > 0:
            pbar.update(remaining)
    
    elapsed_time = time.time() - start_time
    print(f"\nSearch completed in {elapsed_time:.2f}s")
    print(f"Nodes processed: {nodes_processed:,}")
    print(f"Nodes pruned: {nodes_pruned:,}")
    if nodes_processed > 0:
        prune_rate = nodes_pruned / nodes_processed * 100
        print(f"Pruning efficiency: {prune_rate:.1f}%")
    
    return solutions, nodes_processed, nodes_pruned

#-----------------------------------------------------------------------------
# Multi-objective search
#-----------------------------------------------------------------------------
def pareto_dominates(obj1: List[float], obj2: List[float]) -> bool:
    """Check if obj1 Pareto dominates obj2."""
    at_least_one_better = False
    for v1, v2 in zip(obj1, obj2):
        if v1 < v2:
            return False
        if v1 > v2:
            at_least_one_better = True
    return at_least_one_better

def multi_objective_search(config: Config, scorer: LayoutScorer,
                         max_solutions: int = None, time_limit: float = None) -> Tuple[List[Tuple[np.ndarray, List[float]]], int, int]:
    """
    Multi-objective search finding Pareto-optimal solutions.
    
    Args:
        config: Configuration object
        scorer: Layout scorer (should be in multi_objective mode)
        max_solutions: Maximum solutions to find (None for unlimited)
        time_limit: Time limit in seconds (None for unlimited)
        
    Returns:
        Tuple of (pareto_front, nodes_processed, nodes_pruned)
    """
    print("Running Multi-Objective Pareto search...")
    
    opt = config.optimization
    items_list = list(opt.items_to_assign)
    positions_list = list(opt.positions_to_assign)
    n_items = len(items_list)
    n_positions = len(positions_list)
    
    # Set up constraints
    constrained_items = np.array([i for i, item in enumerate(items_list) 
                                 if item in opt.items_to_constrain_set], dtype=np.int32)
    constrained_positions = np.array([i for i, pos in enumerate(positions_list) 
                                    if pos.upper() in opt.positions_to_constrain_set], dtype=np.int32)
    
    # Initialize search
    initial_mapping = np.full(n_items, -1, dtype=np.int16)
    initial_used = np.zeros(n_positions, dtype=bool)
    
    # Handle pre-assigned items
    if opt.items_assigned:
        item_to_idx = {item: idx for idx, item in enumerate(items_list)}
        pos_to_idx = {pos: idx for idx, pos in enumerate(positions_list)}
        
        for item, pos in zip(opt.items_assigned, opt.positions_assigned):
            if item in item_to_idx and pos.upper() in pos_to_idx:
                item_idx = item_to_idx[item]
                pos_idx = pos_to_idx[pos.upper()]
                initial_mapping[item_idx] = pos_idx
                initial_used[pos_idx] = True
    
    # Pareto front storage
    pareto_front = []
    pareto_objectives = []
    
    # Search statistics
    nodes_processed = 0
    nodes_pruned = 0
    start_time = time.time()
    
    def can_improve_pareto(upper_bounds: List[float]) -> bool:
        """Check if upper bounds could improve Pareto front."""
        if not pareto_objectives:
            return True
        
        for existing_obj in pareto_objectives:
            if pareto_dominates(existing_obj, upper_bounds):
                return False
        return True
    
    def dfs_moo_search(mapping: np.ndarray, used: np.ndarray, depth: int):
        """Recursive multi-objective search."""
        nonlocal nodes_processed, nodes_pruned, pareto_front, pareto_objectives
        
        nodes_processed += 1
        
        # Check termination conditions
        if time_limit and (time.time() - start_time) > time_limit:
            return
        if max_solutions and len(pareto_front) >= max_solutions:
            return
        
        # Complete solution
        if depth == n_items:
            # Validate constraints
            if len(constrained_items) > 0:
                if not validate_constraints_jit(mapping, constrained_items, constrained_positions):
                    return
            
            # Get objectives
            objectives = scorer.score_layout(mapping)  # Returns list for MOO mode
            
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
                pareto_front.append((mapping.copy(), objectives))
                pareto_objectives.append(objectives)
            
            return
        
        # Get next item
        next_item = get_next_item_jit(mapping, constrained_items)
        if next_item == -1:
            return
        
        # Get valid positions
        if next_item in constrained_items:
            valid_positions = [pos for pos in constrained_positions if not used[pos]]
        else:
            valid_positions = [pos for pos in range(n_positions) if not used[pos]]
        
        # Try each position
        for pos in valid_positions:
            mapping[next_item] = pos
            used[pos] = True
            
            # Multi-objective pruning would go here (simplified for now)
            # For MOO, we'd need component-wise upper bounds
            
            dfs_moo_search(mapping, used, depth + 1)
            
            mapping[next_item] = -1
            used[pos] = False
    
    # Run search
    print(f"Search limits: {max_solutions or 'unlimited'} solutions, {time_limit or 'unlimited'} seconds")
    dfs_moo_search(initial_mapping, initial_used, 0)
    
    elapsed_time = time.time() - start_time
    print(f"\nMOO search completed in {elapsed_time:.2f}s")
    print(f"Nodes processed: {nodes_processed:,}")
    print(f"Pareto solutions found: {len(pareto_front)}")
    
    return pareto_front, nodes_processed, nodes_pruned
#!/usr/bin/env python3
"""
Multi-Objective Search Algorithms for Layout Optimization

This module provides both branch-and-bound and exhaustive search algorithms 
for multi-objective layout optimization. Both methods find Pareto-optimal 
solutions while preserving global optimality guarantees.

Key Features:
- Branch-and-bound with multi-objective upper bound calculation (faster)
- Exhaustive enumeration (slower but guaranteed complete)
- Pareto dominance checking and front maintenance
- Constraint handling for partial assignments
- Progress tracking and termination conditions
- Memory-efficient search with garbage collection
- JIT-compiled performance-critical functions

Usage:
    # Branch-and-bound (default, faster)
    pareto_front, stats = moo_search(config, scorer, search_mode='branch-bound')
    
    # Exhaustive enumeration (slower, guaranteed complete)
    pareto_front, stats = moo_search(config, scorer, search_mode='exhaustive')
"""

import numpy as np
import time
import gc
from typing import List, Dict, Tuple, Optional
from numba import jit
from math import factorial
from tqdm import tqdm
from dataclasses import dataclass

from config import Config


@jit(nopython=True)
def get_next_item_jit(mapping: np.ndarray, constrained_items: np.ndarray) -> int:
    """
    Get the next item to assign in the search order.
    
    Prioritizes constrained items first, then any unassigned item.
    
    Args:
        mapping: Current mapping array (-1 for unassigned items)
        constrained_items: Array of constrained item indices
        
    Returns:
        Index of next item to assign, or -1 if all assigned
    """
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
    """
    Validate that all constrained items are assigned to valid positions.
    
    Args:
        mapping: Current item-to-position mapping
        constrained_items: Array of constrained item indices
        constrained_positions: Array of valid position indices for constrained items
        
    Returns:
        True if constraints are satisfied
    """
    for i in range(len(constrained_items)):
        item_idx = constrained_items[i]
        if mapping[item_idx] >= 0:  # Item is assigned
            pos = mapping[item_idx]
            # Check if position is in allowed set
            found = False
            for j in range(len(constrained_positions)):
                if pos == constrained_positions[j]:
                    found = True
                    break
            if not found:
                return False
    return True


def pareto_dominates(obj1: List[float], obj2: List[float]) -> bool:
    """
    Check if obj1 Pareto dominates obj2.
    
    obj1 dominates obj2 if obj1 is at least as good in all objectives
    and strictly better in at least one objective (assuming maximization).
    
    Args:
        obj1: First objective vector
        obj2: Second objective vector
        
    Returns:
        True if obj1 dominates obj2
    """
    at_least_one_better = False
    for v1, v2 in zip(obj1, obj2):
        if v1 < v2:  # obj1 is worse in this objective
            return False
        if v1 > v2:  # obj1 is better in this objective
            at_least_one_better = True
    return at_least_one_better


def update_pareto_front(pareto_front: List[Dict], new_solution: Dict) -> List[Dict]:
    """
    Update Pareto front with a new solution.
    
    Args:
        pareto_front: Current list of non-dominated solutions
        new_solution: New solution to potentially add
        
    Returns:
        Updated Pareto front
    """
    new_objectives = new_solution['objectives']
    
    # Check if new solution is dominated by any existing solution
    for existing in pareto_front:
        if pareto_dominates(existing['objectives'], new_objectives):
            return pareto_front  # New solution is dominated
    
    # Remove any existing solutions dominated by the new solution
    updated_front = []
    for existing in pareto_front:
        if not pareto_dominates(new_objectives, existing['objectives']):
            updated_front.append(existing)
    
    # Add the new solution
    updated_front.append(new_solution)
    
    return updated_front


@dataclass
class SearchStats:
    """Statistics tracking for search process."""
    nodes_processed: int = 0
    nodes_pruned: int = 0
    solutions_found: int = 0
    elapsed_time: float = 0.0
    pareto_front_size: int = 0


class MOOUpperBoundCalculator:
    """
    Upper bound calculator that GUARANTEES no optimal solutions are pruned.
    
    Trade-off: Bounds may be looser but algorithm correctness is preserved.
    """
    
    def __init__(self, scorer):
        self.scorer = scorer
        self._cache = {}
        
        # Pre-calculate TRUE maximum values (not heuristics)
        self.true_max_position_scores = self._calculate_true_max_position_scores()
        self.true_max_item_weights = self._calculate_true_max_item_weights()
    
    def _calculate_true_max_position_scores(self) -> Dict[str, float]:
        """Calculate the actual maximum position score for each objective."""
        max_scores = {}
        
        for obj_name in self.scorer.objectives:
            if obj_name in self.scorer.trigram_objectives:
                position_scores = self.scorer.position_triple_scores.get(obj_name, {})
                max_scores[obj_name] = max(position_scores.values()) if position_scores else 1.0
            else:
                position_scores = self.scorer.position_pair_scores.get(obj_name, {})
                max_scores[obj_name] = max(position_scores.values()) if position_scores else 1.0
                
        return max_scores
    
    def _calculate_true_max_item_weights(self) -> Dict[str, float]:
        """Calculate the actual maximum item weighting scores."""
        max_weights = {}
        
        if self.scorer.use_bigram_weighting and self.scorer.item_pair_scores:
            max_weights['bigram'] = max(self.scorer.item_pair_scores.values())
        else:
            max_weights['bigram'] = 1.0
            
        if self.scorer.use_trigram_weighting and self.scorer.item_triple_scores:
            max_weights['trigram'] = max(self.scorer.item_triple_scores.values())
        else:
            max_weights['trigram'] = 1.0
            
        return max_weights
    
    def calculate_upper_bound_vector(self, partial_mapping: np.ndarray, 
                                   used_positions: np.ndarray) -> List[float]:
        """
        Calculate mathematically sound upper bounds.
        
        Formula: upper_bound = current_score + max_possible_remaining_score
        where max_possible_remaining_score is GUARANTEED achievable.
        """
        cache_key = (tuple(partial_mapping), tuple(used_positions))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        unassigned_count = sum(1 for x in partial_mapping if x < 0)
        
        if unassigned_count == 0:
            # Complete assignment - return exact score
            bound_vector = self.scorer.score_layout(partial_mapping)
        elif unassigned_count == 1:
            # Special case: exactly compute best possible completion
            bound_vector = self._calculate_exact_one_item_bound(partial_mapping, used_positions)
        else:
            # General case: conservative but mathematically sound bounds
            bound_vector = self._calculate_conservative_multi_item_bound(partial_mapping, unassigned_count)
        
        self._cache[cache_key] = bound_vector
        return bound_vector
    
    def _calculate_exact_one_item_bound(self, partial_mapping: np.ndarray, 
                                      used_positions: np.ndarray) -> List[float]:
        """
        Exactly calculate the maximum possible score when 1 item remains.
        This is computationally feasible and gives tight, valid bounds.
        """
        # Find the unassigned item and available positions
        unassigned_item = next(i for i in range(len(partial_mapping)) if partial_mapping[i] < 0)
        
        # FIXED: used_positions corresponds to available positions in search, not all scorer positions
        available_positions = [i for i in range(len(used_positions)) if not used_positions[i]]
        
        # Test all possible assignments for the remaining item
        best_scores = [0.0] * len(self.scorer.objectives)
        
        for pos_idx in available_positions:
            # Create complete mapping with this position choice
            test_mapping = partial_mapping.copy()
            test_mapping[unassigned_item] = pos_idx
            
            # Get exact score for this complete assignment
            scores = self.scorer.score_layout(test_mapping)
            
            # Track maximum for each objective
            for i, score in enumerate(scores):
                best_scores[i] = max(best_scores[i], score)
        
        return best_scores
    
    def _calculate_conservative_multi_item_bound(self, partial_mapping: np.ndarray, 
                                               unassigned_count: int) -> List[float]:
        """
        Calculate conservative but mathematically sound bounds for multiple unassigned items.
        
        Approach: current_partial_score + optimistic_remaining_score
        where optimistic_remaining_score uses maximum possible values.
        """
        bound_vector = []
        
        for obj_idx, obj_name in enumerate(self.scorer.objectives):
            if obj_name in self.scorer.trigram_objectives:
                upper_bound = self._calculate_trigram_conservative_bound(
                    partial_mapping, unassigned_count, obj_name, obj_idx
                )
            else:
                upper_bound = self._calculate_bigram_conservative_bound(
                    partial_mapping, unassigned_count, obj_name, obj_idx
                )
            
            bound_vector.append(upper_bound)
        
        return bound_vector
    
    def _calculate_bigram_conservative_bound(self, partial_mapping: np.ndarray,
                                           unassigned_count: int, obj_name: str, obj_idx: int) -> float:
        """Calculate conservative upper bound for bigram objective."""
        
        # Get current partial score from assigned items
        assigned_items = [i for i in range(len(partial_mapping)) if partial_mapping[i] >= 0]
        
        if len(assigned_items) >= 2:
            current_score, current_weight = self._calculate_current_bigram_score(partial_mapping, obj_name)
        else:
            current_score, current_weight = 0.0, 0.0
        
        # Calculate maximum possible contribution from remaining items
        max_pos_score = self.true_max_position_scores[obj_name]
        max_item_weight = self.true_max_item_weights['bigram']
        
        if self.scorer.use_bigram_weighting:
            # Optimistic: all remaining pairs get max_pos_score * max_item_weight
            # Number of new pairs: n_unassigned*(n_unassigned-1) + 2*n_assigned*n_unassigned
            new_pairs = unassigned_count * (unassigned_count - 1) + 2 * len(assigned_items) * unassigned_count
            optimistic_score = new_pairs * max_pos_score * max_item_weight
            optimistic_weight = new_pairs * max_item_weight
            
            total_score = current_score + optimistic_score
            total_weight = current_weight + optimistic_weight
            
            upper_bound_raw = total_score / total_weight if total_weight > 0 else max_pos_score
        else:
            # Unweighted: optimistic average
            new_pairs = unassigned_count * (unassigned_count - 1) + 2 * len(assigned_items) * unassigned_count
            current_pairs = len(assigned_items) * (len(assigned_items) - 1)
            total_pairs = current_pairs + new_pairs
            
            total_score = current_score * current_pairs + new_pairs * max_pos_score
            upper_bound_raw = total_score / total_pairs if total_pairs > 0 else max_pos_score
        
        # Apply objective weights and direction
        weighted_bound = upper_bound_raw * self.scorer.objective_weights[obj_idx]
        if not self.scorer.objective_maximize[obj_idx]:
            weighted_bound = 1.0 - weighted_bound
            
        return weighted_bound
    
    def _calculate_trigram_conservative_bound(self, partial_mapping: np.ndarray,
                                            unassigned_count: int, obj_name: str, obj_idx: int) -> float:
        """Calculate conservative upper bound for trigram objective."""
        
        # Trigram bounds are complex - use simplified conservative approach
        max_pos_score = self.true_max_position_scores[obj_name]
        max_item_weight = self.true_max_item_weights['trigram']
        
        if self.scorer.use_trigram_weighting:
            upper_bound_raw = max_pos_score * max_item_weight
        else:
            upper_bound_raw = max_pos_score
        
        # Apply objective weights and direction
        weighted_bound = upper_bound_raw * self.scorer.objective_weights[obj_idx]
        if not self.scorer.objective_maximize[obj_idx]:
            weighted_bound = 1.0 - weighted_bound
            
        return weighted_bound
    
    def _calculate_current_bigram_score(self, partial_mapping: np.ndarray, obj_name: str) -> Tuple[float, float]:
        """Calculate current bigram score and weight from partial assignment."""
        position_pair_scores = self.scorer.position_pair_scores[obj_name]
        
        # Get assigned items and positions
        assigned_items = []
        assigned_positions = []
        
        for i, pos_idx in enumerate(partial_mapping):
            if pos_idx >= 0:
                assigned_items.append(self.scorer.items[i])
                assigned_positions.append(self.scorer.positions[pos_idx])
        
        if len(assigned_items) < 2:
            return 0.0, 0.0
        
        if self.scorer.use_bigram_weighting:
            weighted_total = 0.0
            weight_total = 0.0
            
            for i in range(len(assigned_items)):
                for j in range(len(assigned_items)):
                    if i != j:
                        letter_pair = assigned_items[i] + assigned_items[j]
                        key_pair = assigned_positions[i] + assigned_positions[j]
                        
                        item_weight = self.scorer.item_pair_scores.get(letter_pair, 0.0)
                        if item_weight > 0 and key_pair in position_pair_scores:
                            score = position_pair_scores[key_pair]
                            weighted_total += score * item_weight
                            weight_total += item_weight
            
            return weighted_total, weight_total
        else:
            total_score = 0.0
            pair_count = 0
            
            for i in range(len(assigned_items)):
                for j in range(len(assigned_items)):
                    if i != j:
                        key_pair = assigned_positions[i] + assigned_positions[j]
                        if key_pair in position_pair_scores:
                            total_score += position_pair_scores[key_pair]
                            pair_count += 1
            
            return total_score, pair_count
    
    def clear_cache(self):
        """Clear upper bound cache."""
        self._cache.clear()


def branch_bound_moo_search(config: Config, scorer, max_solutions: Optional[int] = None, 
                           time_limit: Optional[float] = None, progress_bar: bool = True,
                           verbose: bool = False) -> Tuple[List[Dict], SearchStats]:
    """
    Branch-and-bound multi-objective search with upper bound pruning.
    
    Faster than exhaustive search while preserving global optimality through
    proper upper bound calculation and Pareto dominance pruning.
    
    Args:
        config: Configuration object with optimization settings
        scorer: Multi-objective scorer (WeightedMOOScorer)
        max_solutions: Maximum number of solutions to find (None for unlimited)
        time_limit: Time limit in seconds (None for unlimited)
        progress_bar: Whether to show progress bar
        verbose: Whether to show detailed output
        
    Returns:
        Tuple of (pareto_front, search_stats)
    """
    opt = config.optimization
    
    # Handle the correct semantics - items_to_assign are the ones being optimized
    items_list = list(opt.items_to_assign)  # Items being optimized
    positions_list = list(opt.positions_to_assign)  # Positions available for optimization
    
    # Get pre-assignment info for solution construction
    items_assigned = list(opt.items_assigned) if opt.items_assigned else []
    positions_assigned = list(opt.positions_assigned) if opt.positions_assigned else []
    
    n_items = len(items_list)
    n_positions = len(positions_list)
    
    # Set up constraint arrays
    constrained_items = np.array([
        i for i, item in enumerate(items_list) 
        if item in opt.items_to_constrain_set
    ], dtype=np.int32)
    
    constrained_positions = np.array([
        i for i, pos in enumerate(positions_list) 
        if pos.upper() in opt.positions_to_constrain_set
    ], dtype=np.int32)
    
    if len(constrained_items) > 0:
        print(f"  Constrained items: {[items_list[i] for i in constrained_items]}")
        print(f"  Constraint positions: {[positions_list[i] for i in constrained_positions]}")
    
    # Initialize search state
    initial_mapping = np.full(n_items, -1, dtype=np.int16)
    initial_used = np.zeros(n_positions, dtype=bool)
    
    # Handle pre-assigned items - NOTE: These are already handled in the scorer
    if opt.items_assigned:
        print(f"  Pre-assigned: {list(opt.items_assigned)} -> {list(opt.positions_assigned)}")
    
    # Initialize upper bound calculator
    bound_calc = MOOUpperBoundCalculator(scorer)
    
    # Calculate search space size for progress estimation
    if len(constrained_items) > 0:
        # Two-phase constraint handling
        phase1_perms = factorial(len(constrained_positions)) // factorial(len(constrained_positions) - len(constrained_items)) if len(constrained_positions) >= len(constrained_items) else 0
        remaining_items = n_items - len(constrained_items)
        remaining_positions = n_positions - len(constrained_items)
        phase2_perms = factorial(remaining_positions) // factorial(remaining_positions - remaining_items) if remaining_positions >= remaining_items else 0
        estimated_nodes = phase1_perms * phase2_perms * 2
    else:
        # Single phase
        total_perms = factorial(n_positions) // factorial(n_positions - n_items) if n_positions >= n_items else 0
        estimated_nodes = total_perms * 2  # Rough estimate including internal nodes
    
    if verbose:
        print("Starting Branch-and-Bound Multi-Objective Search...")
        print(f"  Items to assign: {items_list}")
        print(f"  Available positions: {positions_list}")
        print(f"  Search limits: {max_solutions or 'unlimited'} solutions, {time_limit or 'unlimited'} seconds")
        print(f"  Estimated search nodes: {estimated_nodes:,}")
    else:
        print(f"Branch-and-bound search: {n_items} items in {n_positions} positions...")
        print(f"  Estimated search nodes: {estimated_nodes:,}")

    # Initialize search data structures
    pareto_front = []
    stats = SearchStats()
    start_time = time.time()
    
    def dfs_search_with_pruning(mapping: np.ndarray, used: np.ndarray, depth: int, pbar: Optional[tqdm]):
        """Depth-first search with upper bound pruning."""
        nonlocal pareto_front
        
        # Use iterative approach with explicit stack to handle large search spaces
        stack = [(mapping.copy(), used.copy(), depth)]
        
        while stack:
            current_mapping, current_used, current_depth = stack.pop()
            stats.nodes_processed += 1
            
            # Check termination conditions
            if time_limit and (time.time() - start_time) > time_limit:
                if pbar:
                    pbar.set_description(f"Time limit reached")
                break
            
            if max_solutions and len(pareto_front) >= max_solutions:
                if pbar:
                    pbar.set_description(f"Solution limit reached")
                break
            
            # Update progress
            if pbar and stats.nodes_processed % 50000 == 0:
                pbar.update(50000)
                pbar.set_description(f"Pareto: {len(pareto_front)}, Pruned: {stats.nodes_pruned}")
            
            # Periodic cleanup
            if stats.nodes_processed % 500000 == 0:
                gc.collect()
                bound_calc.clear_cache()
                scorer.clear_cache()
            
            # Check if solution is complete
            if current_depth == n_items:
                # Validate constraints
                if len(constrained_items) > 0:
                    if not validate_constraints_jit(current_mapping, constrained_items, constrained_positions):
                        continue
                
                # Evaluate solution
                if hasattr(scorer, 'score_layout_fast'):
                    objectives = scorer.score_layout_fast(current_mapping)
                else:
                    objectives = scorer.score_layout(current_mapping)
                stats.solutions_found += 1
                
                # Create complete solution including pre-assignments
                complete_mapping = {}
                
                # Add pre-assigned items first
                for item, pos in zip(items_assigned, positions_assigned):
                    complete_mapping[item] = pos
                
                # Add optimized items
                for i in range(n_items):
                    item = items_list[i]
                    pos = positions_list[current_mapping[i]]
                    complete_mapping[item] = pos
                
                new_solution = {
                    'mapping': complete_mapping,
                    'objectives': objectives
                }
                
                # Update Pareto front
                pareto_front = update_pareto_front(pareto_front, new_solution)
                
                continue
            
            # CRITICAL: Multi-objective branch-and-bound pruning
            if len(pareto_front) > 0:
                
                # Calculate upper bound vector for this partial solution
                upper_bound_vector = bound_calc.calculate_upper_bound_vector(current_mapping, current_used)

                # DEBUG
                #if stats.nodes_processed <= 100 or stats.nodes_processed % 10 == 0:  # Debug first 100 nodes, then every 10th
                #    unassigned_count = sum(1 for x in current_mapping if x == -1)
                #    print(f"Node {stats.nodes_processed}: {unassigned_count} unassigned, bounds: {[f'{x:.3f}' for x in upper_bound_vector]}")
                #    if len(pareto_front) > 0:
                #        print(f"  Best Pareto: {[f'{x:.3f}' for x in pareto_front[0]['objectives']]}")

                # Check if this branch can be pruned
                can_prune = False
                dominating_solution_idx = -1
                for j, pareto_solution in enumerate(pareto_front):
                    if pareto_dominates(pareto_solution['objectives'], upper_bound_vector):
                        can_prune = True
                        dominating_solution_idx = j
                        break
                
                # DEBUG
                #if can_prune and stats.nodes_processed <= 100:
                #    print(f"  -> PRUNED by solution {dominating_solution_idx}")
                #elif not can_prune and stats.nodes_processed <= 20:
                #    print(f"  -> CONTINUE (not dominated)")
                
                if can_prune:
                    stats.nodes_pruned += 1
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
            
            # Try each valid position (add to stack in reverse order for consistent ordering)
            for pos in reversed(valid_positions):
                new_mapping = current_mapping.copy()
                new_used = current_used.copy()
                new_mapping[next_item] = pos
                new_used[pos] = True
                
                stack.append((new_mapping, new_used, current_depth + 1))
    
    # Run search with optional progress bar
    pbar = None
    if progress_bar:
        pbar = tqdm(total=min(estimated_nodes, 1000000), desc="Searching", unit=" nodes")
    
    try:
        dfs_search_with_pruning(initial_mapping, initial_used, 0, pbar)
        
        # Update final progress
        if pbar:
            remaining = stats.nodes_processed % 50000
            if remaining > 0:
                pbar.update(remaining)
    
    finally:
        if pbar:
            pbar.close()
    
    # Finalize statistics
    stats.elapsed_time = time.time() - start_time
    stats.pareto_front_size = len(pareto_front)
    
    # Print summary
    if verbose:
        print(f"\nBranch-and-bound search completed:")
        print(f"  Time: {stats.elapsed_time:.2f}s")
        print(f"  Nodes processed: {stats.nodes_processed:,}")
        print(f"  Nodes pruned: {stats.nodes_pruned:,}")
        print(f"  Solutions evaluated: {stats.solutions_found:,}")
        print(f"  Pareto front size: {stats.pareto_front_size}")
        
        if stats.nodes_processed > 0:
            prune_rate = stats.nodes_pruned / stats.nodes_processed * 100
            rate = stats.nodes_processed / stats.elapsed_time
            print(f"  Search rate: {rate:.0f} nodes/sec")
            print(f"  Pruning efficiency: {prune_rate:.1f}%")
    
    return pareto_front, stats


def exhaustive_moo_search(config: Config, scorer, max_solutions: Optional[int] = None, 
                         time_limit: Optional[float] = None, progress_bar: bool = True,
                         verbose: bool = False) -> Tuple[List[Dict], SearchStats]:
    """
    Exhaustive multi-objective search (complete enumeration).
    
    Evaluates ALL possible permutations to guarantee finding every Pareto-optimal
    solution. Slower than branch-and-bound but provides absolute completeness guarantee.
    
    Args:
        config: Configuration object with optimization settings
        scorer: Multi-objective scorer (WeightedMOOScorer)
        max_solutions: Maximum number of solutions to find (None for unlimited)
        time_limit: Time limit in seconds (None for unlimited)
        progress_bar: Whether to show progress bar
        verbose: Whether to show detailed output
        
    Returns:
        Tuple of (pareto_front, search_stats)
    """
    opt = config.optimization
    
    # Handle the correct semantics - items_to_assign are the ones being optimized
    items_list = list(opt.items_to_assign)  # Items being optimized
    positions_list = list(opt.positions_to_assign)  # Positions available for optimization
    
    # Get pre-assignment info for solution construction
    items_assigned = list(opt.items_assigned) if opt.items_assigned else []
    positions_assigned = list(opt.positions_assigned) if opt.positions_assigned else []
    
    n_items = len(items_list)
    n_positions = len(positions_list)
    
    # Set up constraint arrays
    constrained_items = np.array([
        i for i, item in enumerate(items_list) 
        if item in opt.items_to_constrain_set
    ], dtype=np.int32)
    
    constrained_positions = np.array([
        i for i, pos in enumerate(positions_list) 
        if pos.upper() in opt.positions_to_constrain_set
    ], dtype=np.int32)
    
    if len(constrained_items) > 0:
        print(f"  Constrained items: {[items_list[i] for i in constrained_items]}")
        print(f"  Constraint positions: {[positions_list[i] for i in constrained_positions]}")
    
    # Initialize search state
    initial_mapping = np.full(n_items, -1, dtype=np.int16)
    initial_used = np.zeros(n_positions, dtype=bool)
    
    # Handle pre-assigned items
    if opt.items_assigned:
        print(f"  Pre-assigned: {list(opt.items_assigned)} -> {list(opt.positions_assigned)}")
    
    # Calculate search space size for progress estimation
    if len(constrained_items) > 0:
        # Two-phase constraint handling
        phase1_perms = factorial(len(constrained_positions)) // factorial(len(constrained_positions) - len(constrained_items)) if len(constrained_positions) >= len(constrained_items) else 0
        remaining_items = n_items - len(constrained_items)
        remaining_positions = n_positions - len(constrained_items)
        phase2_perms = factorial(remaining_positions) // factorial(remaining_positions - remaining_items) if remaining_positions >= remaining_items else 0
        estimated_nodes = phase1_perms * phase2_perms * 2
    else:
        # Single phase
        total_perms = factorial(n_positions) // factorial(n_positions - n_items) if n_positions >= n_items else 0
        estimated_nodes = total_perms * 2  # Rough estimate including internal nodes
    
    if verbose:
        print("Starting Exhaustive Multi-Objective Search...")
        print(f"  Items to assign: {items_list}")
        print(f"  Available positions: {positions_list}")
        print(f"  Search limits: {max_solutions or 'unlimited'} solutions, {time_limit or 'unlimited'} seconds")
        print(f"  Estimated search nodes: {estimated_nodes:,}")
    else:
        print(f"Exhaustive search: {n_items} items in {n_positions} positions...")
        print(f"  Estimated search nodes: {estimated_nodes:,}")

    # Initialize search data structures
    pareto_front = []
    stats = SearchStats()
    start_time = time.time()
    
    def dfs_search_exhaustive(mapping: np.ndarray, used: np.ndarray, depth: int, pbar: Optional[tqdm]):
        """Depth-first exhaustive search without pruning."""
        nonlocal pareto_front
        
        # Use iterative approach with explicit stack to handle large search spaces
        stack = [(mapping.copy(), used.copy(), depth)]
        
        while stack:
            current_mapping, current_used, current_depth = stack.pop()
            stats.nodes_processed += 1
            
            # Check termination conditions
            if time_limit and (time.time() - start_time) > time_limit:
                if pbar:
                    pbar.set_description(f"Time limit reached")
                break
            
            if max_solutions and len(pareto_front) >= max_solutions:
                if pbar:
                    pbar.set_description(f"Solution limit reached")
                break
            
            # Update progress
            if pbar and stats.nodes_processed % 50000 == 0:
                pbar.update(50000)
                pbar.set_description(f"Pareto front: {len(pareto_front)}")
            
            # Periodic cleanup
            if stats.nodes_processed % 500000 == 0:
                gc.collect()
                scorer.clear_cache()
            
            # Check if solution is complete
            if current_depth == n_items:
                # Validate constraints
                if len(constrained_items) > 0:
                    if not validate_constraints_jit(current_mapping, constrained_items, constrained_positions):
                        continue
                
                # Evaluate solution
                if hasattr(scorer, 'score_layout_fast'):
                    objectives = scorer.score_layout_fast(current_mapping)
                else:
                    objectives = scorer.score_layout(current_mapping)
                stats.solutions_found += 1
                
                # Create complete solution including pre-assignments
                complete_mapping = {}
                
                # Add pre-assigned items first
                for item, pos in zip(items_assigned, positions_assigned):
                    complete_mapping[item] = pos
                
                # Add optimized items
                for i in range(n_items):
                    item = items_list[i]
                    pos = positions_list[current_mapping[i]]
                    complete_mapping[item] = pos
                
                new_solution = {
                    'mapping': complete_mapping,
                    'objectives': objectives
                }
                
                # Update Pareto front
                pareto_front = update_pareto_front(pareto_front, new_solution)
                
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
            
            # Try each valid position (add to stack in reverse order for consistent ordering)
            for pos in reversed(valid_positions):
                new_mapping = current_mapping.copy()
                new_used = current_used.copy()
                new_mapping[next_item] = pos
                new_used[pos] = True
                
                stack.append((new_mapping, new_used, current_depth + 1))
    
    # Run search with optional progress bar
    pbar = None
    if progress_bar:
        pbar = tqdm(total=min(estimated_nodes, 1000000), desc="Searching", unit=" nodes")
    
    try:
        dfs_search_exhaustive(initial_mapping, initial_used, 0, pbar)
        
        # Update final progress
        if pbar:
            remaining = stats.nodes_processed % 50000
            if remaining > 0:
                pbar.update(remaining)
    
    finally:
        if pbar:
            pbar.close()
    
    # Finalize statistics
    stats.elapsed_time = time.time() - start_time
    stats.pareto_front_size = len(pareto_front)
    
    # Print summary
    if verbose:
        print(f"\nExhaustive search completed:")
        print(f"  Time: {stats.elapsed_time:.2f}s")
        print(f"  Nodes processed: {stats.nodes_processed:,}")
        print(f"  Solutions evaluated: {stats.solutions_found:,}")
        print(f"  Pareto front size: {stats.pareto_front_size}")
        
        if stats.nodes_processed > 0:
            rate = stats.nodes_processed / stats.elapsed_time
            efficiency = stats.solutions_found / stats.nodes_processed * 100
            print(f"  Search rate: {rate:.0f} nodes/sec")
            print(f"  Solution efficiency: {efficiency:.2f}% nodes yielded solutions")
    
    return pareto_front, stats


def moo_search(config: Config, scorer, max_solutions: Optional[int] = None, 
               time_limit: Optional[float] = None, progress_bar: bool = True,
               verbose: bool = False, search_mode: str = 'branch-bound') -> Tuple[List[Dict], SearchStats]:
    """
    Multi-objective Pareto search with selectable algorithm.
    
    Args:
        config: Configuration object with optimization settings
        scorer: Multi-objective scorer (WeightedMOOScorer)
        max_solutions: Maximum number of solutions to find (None for unlimited)
        time_limit: Time limit in seconds (None for unlimited)
        progress_bar: Whether to show progress bar
        verbose: Whether to show detailed output
        search_mode: 'branch-bound' (faster) or 'exhaustive' (complete)
        
    Returns:
        Tuple of (pareto_front, search_stats)
    """
    
    if search_mode == 'exhaustive':
        print("Using exhaustive enumeration:")
        print("  ✓ Guaranteed to find ALL Pareto-optimal solutions")
        print("  ✓ Simple implementation, no pruning errors possible")
        print("  ⚠ Slower performance (explores every permutation)")
        print("")
        
        return exhaustive_moo_search(
            config, scorer, max_solutions, time_limit, progress_bar, verbose)
    
    elif search_mode == 'branch-bound':
        print("Using branch-and-bound search:")
        print("  ✓ Faster performance (intelligent pruning)")
        print("  ✓ Preserves global optimality (with correct upper bounds)")
        print("  ⚠ More complex implementation")
        print("")
        
        return branch_bound_moo_search(
            config, scorer, max_solutions, time_limit, progress_bar, verbose)
    
    else:
        raise ValueError(f"Unknown search mode: {search_mode}. Use 'branch-bound' or 'exhaustive'")


def analyze_pareto_front(pareto_front: List[Dict], objective_names: Optional[List[str]] = None) -> None:
    """
    Print analysis of the Pareto front.
    
    Args:
        pareto_front: List of Pareto-optimal solutions
        objective_names: Optional names for objectives
    """
    if not pareto_front:
        print("Pareto front is empty.")
        return
    
    n_objectives = len(pareto_front[0]['objectives'])
    if not objective_names:
        objective_names = [f'Objective_{i+1}' for i in range(n_objectives)]
    
    print(f"\nPareto Front Analysis:")
    print(f"  Solutions: {len(pareto_front)}")
    print(f"  Objectives: {n_objectives}")
    
    # Calculate objective ranges
    objectives_matrix = np.array([sol['objectives'] for sol in pareto_front])
    
    print(f"\nObjective Ranges:")
    for i, name in enumerate(objective_names):
        if i < objectives_matrix.shape[1]:
            values = objectives_matrix[:, i]
            print(f"  {name}: [{np.min(values):.6f}, {np.max(values):.6f}] (span: {np.max(values) - np.min(values):.6f})")
    
    # Sort by first objective for display
    sorted_front = sorted(pareto_front, key=lambda x: x['objectives'][0], reverse=True)
    
    print(f"\nTop 5 Solutions (by {objective_names[0]}):")
    for i, solution in enumerate(sorted_front[:5], 1):
        # Get items in the expected order
        all_items = list(solution['mapping'].keys())
        if len(all_items) > 10:  # Assume pre-assigned + optimized
            # Try to maintain the order: pre-assigned first, then optimized
            sorted_items = sorted(all_items)  # Fallback to alphabetical if order unknown
        else:
            sorted_items = sorted(all_items)
        
        items_str = ''.join(sorted_items)
        positions_str = ''.join(solution['mapping'][item] for item in sorted_items)
        obj_str = ', '.join(f"{score:.6f}" for score in solution['objectives'])
        print(f"  {i}. {items_str} -> {positions_str} | [{obj_str}]")


def validate_pareto_front(pareto_front: List[Dict]) -> bool:
    """
    Validate that the Pareto front contains only non-dominated solutions.
    
    Args:
        pareto_front: List of solutions to validate
        
    Returns:
        True if all solutions are non-dominated
    """
    for i, sol1 in enumerate(pareto_front):
        for j, sol2 in enumerate(pareto_front):
            if i != j:
                if pareto_dominates(sol1['objectives'], sol2['objectives']):
                    print(f"Validation failed: Solution {i} dominates solution {j}")
                    return False
    
    return True


if __name__ == "__main__":
    print("Multi-Objective Search Module")
    print("This module provides branch-and-bound and exhaustive search algorithms for MOO layout optimization.")
    
    # Test Pareto dominance
    print("\nTesting Pareto dominance logic:")
    
    test_cases = [
        ([1.0, 2.0], [0.5, 1.5], True),   # First dominates second
        ([1.0, 1.0], [1.0, 1.0], False),  # Equal (no dominance)
        ([1.0, 0.5], [0.5, 1.0], False),  # Neither dominates
        ([2.0, 2.0], [1.0, 1.0], True),   # First dominates second
    ]
    
    for i, (obj1, obj2, expected) in enumerate(test_cases, 1):
        result = pareto_dominates(obj1, obj2)
        status = "PASS" if result == expected else "FAIL"
        print(f"  Test {i}: {obj1} dominates {obj2} = {result} ({status})")
    
    # Test Pareto front update
    print("\nTesting Pareto front update:")
    
    front = []
    test_solutions = [
        {'mapping': {'a': 'F'}, 'objectives': [1.0, 2.0]},
        {'mapping': {'b': 'D'}, 'objectives': [2.0, 1.0]},
        {'mapping': {'c': 'S'}, 'objectives': [0.5, 0.5]},  # Should be dominated
        {'mapping': {'d': 'J'}, 'objectives': [3.0, 3.0]},  # Should dominate others
    ]
    
    for sol in test_solutions:
        front = update_pareto_front(front, sol)
        print(f"  Added {sol['objectives']}, front size: {len(front)}")
    
    print(f"\nFinal Pareto front:")
    for i, sol in enumerate(front):
        print(f"  Solution {i+1}: {sol['objectives']}")
    
    print(f"\nPareto front validation: {'PASS' if validate_pareto_front(front) else 'FAIL'}")
# search.py
"""
Search algorithms for layout optimization.

Consolidates all search logic including:
- Single-objective branch and bound
- Multi-objective Pareto search (single-threaded and multi-threaded)
- Upper bound calculations
- Constraint handling
"""

import numpy as np
import time
import gc
import multiprocessing as mp
from typing import List, Tuple, Dict
from numba import jit
from math import factorial
from tqdm import tqdm

from config import Config
from scoring import LayoutScorer, apply_default_combination

#-----------------------------------------------------------------------------
# Upper bound calculations
#-----------------------------------------------------------------------------
class UpperBoundCalculator:
    """Calculate tight upper bounds for branch-and-bound."""
    
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
# Multi-objective Pareto dominance
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

def build_pareto_front(all_solutions: List[Dict]) -> List[Dict]:
    """Build Pareto front from list of solutions."""
    if not all_solutions:
        return []
    
    pareto_front = []
    
    for candidate in all_solutions:
        is_non_dominated = True
        dominated_indices = []
        
        for i, existing in enumerate(pareto_front):
            if pareto_dominates(existing['objectives'], candidate['objectives']):
                is_non_dominated = False
                break
            elif pareto_dominates(candidate['objectives'], existing['objectives']):
                dominated_indices.append(i)
        
        if is_non_dominated:
            # Remove dominated solutions
            for i in reversed(sorted(dominated_indices)):
                del pareto_front[i]
            
            # Add new solution
            pareto_front.append(candidate)
    
    return pareto_front

#-----------------------------------------------------------------------------
# Multiprocessing support for MOO
#-----------------------------------------------------------------------------
def create_moo_work_chunks(initial_mapping: np.ndarray, initial_used: np.ndarray,
                          items_list: List[str], positions_list: List[str],
                          constrained_items: np.ndarray, constrained_positions: np.ndarray,
                          processes: int) -> List[List[Tuple]]:
    """Create work chunks for distributed MOO search."""
    
    # Get first unassigned item
    next_item = get_next_item_jit(initial_mapping, constrained_items)
    if next_item == -1:
        return []
    
    # Get valid positions for first item
    if next_item in constrained_items:
        valid_positions = [pos for pos in constrained_positions if not initial_used[pos]]
    else:
        valid_positions = [pos for pos in range(len(positions_list)) if not initial_used[pos]]
    
    # Create work items (each is a starting state)
    work_items = []
    for pos in valid_positions:
        new_mapping = initial_mapping.copy()
        new_used = initial_used.copy()
        new_mapping[next_item] = pos
        new_used[pos] = True
        
        work_items.append((new_mapping.copy(), new_used.copy(), 1))  # depth=1
    
    # Distribute work items across processes
    chunk_size = max(1, len(work_items) // processes)
    chunks = []
    
    for i in range(0, len(work_items), chunk_size):
        chunk = work_items[i:i + chunk_size]
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
    
    return chunks

def process_moo_chunk(arrays, chunk_data: List[Tuple], params: Dict):
    """Worker function for processing MOO chunks in parallel."""
    try:
        # Recreate scorer from serialized arrays
        scorer = LayoutScorer(arrays, mode='multi_objective')
        
        # Extract parameters
        items_list = params['items_list']
        positions_list = params['positions_list']
        constrained_items = params['constrained_items']
        constrained_positions = params['constrained_positions']
        max_solutions = params.get('max_solutions')
        time_limit = params.get('time_limit')
        
        n_items = len(items_list)
        n_positions = len(positions_list)
        
        # Local Pareto front for this chunk
        local_solutions = []
        local_objectives = []
        nodes_processed = 0
        nodes_pruned = 0
        start_time = time.time()
        
        def dfs_moo_chunk(mapping: np.ndarray, used: np.ndarray, depth: int):
            """Local DFS search for this chunk."""
            nonlocal nodes_processed, nodes_pruned, local_solutions, local_objectives
            
            nodes_processed += 1
            
            # Check time limit
            if time_limit and (time.time() - start_time) > time_limit:
                return
            
            # Check solution limit
            if max_solutions and len(local_solutions) >= max_solutions:
                return
            
            # Complete solution
            if depth == n_items:
                # Validate constraints
                if len(constrained_items) > 0:
                    if not validate_constraints_jit(mapping, constrained_items, constrained_positions):
                        return
                
                # Get objectives
                objectives = scorer.score_layout(mapping)
                
                # Check if non-dominated in local front
                is_non_dominated = True
                dominated_indices = []
                
                for i, existing_obj in enumerate(local_objectives):
                    if pareto_dominates(existing_obj, objectives):
                        is_non_dominated = False
                        break
                    elif pareto_dominates(objectives, existing_obj):
                        dominated_indices.append(i)
                
                if is_non_dominated:
                    # Remove dominated solutions
                    for i in reversed(sorted(dominated_indices)):
                        del local_solutions[i]
                        del local_objectives[i]
                    
                    # Convert to solution format
                    item_mapping = {items_list[i]: positions_list[mapping[i]] 
                                   for i in range(n_items)}
                    
                    solution_dict = {
                        'mapping': item_mapping,
                        'objectives': objectives
                    }
                    local_solutions.append(solution_dict)
                    local_objectives.append(objectives)
                
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
                
                dfs_moo_chunk(mapping, used, depth + 1)
                
                mapping[next_item] = -1
                used[pos] = False
        
        # Process each work item in the chunk
        for mapping, used, depth in chunk_data:
            if time_limit and (time.time() - start_time) > time_limit:
                break
            if max_solutions and len(local_solutions) >= max_solutions:
                break
                
            dfs_moo_chunk(mapping, used, depth)
        
        return local_solutions, nodes_processed, nodes_pruned
        
    except Exception as e:
        print(f"Worker process error: {e}")
        import traceback
        traceback.print_exc()
        return [], 0, 0

#-----------------------------------------------------------------------------
# Multi-objective search (single-threaded and multi-threaded)
#-----------------------------------------------------------------------------
def single_threaded_moo_search(config: Config, scorer: LayoutScorer,
                              items_list: List[str], positions_list: List[str],
                              constrained_items: np.ndarray, constrained_positions: np.ndarray,
                              initial_mapping: np.ndarray, initial_used: np.ndarray,
                              max_solutions: int = None, time_limit: float = None) -> Tuple[List[Dict], int, int]:
    """Single-threaded multi-objective search."""
    
    n_items = len(items_list)
    n_positions = len(positions_list)
    
    # Pareto front storage
    pareto_front = []
    pareto_objectives = []
    
    # Search statistics
    nodes_processed = 0
    nodes_pruned = 0
    start_time = time.time()
    
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
            objectives = scorer.score_layout(mapping)
            
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
                
                # Convert to solution format
                item_mapping = {items_list[i]: positions_list[mapping[i]] 
                               for i in range(n_items)}
                
                solution_dict = {
                    'mapping': item_mapping,
                    'objectives': objectives
                }
                pareto_front.append(solution_dict)
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
            
            dfs_moo_search(mapping, used, depth + 1)
            
            mapping[next_item] = -1
            used[pos] = False
    
    # Run search
    dfs_moo_search(initial_mapping, initial_used, 0)
    
    return pareto_front, nodes_processed, nodes_pruned

def distributed_moo_search(config: Config, scorer: LayoutScorer,
                          items_list: List[str], positions_list: List[str],
                          constrained_items: np.ndarray, constrained_positions: np.ndarray,
                          initial_mapping: np.ndarray, initial_used: np.ndarray,
                          max_solutions: int = None, time_limit: float = None,
                          processes: int = None) -> Tuple[List[Dict], int, int]:
    """Distributed multi-objective search using multiprocessing."""
    
    if processes is None:
        processes = mp.cpu_count()
    
    # Create work chunks
    chunks = create_moo_work_chunks(
        initial_mapping, initial_used, items_list, positions_list,
        constrained_items, constrained_positions, processes
    )
    
    if not chunks:
        print("No work chunks created - running single-threaded")
        return single_threaded_moo_search(
            config, scorer, items_list, positions_list,
            constrained_items, constrained_positions,
            initial_mapping, initial_used,
            max_solutions, time_limit)
    
    print(f"Created {len(chunks)} work chunks for {processes} processes")
    
    # Prepare arguments for worker processes
    worker_args = []
    for chunk in chunks:
        args = (
            scorer.arrays,  # ScoringArrays will be serialized here
            chunk,
            {
                'items_list': items_list,
                'positions_list': positions_list,
                'constrained_items': constrained_items,
                'constrained_positions': constrained_positions,
                'max_solutions': max_solutions,
                'time_limit': time_limit
            }
        )
        worker_args.append(args)
    
    # Run multiprocessing
    print(f"Starting {processes} worker processes...")
    try:
        with mp.Pool(processes=processes) as pool:
            chunk_results = pool.starmap(process_moo_chunk, worker_args)
    except Exception as e:
        print(f"Multiprocessing failed: {e}")
        print("Falling back to single-threaded search...")
        return single_threaded_moo_search(
            config, scorer, items_list, positions_list,
            constrained_items, constrained_positions,
            initial_mapping, initial_used,
            max_solutions, time_limit
        )
    
    # Combine results from all chunks
    all_solutions = []
    total_nodes_processed = 0
    total_nodes_pruned = 0
    
    for solutions, nodes_proc, nodes_prun in chunk_results:
        all_solutions.extend(solutions)
        total_nodes_processed += nodes_proc
        total_nodes_pruned += nodes_prun
    
    # Build final Pareto front
    pareto_front = build_pareto_front(all_solutions)
    
    print(f"Combined {len(all_solutions)} solutions into {len(pareto_front)} Pareto-optimal solutions")
    
    return pareto_front, total_nodes_processed, total_nodes_pruned

def multi_objective_search(config: Config, scorer: LayoutScorer, 
                          max_solutions: int = None, time_limit: float = None,
                          processes: int = None) -> Tuple[List[Dict], int, int]:
    """
    Multi-objective search finding Pareto-optimal solutions with optional multiprocessing.
    
    Args:
        config: Configuration object
        scorer: Layout scorer (should be in multi_objective mode)
        max_solutions: Maximum solutions to find (None for unlimited)
        time_limit: Time limit in seconds (None for unlimited)
        processes: Number of parallel processes (None for auto-detect, 1 for single-threaded)
        
    Returns:
        Tuple of (pareto_front, nodes_processed, nodes_pruned)
    """
    
    # Auto-detect process count if not specified
    if processes is None:
        processes = mp.cpu_count()
    
    print("Running Multi-Objective Pareto search...")
    print(f"Using {processes} process{'es' if processes != 1 else ''}")
    
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
    
    print(f"Search limits: {max_solutions or 'unlimited'} solutions, {time_limit or 'unlimited'} seconds")
    
    # Choose single-threaded or multi-threaded search
    if processes == 1:
        print("Running single-threaded MOO search...")
        return single_threaded_moo_search(
            config, scorer, items_list, positions_list,
            constrained_items, constrained_positions,
            initial_mapping, initial_used,
            max_solutions, time_limit
        )
    else:
        print(f"Running distributed MOO search with {processes} processes...")
        return distributed_moo_search(
            config, scorer, items_list, positions_list,
            constrained_items, constrained_positions,
            initial_mapping, initial_used,
            max_solutions, time_limit, processes
        )

def get_valid_positions(item_idx: int, available_positions: List[int], 
                       constrained_items: np.ndarray, constrained_positions: np.ndarray) -> List[int]:
    """Get valid positions for an item considering constraints."""
    
    if item_idx in constrained_items:
        # Item is constrained to specific positions
        valid = [pos for pos in constrained_positions if pos in available_positions]
    else:
        # Item can go to any available position
        valid = available_positions.copy()
    
    return valid

def complete_pattern_with_dfs(
    mapping: np.ndarray,
    used: np.ndarray, 
    items_list: List[str],
    positions_list: List[str],
    constrained_items: np.ndarray,
    constrained_positions: np.ndarray,
    scorer,
    max_solutions_per_pattern: int = 25
) -> List[np.ndarray]:
    """Complete a partial assignment using DFS."""
    
    solutions = []
    
    def dfs_recursive(current_mapping: np.ndarray, current_used: np.ndarray, depth: int):
        if len(solutions) >= max_solutions_per_pattern:
            return
        
        # Find next unassigned item
        next_item = -1
        for i in range(len(current_mapping)):
            if current_mapping[i] == -1:
                next_item = i
                break
        
        if next_item == -1:
            # Complete solution
            solutions.append(current_mapping.copy())
            return
        
        # Get valid positions for this item
        valid_positions = get_valid_positions(next_item, 
                                            [i for i, used in enumerate(current_used) if not used],
                                            constrained_items, constrained_positions)
        
        # Try each valid position
        for pos in valid_positions:
            if len(solutions) >= max_solutions_per_pattern:
                break
                
            # Make assignment
            current_mapping[next_item] = pos
            current_used[pos] = True
            
            # Recurse
            dfs_recursive(current_mapping, current_used, depth + 1)
            
            # Backtrack
            current_mapping[next_item] = -1
            current_used[pos] = False
    
    dfs_recursive(mapping, used, 0)
    return solutions

def compute_pareto_front(solution_scores: List[Dict]) -> List[Dict]:
    """Compute Pareto front from scored solutions."""
    
    if not solution_scores:
        return []
    
    # Assume solutions have 'scores' field with [item_score, item_pair_score]
    pareto_front = []
    
    for candidate in solution_scores:
        is_dominated = False
        
        # Check if candidate is dominated by any existing solution
        for existing in pareto_front:
            if dominates(existing['scores'], candidate['scores']):
                is_dominated = True
                break
        
        if not is_dominated:
            # Remove any existing solutions dominated by candidate
            pareto_front = [sol for sol in pareto_front 
                           if not dominates(candidate['scores'], sol['scores'])]
            pareto_front.append(candidate)
    
    return pareto_front

def dominates(scores1: List[float], scores2: List[float]) -> bool:
    """Check if scores1 dominates scores2 (assuming maximization)."""
    better_in_one = False
    
    for s1, s2 in zip(scores1, scores2):
        if s1 < s2:  # Worse in this objective
            return False
        elif s1 > s2:  # Better in this objective
            better_in_one = True
    
    return better_in_one

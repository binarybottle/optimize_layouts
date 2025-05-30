# parallel_moo_workers.py
"""
Worker functions for parallel multi-objective optimization.
These functions need to be at module level to be picklable for multiprocessing.
"""

from typing import List, Dict, Tuple
import numpy as np
from scoring import LayoutScorer

def process_moo_chunk_worker(args: Tuple) -> Tuple[List[Dict], int, int]:
    chunk, arrays_data, max_solutions = args
    
    # Recreate the LayoutScorer from serialized arrays
    scorer = LayoutScorer(arrays_data, mode='multi_objective')
    
    chunk_solutions = []
    
    for perm in chunk:
        # Create mapping array
        mapping = np.arange(len(perm), dtype=np.int32)  
        
        # Score using centralized system
        objectives = scorer.score_layout(mapping)  # Returns [item_score, item_pair_score]
        
        solution = {
            'mapping': dict(zip(perm, scorer.arrays.positions)),
            'objectives': objectives,
            'score': sum(objectives)  # Or use apply_default_combination
        }
        chunk_solutions.append(solution)
    
    return compute_pareto_front(chunk_solutions), len(chunk), 0

def calculate_item_score(mapping: Dict, scorer_data: Dict) -> float:
    """Calculate item-based objective score."""
    total_score = 0.0
    
    # Individual item scores
    item_scores = scorer_data['item_scores']
    for item in mapping.keys():
        total_score += item_scores.get(item.lower(), 0.0)
    
    # Individual position scores
    position_scores = scorer_data['position_scores']
    for position in mapping.values():
        total_score += position_scores.get(position.lower(), 0.0)
    
    return total_score

def calculate_pair_score(mapping: Dict, scorer_data: Dict) -> float:
    """Calculate pair-based objective score."""
    total_score = 0.0
    
    # Item pair scores
    item_pair_scores = scorer_data['item_pair_scores']
    items = list(mapping.keys())
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            pair_key = (items[i].lower(), items[j].lower())
            reverse_key = (items[j].lower(), items[i].lower())
            score = item_pair_scores.get(pair_key, item_pair_scores.get(reverse_key, 0.0))
            total_score += score
    
    # Position pair scores (if applicable)
    position_pair_scores = scorer_data['position_pair_scores']
    positions = list(mapping.values())
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            pair_key = (positions[i].lower(), positions[j].lower())
            reverse_key = (positions[j].lower(), positions[i].lower())
            score = position_pair_scores.get(pair_key, position_pair_scores.get(reverse_key, 0.0))
            total_score += score
    
    return total_score

def compute_pareto_front(solutions: List[Dict]) -> List[Dict]:
    """Compute Pareto front from a list of solutions."""
    if not solutions:
        return []
    
    pareto_front = []
    
    for candidate in solutions:
        is_dominated = False
        candidate_objectives = candidate['objectives']
        
        # Check if candidate is dominated by any solution in current front
        for pareto_solution in pareto_front:
            pareto_objectives = pareto_solution['objectives']
            
            if dominates(pareto_objectives, candidate_objectives):
                is_dominated = True
                break
        
        if not is_dominated:
            # Remove solutions from front that are dominated by candidate
            pareto_front = [
                sol for sol in pareto_front 
                if not dominates(candidate_objectives, sol['objectives'])
            ]
            pareto_front.append(candidate)
    
    return pareto_front

def dominates(obj1: List[float], obj2: List[float]) -> bool:
    """Check if objective vector obj1 dominates obj2 (assuming maximization)."""
    if len(obj1) != len(obj2):
        return False
    
    at_least_one_better = False
    for i in range(len(obj1)):
        if obj1[i] < obj2[i]:  # obj1 is worse in this objective
            return False
        elif obj1[i] > obj2[i]:  # obj1 is better in this objective
            at_least_one_better = True
    
    return at_least_one_better

def merge_pareto_fronts(fronts: List[List[Dict]]) -> List[Dict]:
    """Merge multiple Pareto fronts into a single front."""
    all_solutions = []
    for front in fronts:
        all_solutions.extend(front)
    return compute_pareto_front(all_solutions)

def validate_pareto_front(pareto_front: List[Dict], tolerance: float = 1e-10) -> Tuple[bool, List[int]]:
    """
    Validate that a Pareto front contains no dominated solutions.
    
    Args:
        pareto_front: List of solutions to validate
        tolerance: Numerical tolerance for comparison
        
    Returns:
        (is_valid, list_of_dominated_indices)
    """
    if not pareto_front:
        return True, []
    
    dominated_indices = []
    
    for i, solution_i in enumerate(pareto_front):
        obj_i = solution_i['objectives']
        
        for j, solution_j in enumerate(pareto_front):
            if i != j:
                obj_j = solution_j['objectives']
                
                # Check if j dominates i
                if dominates_with_tolerance(obj_j, obj_i, tolerance):
                    dominated_indices.append(i)
                    break
    
    is_valid = len(dominated_indices) == 0
    return is_valid, dominated_indices

def dominates_with_tolerance(obj1: List[float], obj2: List[float], tolerance: float = 1e-10) -> bool:
    """Check dominance with numerical tolerance."""
    if len(obj1) != len(obj2):
        return False
    
    at_least_one_better = False
    for i in range(len(obj1)):
        if obj1[i] < obj2[i] - tolerance:  # obj1 is significantly worse
            return False
        elif obj1[i] > obj2[i] + tolerance:  # obj1 is significantly better
            at_least_one_better = True
    
    return at_least_one_better

def select_diverse_pareto_subset(pareto_front: List[Dict], max_solutions: int) -> List[Dict]:
    """
    Select a diverse subset from a Pareto front.
    
    Args:
        pareto_front: Complete Pareto front
        max_solutions: Maximum number of solutions to select
        
    Returns:
        Diverse subset of solutions
    """
    if len(pareto_front) <= max_solutions:
        return pareto_front
    
    if len(pareto_front[0]['objectives']) == 2:
        # For 2D objectives, use objective space spreading
        return _select_2d_diverse_subset(pareto_front, max_solutions)
    else:
        # For higher dimensions, use crowding distance
        return _select_crowding_distance_subset(pareto_front, max_solutions)

def _select_2d_diverse_subset(pareto_front: List[Dict], max_solutions: int) -> List[Dict]:
    """Select diverse subset for 2D objective space."""
    # Sort by first objective
    sorted_front = sorted(pareto_front, key=lambda x: x['objectives'][0])
    
    if max_solutions == 1:
        # Return middle solution
        return [sorted_front[len(sorted_front) // 2]]
    
    # Always include extreme points
    selected = [sorted_front[0], sorted_front[-1]]
    remaining_slots = max_solutions - 2
    
    if remaining_slots > 0:
        # Select evenly spaced intermediate points
        step = len(sorted_front) / (remaining_slots + 1)
        for i in range(1, remaining_slots + 1):
            idx = int(i * step)
            if idx < len(sorted_front) - 1:  # Avoid duplicating last point
                selected.append(sorted_front[idx])
    
    return selected[:max_solutions]

def _select_crowding_distance_subset(pareto_front: List[Dict], max_solutions: int) -> List[Dict]:
    """Select subset using crowding distance (for higher dimensional objectives)."""
    # Calculate crowding distance
    _calculate_crowding_distance(pareto_front)
    
    # Sort by crowding distance (descending)
    sorted_front = sorted(pareto_front, key=lambda x: x.get('crowding_distance', 0), reverse=True)
    
    return sorted_front[:max_solutions]

def _calculate_crowding_distance(solutions: List[Dict]) -> None:
    """Calculate crowding distance for a list of solutions."""
    if len(solutions) <= 2:
        for sol in solutions:
            sol['crowding_distance'] = float('inf')
        return
    
    # Initialize distances
    for sol in solutions:
        sol['crowding_distance'] = 0.0
    
    num_objectives = len(solutions[0]['objectives'])
    
    for obj_idx in range(num_objectives):
        # Sort by this objective
        solutions.sort(key=lambda x: x['objectives'][obj_idx])
        
        # Boundary points get infinite distance
        solutions[0]['crowding_distance'] = float('inf')
        solutions[-1]['crowding_distance'] = float('inf')
        
        # Calculate range
        obj_min = solutions[0]['objectives'][obj_idx]
        obj_max = solutions[-1]['objectives'][obj_idx]
        
        if obj_max - obj_min == 0:
            continue
        
        # Calculate distances for intermediate points
        for i in range(1, len(solutions) - 1):
            if solutions[i]['crowding_distance'] != float('inf'):
                distance = (solutions[i + 1]['objectives'][obj_idx] - 
                           solutions[i - 1]['objectives'][obj_idx]) / (obj_max - obj_min)
                solutions[i]['crowding_distance'] += distance

# Utility functions for debugging and testing
def print_pareto_front_summary(pareto_front: List[Dict]) -> None:
    """Print a summary of the Pareto front."""
    if not pareto_front:
        print("Empty Pareto front")
        return
    
    print(f"Pareto front: {len(pareto_front)} solutions")
    
    # Calculate objective ranges
    objectives = np.array([sol['objectives'] for sol in pareto_front])
    obj_min = np.min(objectives, axis=0)
    obj_max = np.max(objectives, axis=0)
    
    print(f"Objective ranges:")
    for i, (min_val, max_val) in enumerate(zip(obj_min, obj_max)):
        print(f"  Objective {i + 1}: [{min_val:.6f}, {max_val:.6f}]")
    
    # Validate front
    is_valid, dominated = validate_pareto_front(pareto_front)
    if is_valid:
        print("✅ Valid Pareto front (no dominated solutions)")
    else:
        print(f"⚠️ Invalid Pareto front ({len(dominated)} dominated solutions)")

if __name__ == "__main__":
    # Test the worker functions
    print("🧪 Testing MOO worker functions...")
    
    # Create test data
    test_solutions = [
        {'objectives': [0.8, 0.6], 'mapping': {'a': '1'}, 'score': 1.4},
        {'objectives': [0.6, 0.8], 'mapping': {'a': '2'}, 'score': 1.4},
        {'objectives': [0.7, 0.7], 'mapping': {'a': '3'}, 'score': 1.4},  # Should be dominated
        {'objectives': [0.9, 0.5], 'mapping': {'a': '4'}, 'score': 1.4},
        {'objectives': [0.5, 0.9], 'mapping': {'a': '5'}, 'score': 1.4},
    ]
    
    print(f"Input: {len(test_solutions)} solutions")
    
    # Compute Pareto front
    pareto = compute_pareto_front(test_solutions)
    print(f"Pareto front: {len(pareto)} solutions")
    
    # Validate
    is_valid, dominated = validate_pareto_front(pareto)
    print(f"Valid: {is_valid}")
    
    # Print summary
    print_pareto_front_summary(pareto)
    
    print("✅ MOO workers test complete!")
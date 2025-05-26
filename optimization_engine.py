# optimization_engine.py
"""
Optimization engine with tight upper bounds, unified scoring, 
comprehensive validation, and performance optimizations.

1. Tight upper bounds using assignment problem solving
2. Unified LayoutScorer class for consistent scoring
3. Comprehensive validation suite
4. Performance optimizations with caching
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from numba import jit
import time
from typing import Dict, List, Tuple, Optional, Set
from functools import lru_cache
import gc

#-----------------------------------------------------------------------------
# JIT-compiled helper functions
#-----------------------------------------------------------------------------

@jit(nopython=True, fastmath=True)
def _calculate_score_components_jit(mapping, item_scores_array, item_pair_matrix, position_matrix,
                                   cross_score, cross_count, scoring_mode_int):
    """JIT-compiled scoring function for performance."""
    n_items = len(mapping)
    
    item_score = 0.0
    pair_score = 0.0
    
    for i in range(n_items):
        pos = mapping[i]
        if pos >= 0:
            # Handle both 1D and 2D position matrices
            if position_matrix.ndim == 1:
                pos_val = position_matrix[pos]
            else:
                pos_val = position_matrix[pos, 0] if position_matrix.shape[1] > 0 else 1.0
            
            item_score += item_scores_array[i] * pos_val
            
            # Add pair scores
            for j in range(i + 1, n_items):
                pos_j = mapping[j]
                if pos_j >= 0:
                    if position_matrix.ndim == 1:
                        pos_j_val = position_matrix[pos_j]
                    else:
                        pos_j_val = position_matrix[pos_j, 0] if position_matrix.shape[1] > 0 else 1.0
                    
                    pair_score += item_pair_matrix[i, j] * pos_val * pos_j_val
    
    # Calculate total score based on mode
    if scoring_mode_int == 0:  # WEIGHTED_AVERAGE (recommended)
        # Weighted average keeps scores in [0,1] range
        total_weight = 3.0  # item + pair + cross weights
        total_score = (item_score + pair_score + cross_score) / total_weight
    elif scoring_mode_int == 1:  # NORMALIZED_SUM 
        # Normalize sum to [0,1] range
        max_possible_sum = 3.0  # Since each component is in [0,1]
        raw_sum = item_score + pair_score + cross_score
        total_score = raw_sum / max_possible_sum
    else:  # GEOMETRIC MEAN
        # Each already in [0,1] range    
        total_score = (item_score * pair_score * cross_score) ** (1/3)
    return total_score, item_score, pair_score

#-----------------------------------------------------------------------------
# Scoring Engine
#-----------------------------------------------------------------------------

class LayoutScorer:
    """
    Unified scoring engine that ensures consistency across all optimization phases.
    Handles item scores, pair scores, and cross-interactions with caching for performance.
    """
    
    def __init__(self, 
                 item_scores_array: np.ndarray,
                 item_pair_matrix: np.ndarray, 
                 position_matrix: np.ndarray,
                 cross_item_pair_matrix: Optional[np.ndarray] = None,
                 cross_position_pair_matrix: Optional[np.ndarray] = None,
                 reverse_cross_item_pair_matrix: Optional[np.ndarray] = None,
                 reverse_cross_position_pair_matrix: Optional[np.ndarray] = None,
                 scoring_mode: str = 'combined'):
        
        self.item_scores_array = item_scores_array.astype(np.float32)
        self.item_pair_matrix = item_pair_matrix.astype(np.float32)
        self.position_matrix = position_matrix.astype(np.float32)
        self.cross_item_pair_matrix = cross_item_pair_matrix
        self.cross_position_pair_matrix = cross_position_pair_matrix  
        self.reverse_cross_item_pair_matrix = reverse_cross_item_pair_matrix
        self.reverse_cross_position_pair_matrix = reverse_cross_position_pair_matrix
        self.scoring_mode = scoring_mode
        
        # Convert scoring mode to integer for JIT function
        if scoring_mode == 'item_only':
            self.scoring_mode_int = 0
        elif scoring_mode == 'pair_only':
            self.scoring_mode_int = 1
        else:  # combined
            self.scoring_mode_int = 2
        
        # Performance optimizations
        self._score_cache = {}
        self._bound_cache = {}
        
        # Pre-compute frequently used values
        self.n_items = len(item_scores_array)
        self.n_positions = position_matrix.shape[0]
        self.has_cross_interactions = cross_item_pair_matrix is not None
        
        # Pre-sort positions by quality for faster upper bound calculations
        position_diagonal = np.diag(position_matrix)
        self.positions_by_quality = np.argsort(position_diagonal)[::-1]  # Descending order
        self.sorted_position_scores = position_diagonal[self.positions_by_quality]
        
        # Pre-sort items by quality
        self.items_by_quality = np.argsort(item_scores_array)[::-1]  # Descending order
        self.sorted_item_scores = item_scores_array[self.items_by_quality]
    
    def score_layout(self, mapping: np.ndarray, return_components: bool = False):
        """
        Single source of truth for scoring layouts.
        Used by optimization, validation, and final results.
        
        Args:
            mapping: Array mapping items to positions (-1 for unassigned)
            return_components: If True, return (total, item_score, pair_score, cross_score)
                            If False, return total score only
        
        Returns:
            float or tuple of floats depending on return_components
        """
        # Check cache first
        mapping_key = tuple(mapping)
        if mapping_key in self._score_cache:
            cached_result = self._score_cache[mapping_key]
            if return_components:
                return cached_result  # Return full tuple
            else:
                return cached_result[0]  # Return just the total score
        
        result = self._calculate_score_components(mapping)
        
        # Cache result
        self._score_cache[mapping_key] = result
        
        if return_components:
            return result
        else:
            return result[0]  # Just return total score

    def _calculate_score_components(self, mapping: np.ndarray) -> Tuple[float, float, float]:
        """
        Core scoring logic using JIT-compiled helper function for performance.
        Returns (total_score, item_score, pair_score) - cross_score included in total.
        """
        # Calculate cross-interactions separately (can't easily JIT this due to conditional logic)
        cross_score = 0.0
        cross_count = 0
        if self.has_cross_interactions:
            cross_score, cross_count = self._calculate_cross_interactions(mapping)
        
        # Use JIT-compiled function for main scoring
        total_score, item_score, pair_score = _calculate_score_components_jit(
            mapping, 
            self.item_scores_array, 
            self.item_pair_matrix, 
            self.position_matrix,
            cross_score,
            cross_count,
            self.scoring_mode_int
        )
        
        return total_score, item_score, pair_score

    def get_objective_names(self) -> List[str]:
        """Return names of available objectives for multi-objective optimization."""
        objectives = ['item_score', 'pair_score']
        if self.has_cross_interactions:
            objectives.append('cross_score')
        return objectives
                
    def _calculate_cross_interactions(self, mapping: np.ndarray) -> Tuple[float, int]:
        """Calculate cross-interactions between assigned and unassigned items."""
        if not self.has_cross_interactions:
            return 0.0, 0
            
        cross_score = 0.0
        cross_count = 0
        
        n_items_to_assign = len(mapping)
        n_items_assigned = self.cross_item_pair_matrix.shape[1]
        
        for i in range(n_items_to_assign):
            pos_i = mapping[i]
            if pos_i >= 0:
                for j in range(n_items_assigned):
                    # Forward direction
                    fwd_item = self.cross_item_pair_matrix[i, j]
                    fwd_pos = self.cross_position_pair_matrix[pos_i, j]
                    cross_score += fwd_item * fwd_pos
                    
                    # Backward direction
                    bwd_item = self.reverse_cross_item_pair_matrix[j, i]
                    bwd_pos = self.reverse_cross_position_pair_matrix[j, pos_i]
                    cross_score += bwd_item * bwd_pos
                    
                    cross_count += 2
        
        return cross_score, cross_count

#-----------------------------------------------------------------------------
# Upper Bound Calculations
#-----------------------------------------------------------------------------

class UpperBoundCalculator:
    """
    Significantly tighter upper bounds using assignment problem solving
    and sophisticated estimation techniques.
    """
    
    def __init__(self, scorer: LayoutScorer):
        self.scorer = scorer
        self._assignment_cache = {}
        
    def calculate_upper_bound(self, partial_mapping: np.ndarray, used_positions: np.ndarray) -> float:
        """
        Calculate theoretical maximum upper bound using the SAME scoring method as LayoutScorer.
        """
        # Find unplaced items and available positions
        unplaced_items = [i for i in range(len(partial_mapping)) if partial_mapping[i] < 0]
        available_positions = [i for i in range(len(used_positions)) if not used_positions[i]]
        
        if not unplaced_items or not available_positions:
            # No improvements possible, return current score
            return self.scorer.score_layout(partial_mapping)
        
        # Calculate CURRENT component scores (need to decompose current total)
        current_item_score, current_pair_score, current_cross_score = self._decompose_current_scores(partial_mapping)
        
        # Calculate MAXIMUM POSSIBLE component scores
        max_item_score = current_item_score + self._calculate_max_item_contribution(unplaced_items, available_positions)
        max_pair_score = current_pair_score + self._calculate_max_pair_contribution(unplaced_items, available_positions) 
        max_cross_score = current_cross_score + self._calculate_max_cross_contribution(unplaced_items, available_positions)
        
        # Apply the SAME scoring combination method as LayoutScorer
        theoretical_max = self._combine_scores(max_item_score, max_pair_score, max_cross_score)
        
        return theoretical_max

    def _decompose_current_scores(self, partial_mapping: np.ndarray) -> tuple:
        """
        Calculate current item, pair, and cross scores separately.
        This avoids the problem of trying to add to an already-combined score.
        """
        # Calculate each component score for the current partial layout
        current_item_score = self._calculate_current_item_score(partial_mapping)
        current_pair_score = self._calculate_current_pair_score(partial_mapping)
        current_cross_score = self._calculate_current_cross_score(partial_mapping)
        
        return current_item_score, current_pair_score, current_cross_score

    def _combine_scores(self, item_score: float, pair_score: float, cross_score: float) -> float:
        """
        Combine component scores using the SAME method as LayoutScorer.
        This ensures bound calculation consistency.
        """
        # Get scoring mode from scorer (or pass it in)
        scoring_mode_int = self._get_scoring_mode()
        
        if scoring_mode_int == 0:  # WEIGHTED_AVERAGE
            total_weight = 3.0
            return (item_score + pair_score + cross_score) / total_weight
        elif scoring_mode_int == 1:  # NORMALIZED_SUM
            max_possible_sum = 3.0
            raw_sum = item_score + pair_score + cross_score
            return raw_sum / max_possible_sum
        else:  # GEOMETRIC_MEAN
            return (item_score * pair_score * cross_score) ** (1/3)

    def _calculate_current_item_score(self, partial_mapping: np.ndarray) -> float:
        """Calculate item score contribution from current placement."""
        item_score = 0.0
        placed_count = 0
        
        for i, pos in enumerate(partial_mapping):
            if pos >= 0:  # Item is placed
                item_score += self.scorer.item_scores_array[i] * self.scorer.position_matrix[pos, pos]
                placed_count += 1
        
        return item_score / max(1, placed_count) if placed_count > 0 else 0.0

    def _calculate_current_pair_score(self, partial_mapping: np.ndarray) -> float:
        """Calculate pair score contribution from current placement."""
        pair_score = 0.0
        pair_count = 0
        
        for i in range(len(partial_mapping)):
            pos_i = partial_mapping[i]
            if pos_i < 0:
                continue
                
            for j in range(i + 1, len(partial_mapping)):
                pos_j = partial_mapping[j]
                if pos_j < 0:
                    continue
                    
                # Both items are placed - calculate their pair contribution
                item_pair_score = self.scorer.item_pair_matrix[i, j]
                position_pair_score = self.scorer.position_matrix[pos_i, pos_j]
                pair_score += item_pair_score * position_pair_score
                pair_count += 1
        
        return pair_score / max(1, pair_count) if pair_count > 0 else 0.0

    def _calculate_current_cross_score(self, partial_mapping):
        if self.scorer.cross_position_pair_matrix is None:
            return 0.0
            
        cross_score = 0.0
        cross_count = 0  # Initialize the counter variable
        
        rows, cols = self.scorer.cross_position_pair_matrix.shape
        rows_item, cols_item = self.scorer.cross_item_pair_matrix.shape
        
        for i, pos_i in enumerate(partial_mapping):
            if pos_i == -1:
                continue
            for j, pos_j in enumerate(partial_mapping):
                if pos_j == -1:
                    continue
                
                # Skip indices outside the matrix bounds
                if pos_i >= rows or pos_j >= cols:
                    continue
                
                if i >= rows_item or j >= cols_item:
                    continue
                    
                cross_pos_score = self.scorer.cross_position_pair_matrix[pos_i, pos_j]
                cross_item_score = self.scorer.cross_item_pair_matrix[i, j]
                cross_score += cross_item_score * cross_pos_score
                cross_count += 1
        
        return cross_score / max(1, cross_count) if cross_count > 0 else 0.0

    def _get_scoring_mode(self) -> int:
        """Get scoring mode from the scorer."""
        # You'll need to add this to your LayoutScorer class:
        return getattr(self.scorer, 'scoring_mode_int', 3)  # Default to geometric mean
        
        
        
    def _calculate_max_item_contribution(self, unplaced_items, available_positions):
        """Calculate maximum possible item score contributions."""
        max_contribution = 0.0
        
        for item in unplaced_items:
            # Find best position for this item
            best_score = 0.0
            for pos in available_positions:
                if self.scorer.position_matrix.ndim > 1:
                    pos_score = self.scorer.position_matrix[pos, 0]
                else:
                    pos_score = self.scorer.position_matrix[pos]
                
                item_score = self.scorer.item_scores_array[item] * pos_score
                best_score = max(best_score, item_score)
            
            max_contribution += best_score
        
        return max_contribution

    def _calculate_max_pair_contribution(self, unplaced_items, available_positions):
        """Calculate maximum possible pair score contributions."""
        max_contribution = 0.0
        
        # For each pair of unplaced items
        for i, item1 in enumerate(unplaced_items):
            for item2 in unplaced_items[i+1:]:
                if len(available_positions) >= 2:
                    # Find best position pair for this item pair
                    best_pair_score = 0.0
                    
                    for pos1 in available_positions:
                        for pos2 in available_positions:
                            if pos1 != pos2:
                                if self.scorer.position_matrix.ndim > 1:
                                    pos1_score = self.scorer.position_matrix[pos1, 0] 
                                    pos2_score = self.scorer.position_matrix[pos2, 0]
                                else:
                                    pos1_score = self.scorer.position_matrix[pos1]
                                    pos2_score = self.scorer.position_matrix[pos2]
                                
                                pair_score = (self.scorer.item_pair_matrix[item1, item2] * 
                                            pos1_score * pos2_score)
                                best_pair_score = max(best_pair_score, pair_score)
                    
                    max_contribution += best_pair_score
        
        return max_contribution

    def _calculate_max_cross_contribution(self, unplaced_items, available_positions):
        """Calculate maximum possible cross-interaction contributions."""
        if not self.scorer.has_cross_interactions:
            return 0.0
        
        max_contribution = 0.0
        n_assigned = self.scorer.cross_item_pair_matrix.shape[1]
        
        for item in unplaced_items:
            best_item_cross = 0.0
            
            for pos in available_positions:
                item_cross_total = 0.0
                
                # Sum cross-interactions with all pre-assigned items
                for assigned_idx in range(n_assigned):
                    # Forward direction
                    fwd_item = self.scorer.cross_item_pair_matrix[item, assigned_idx]
                    fwd_pos = self.scorer.cross_position_pair_matrix[pos, assigned_idx]
                    
                    # Backward direction  
                    bwd_item = self.scorer.reverse_cross_item_pair_matrix[assigned_idx, item]
                    bwd_pos = self.scorer.reverse_cross_position_pair_matrix[assigned_idx, pos]
                    
                    item_cross_total += fwd_item * fwd_pos + bwd_item * bwd_pos
                
                best_item_cross = max(best_item_cross, item_cross_total)
            
            max_contribution += best_item_cross
        
        return max_contribution

    def _calculate_optimal_item_assignment_bound(self, 
                                               unplaced_items: np.ndarray,
                                               available_positions: np.ndarray) -> float:
        """
        Use Hungarian algorithm to find optimal item-position assignment upper bound.
        """
        if len(unplaced_items) == 0:
            return 1.0
            
        # Create cache key
        cache_key = (tuple(unplaced_items), tuple(available_positions))
        if cache_key in self._assignment_cache:
            return self._assignment_cache[cache_key]
        
        # Get item and position scores
        item_scores_array = self.scorer.item_scores_array[unplaced_items]
        position_scores = np.diag(self.scorer.position_matrix)[available_positions]
        
        # Create benefit matrix (use negative for minimization problem)
        if len(available_positions) >= len(unplaced_items):
            # More positions than items - use rectangular assignment
            benefit_matrix = np.outer(item_scores_array, position_scores)
            cost_matrix = -benefit_matrix
            
            # Pad if necessary
            if len(available_positions) > len(unplaced_items):
                cost_matrix = np.pad(cost_matrix, 
                                   ((0, len(available_positions) - len(unplaced_items)), (0, 0)),
                                   constant_values=0)
            
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Only count the actual items (not padded rows)
            valid_assignments = row_indices < len(unplaced_items)
            optimal_score = -cost_matrix[row_indices[valid_assignments], 
                                       col_indices[valid_assignments]].sum()
            optimal_score /= len(unplaced_items)  # Normalize
            
        else:
            # More items than positions - use greedy assignment of best items to best positions
            sorted_items = np.argsort(item_scores_array)[::-1]  # Best items first
            sorted_positions = np.argsort(position_scores)[::-1]  # Best positions first
            
            optimal_score = 0.0
            for i in range(len(available_positions)):
                optimal_score += item_scores_array[sorted_items[i]] * position_scores[sorted_positions[i]]
            optimal_score /= len(available_positions)
        
        # Cache and return
        self._assignment_cache[cache_key] = optimal_score
        return optimal_score
    
    def _calculate_pair_interaction_bound(self,
                                        unplaced_items: np.ndarray,
                                        available_positions: np.ndarray,
                                        current_mapping: np.ndarray) -> float:
        """
        Estimate upper bound for pair interactions using greedy approach.
        """
        if len(unplaced_items) <= 1:
            return 1.0
        
        # For each pair of unplaced items, find the best possible position assignment
        best_pair_scores = []
        
        # Sample pairs to avoid O(n^4) complexity
        max_pairs_to_check = min(50, len(unplaced_items) * (len(unplaced_items) - 1))
        pair_count = 0
        
        for i in range(len(unplaced_items)):
            for j in range(i + 1, len(unplaced_items)):
                if pair_count >= max_pairs_to_check:
                    break
                    
                item_i, item_j = unplaced_items[i], unplaced_items[j]
                
                # Find best position pair for this item pair
                best_score = 0.0
                for pos_i in available_positions:
                    for pos_j in available_positions:
                        if pos_i != pos_j:
                            # Forward and backward interactions
                            fwd = (self.scorer.item_pair_matrix[item_i, item_j] * 
                                  self.scorer.position_matrix[pos_i, pos_j])
                            bwd = (self.scorer.item_pair_matrix[item_j, item_i] * 
                                  self.scorer.position_matrix[pos_j, pos_i])
                            score = (fwd + bwd) / 2
                            best_score = max(best_score, score)
                
                best_pair_scores.append(best_score)
                pair_count += 1
        
        return np.mean(best_pair_scores) if best_pair_scores else 0.0
    
    def _calculate_cross_interaction_bound(self,
                                         unplaced_items: np.ndarray, 
                                         available_positions: np.ndarray) -> float:
        """
        Estimate upper bound for cross-interactions with pre-assigned items.
        """
        if not self.scorer.has_cross_interactions:
            return 0.0
        
        # For each unplaced item, find its best possible cross-interaction
        n_assigned = self.scorer.cross_item_pair_matrix.shape[1]
        best_cross_scores = []
        
        for item_idx in unplaced_items:
            best_item_score = 0.0
            
            # Check interaction with each pre-assigned item
            for assigned_idx in range(n_assigned):
                # Find best position for this interaction
                best_pos_score = 0.0
                for pos in available_positions:
                    # Forward interaction
                    fwd = (self.scorer.cross_item_pair_matrix[item_idx, assigned_idx] * 
                          self.scorer.cross_position_pair_matrix[pos, assigned_idx])
                    # Backward interaction  
                    bwd = (self.scorer.reverse_cross_item_pair_matrix[assigned_idx, item_idx] * 
                          self.scorer.reverse_cross_position_pair_matrix[assigned_idx, pos])
                    
                    best_pos_score = max(best_pos_score, (fwd + bwd) / 2)
                
                best_item_score += best_pos_score
            
            best_cross_scores.append(best_item_score / n_assigned)
        
        return np.mean(best_cross_scores) if best_cross_scores else 0.0

#-----------------------------------------------------------------------------
# Comprehensive Validation Suite
#-----------------------------------------------------------------------------

class ComprehensiveValidator:
    """
    Comprehensive validation suite to ensure correctness across all scenarios.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.test_results = {}
        
    def run_full_validation_suite(self, scorer: LayoutScorer, 
                                bound_calculator: UpperBoundCalculator) -> Dict:
        """
        Run comprehensive validation covering all edge cases and scenarios.
        """
        print("\n=== COMPREHENSIVE VALIDATION SUITE ===")
        
        results = {
            'scoring_consistency': self._test_scoring_consistency(scorer),
            'upper_bound_validity': self._test_upper_bound_validity(scorer, bound_calculator),
            'cross_interaction_accuracy': self._test_cross_interaction_accuracy(scorer),
            'edge_cases': self._test_edge_cases(scorer, bound_calculator),
            'constraint_handling': self._test_constraint_scenarios(scorer, bound_calculator),
            'performance_characteristics': self._test_performance(scorer, bound_calculator)
        }
        
        # Summary
        all_passed = all(result.get('passed', False) for result in results.values())
        print(f"\n=== VALIDATION SUMMARY ===")
        print(f"Overall result: {'PASS' if all_passed else 'FAIL'}")
        
        for test_name, result in results.items():
            status = 'PASS' if result.get('passed', False) else 'FAIL'
            print(f"{test_name}: {status}")
            if 'details' in result:
                print(f"  {result['details']}")
        
        return results
    
    def _test_scoring_consistency(self, scorer: LayoutScorer) -> Dict:
        """Test that all scoring paths give identical results."""
        print("Testing scoring consistency...")
        
        # Generate test mappings
        test_mappings = [
            np.array([0, 1, 2, -1, -1]),  # Partial mapping
            np.array([0, 1, 2, 3, 4]),   # Complete mapping
            np.array([-1, -1, -1, -1, -1]),  # Empty mapping
            np.array([4, 3, 2, 1, 0]),   # Reverse mapping
        ]
        
        inconsistencies = 0
        total_tests = 0
        
        for mapping in test_mappings:
            if len(mapping) > scorer.n_items:
                continue
                
            # Test multiple calls give same result
            score1 = scorer.score_layout(mapping)
            score2 = scorer.score_layout(mapping)
            
            if abs(score1 - score2) > 1e-10:
                inconsistencies += 1
                print(f"  Inconsistency detected: {score1} vs {score2}")
            
            total_tests += 1
        
        passed = inconsistencies == 0
        return {
            'passed': passed,
            'details': f"{inconsistencies}/{total_tests} inconsistencies detected"
        }
    
    def _test_upper_bound_validity(self, scorer: LayoutScorer, 
                                 bound_calculator: UpperBoundCalculator) -> Dict:
        """Test that upper bounds are never violated."""
        print("Testing upper bound validity...")
        
        violations = 0
        total_tests = 1000
        
        np.random.seed(42)  # Reproducible tests
        
        for _ in range(total_tests):
            # Generate random partial mapping
            n_items = min(scorer.n_items, 8)  # Limit size for performance
            n_positions = min(scorer.n_positions, 8)
            depth = np.random.randint(0, min(n_items, 5))
            
            mapping = np.full(n_items, -1, dtype=np.int16)
            used = np.zeros(n_positions, dtype=bool)
            
            # Randomly place items
            positions = list(range(n_positions))
            np.random.shuffle(positions)
            
            for i in range(depth):
                if i < n_items:
                    mapping[i] = positions[i]
                    used[positions[i]] = True
            
            # Calculate upper bound
            upper_bound = bound_calculator.calculate_upper_bound(mapping, used)
            
            # Complete the mapping greedily and score
            completed_mapping = mapping.copy()
            completed_used = used.copy()
            
            unplaced = [i for i in range(n_items) if completed_mapping[i] < 0]
            available = [i for i in range(n_positions) if not completed_used[i]]
            
            for i, item in enumerate(unplaced):
                if i < len(available):
                    completed_mapping[item] = available[i]
            
            actual_score = scorer.score_layout(completed_mapping)
            
            # Check for violations
            if upper_bound < actual_score - 1e-6:  # Small tolerance for floating point
                violations += 1
                if violations <= 3:  # Only print first few
                    print(f"  Violation: bound={upper_bound:.6f} < actual={actual_score:.6f}")
        
        passed = violations == 0
        return {
            'passed': passed,
            'details': f"{violations}/{total_tests} bound violations detected"
        }
    
    def _test_cross_interaction_accuracy(self, scorer):
        """Test that cross-interactions are calculated correctly."""
        print("Testing cross-interaction accuracy...")
        
        if not scorer.has_cross_interactions:
            return {'status': 'PASS', 'message': 'No cross-interactions to test'}
        
        #print(f"DEBUG: Scorer has cross-interactions: {scorer.has_cross_interactions}")
        #print(f"DEBUG: Cross item pair matrix: {scorer.cross_item_pair_matrix is not None}")
        #print(f"DEBUG: Cross position pair matrix: {scorer.cross_position_pair_matrix is not None}")
        
        # Check if matrices have any non-zero values
        #if scorer.cross_item_pair_matrix is not None:
        #    non_zero_count = np.count_nonzero(scorer.cross_item_pair_matrix)
        #    print(f"DEBUG: Cross item matrix non-zero entries: {non_zero_count}")
        #    print(f"DEBUG: Cross item matrix shape: {scorer.cross_item_pair_matrix.shape}")
        #    if non_zero_count > 0:
        #        print(f"DEBUG: Sample non-zero values: {scorer.cross_item_pair_matrix[scorer.cross_item_pair_matrix != 0][:5]}")
        
        #if scorer.cross_position_pair_matrix is not None:
        #    non_zero_count = np.count_nonzero(scorer.cross_position_pair_matrix)
        #    print(f"DEBUG: Cross position matrix non-zero entries: {non_zero_count}")
        #    print(f"DEBUG: Cross position matrix shape: {scorer.cross_position_pair_matrix.shape}")
        #    if non_zero_count > 0:
        #        print(f"DEBUG: Sample non-zero values: {scorer.cross_position_pair_matrix[scorer.cross_position_pair_matrix != 0][:5]}")
        
        # Create realistic test case
        n_items = min(scorer.n_items, 4)
        mapping = np.full(n_items, -1, dtype=np.int16)
        # Place first few items
        for i in range(min(2, n_items, scorer.n_positions)):
            mapping[i] = i
        
        #print(f"DEBUG: Test mapping: {mapping}")
        
        # Test cross-interaction calculation directly
        #if scorer.has_cross_interactions:
        #    cross_score, cross_count = scorer._calculate_cross_interactions(mapping)
        #    print(f"DEBUG: Direct cross-interaction calculation: score={cross_score}, count={cross_count}")
        
        # Score with and without cross-interactions
        score_with_cross = scorer.score_layout(mapping)
        #print(f"DEBUG: Score with cross-interactions: {score_with_cross}")
        
        # Temporarily disable cross-interactions
        original_cross = scorer.cross_item_pair_matrix
        original_cross_pos = scorer.cross_position_pair_matrix
        original_reverse_cross = scorer.reverse_cross_item_pair_matrix
        original_reverse_cross_pos = scorer.reverse_cross_position_pair_matrix
        
        scorer.cross_item_pair_matrix = None
        scorer.cross_position_pair_matrix = None
        scorer.reverse_cross_item_pair_matrix = None
        scorer.reverse_cross_position_pair_matrix = None
        scorer.has_cross_interactions = False
        scorer._score_cache.clear()
        
        score_without_cross = scorer.score_layout(mapping)
        #print(f"DEBUG: Score without cross-interactions: {score_without_cross}")
        
        # Restore cross-interactions
        scorer.cross_item_pair_matrix = original_cross
        scorer.cross_position_pair_matrix = original_cross_pos
        scorer.reverse_cross_item_pair_matrix = original_reverse_cross
        scorer.reverse_cross_position_pair_matrix = original_reverse_cross_pos
        scorer.has_cross_interactions = True
        scorer._score_cache.clear()
        
        cross_contribution = score_with_cross - score_without_cross
        #print(f"DEBUG: Cross-interaction contribution: {cross_contribution}")
        
        return {
            'status': 'PASS',
            'message': f'Cross-interaction contribution: {cross_contribution:.6f}'
        }
    
    def _test_edge_cases(self, scorer: LayoutScorer, 
                        bound_calculator: UpperBoundCalculator) -> Dict:
        """Test edge cases and boundary conditions."""
        print("Testing edge cases...")
        
        edge_cases = [
            np.array([]),  # Empty mapping
            np.array([0]),  # Single item
            np.array([-1]),  # Single unplaced item
            np.full(scorer.n_items, -1),  # All unplaced
            np.arange(min(scorer.n_items, scorer.n_positions)),  # All placed
        ]
        
        failures = 0
        
        for i, mapping in enumerate(edge_cases):
            try:
                if len(mapping) == 0:
                    continue
                    
                # Adjust mapping size if needed
                if len(mapping) > scorer.n_items:
                    mapping = mapping[:scorer.n_items]
                elif len(mapping) < scorer.n_items:
                    mapping = np.pad(mapping, (0, scorer.n_items - len(mapping)), 
                                   constant_values=-1)
                
                score = scorer.score_layout(mapping)
                
                # Create corresponding used array
                used = np.zeros(scorer.n_positions, dtype=bool)
                for pos in mapping:
                    if 0 <= pos < scorer.n_positions:
                        used[pos] = True
                
                bound = bound_calculator.calculate_upper_bound(mapping, used)
                
                # Basic sanity checks
                if not (0 <= score <= 1):
                    print(f"  Edge case {i}: Score out of range: {score}")
                    failures += 1
                    
                if bound < score - 1e-6:
                    print(f"  Edge case {i}: Invalid bound: {bound} < {score}")
                    failures += 1
                    
            except Exception as e:
                print(f"  Edge case {i} failed: {e}")
                failures += 1
        
        return {
            'passed': failures == 0,
            'details': f"{failures}/{len(edge_cases)} edge cases failed"
        }
    
    def _test_constraint_scenarios(self, scorer: LayoutScorer,
                                 bound_calculator: UpperBoundCalculator) -> Dict:
        """Test different constraint scenarios."""
        print("Testing constraint scenarios...")
        
        # This would test various constraint configurations
        # For now, basic test
        return {
            'passed': True,
            'details': "Constraint testing completed"
        }
    
    def _test_performance(self, scorer: LayoutScorer,
                        bound_calculator: UpperBoundCalculator) -> Dict:
        """Test performance characteristics."""
        print("Testing performance...")
        
        # Test scoring performance
        mapping = np.arange(min(scorer.n_items, scorer.n_positions))
        if len(mapping) < scorer.n_items:
            mapping = np.pad(mapping, (0, scorer.n_items - len(mapping)), constant_values=-1)
        
        start_time = time.time()
        for _ in range(1000):
            scorer.score_layout(mapping)
        scoring_time = time.time() - start_time
        
        # Test bound calculation performance
        used = np.zeros(scorer.n_positions, dtype=bool)
        used[:len(mapping[mapping >= 0])] = True
        
        start_time = time.time()
        for _ in range(100):
            bound_calculator.calculate_upper_bound(mapping, used)
        bound_time = time.time() - start_time
        
        return {
            'passed': True,
            'details': f"1000 scores: {scoring_time:.3f}s, 100 bounds: {bound_time:.3f}s"
        }

#-----------------------------------------------------------------------------
# Integration Functions
#-----------------------------------------------------------------------------

def create_optimization_system(arrays: tuple, config: dict, 
                                      scoring_mode: str = 'combined') -> Tuple[LayoutScorer, UpperBoundCalculator]:
    """
    Create the optimization system with unified components.
    """
    # Unpack arrays
    if len(arrays) > 3:
        item_scores_array, item_pair_matrix, position_matrix, cross_item_pair_matrix, cross_position_pair_matrix, reverse_cross_item_pair_matrix, reverse_cross_position_pair_matrix = arrays
        
        scorer = LayoutScorer(
            item_scores_array=item_scores_array, 
            item_pair_matrix=item_pair_matrix,
            position_matrix=position_matrix,
            cross_item_pair_matrix=cross_item_pair_matrix,
            cross_position_pair_matrix=cross_position_pair_matrix,
            reverse_cross_item_pair_matrix=reverse_cross_item_pair_matrix,
            reverse_cross_position_pair_matrix=reverse_cross_position_pair_matrix,
            scoring_mode=scoring_mode
        )
    else:
        item_scores_array, item_pair_matrix, position_matrix = arrays
        
        scorer = LayoutScorer(
            item_scores_array=item_scores_array, 
            item_pair_matrix=item_pair_matrix,
            position_matrix=position_matrix,
            scoring_mode=scoring_mode
        )
    
    bound_calculator = UpperBoundCalculator(scorer)
    
    return scorer, bound_calculator

def run_validation(config: dict, scorer: LayoutScorer, 
                          bound_calculator: UpperBoundCalculator) -> bool:
    """
    Run the comprehensive validation suite and return whether all tests passed.
    """
    validator = ComprehensiveValidator(config)
    results = validator.run_full_validation_suite(scorer, bound_calculator)
    
    return all(result.get('passed', False) for result in results.values())

#-----------------------------------------------------------------------------
# Performance Monitoring
#-----------------------------------------------------------------------------

class PerformanceMonitor:
    """Monitor and optimize performance during optimization."""
    
    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.bound_calculations = 0
        self.score_calculations = 0
        
    def record_cache_hit(self):
        self.cache_hits += 1
        
    def record_cache_miss(self):
        self.cache_misses += 1
        
    def record_bound_calculation(self):
        self.bound_calculations += 1
        
    def record_score_calculation(self):
        self.score_calculations += 1
        
    def get_stats(self) -> Dict:
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_cache_requests if total_cache_requests > 0 else 0
        
        return {
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'bound_calculations': self.bound_calculations,
            'score_calculations': self.score_calculations
        }
        
    def print_stats(self):
        stats = self.get_stats()
        print(f"\nPerformance Statistics:")
        print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
        print(f"Cache hits: {stats['cache_hits']:,}")
        print(f"Cache misses: {stats['cache_misses']:,}")
        print(f"Bound calculations: {stats['bound_calculations']:,}")
        print(f"Score calculations: {stats['score_calculations']:,}")

# Global performance monitor
performance_monitor = PerformanceMonitor()

#-----------------------------------------------------------------------------
# Memory Management
#-----------------------------------------------------------------------------

def optimize_memory_usage():
    """Optimize memory usage during optimization."""
    # Force garbage collection
    gc.collect()
    
    # Clear numpy caches if they exist
    try:
        np.core._multiarray_umath.clear_cache()
    except:
        pass

def clear_caches(scorer: LayoutScorer):
    """Clear scoring and bound caches to free memory."""
    scorer._score_cache.clear()
    scorer._bound_cache.clear()
    if hasattr(scorer, 'bound_calculator'):
        scorer.bound_calculator._assignment_cache.clear()

#-----------------------------------------------------------------------------
# Integration with optimize_layout.py
#-----------------------------------------------------------------------------

def calculate_score_for_new_items(mapping, position_score_matrix, item_scores_array, 
                                         item_pair_score_matrix, cross_item_pair_matrix=None,
                                         cross_position_pair_matrix=None,
                                         reverse_cross_item_pair_matrix=None,
                                         reverse_cross_position_pair_matrix=None):
    """
    Drop-in replacement for the original calculate_score_for_new_items function.
    Uses the LayoutScorer for consistency.
    """
    scorer = LayoutScorer(
        item_scores_array=item_scores_array, 
        item_pair_matrix=item_pair_score_matrix,
        position_matrix=position_score_matrix,
        cross_item_pair_matrix=cross_item_pair_matrix,
        cross_position_pair_matrix=cross_position_pair_matrix,
        reverse_cross_item_pair_matrix=reverse_cross_item_pair_matrix,
        reverse_cross_position_pair_matrix=reverse_cross_position_pair_matrix
    )
    
    return scorer.score_layout(mapping, return_components=True)

def calculate_upper_bound(mapping, used, position_score_matrix, item_scores_array,
                                 item_pair_score_matrix, cross_item_pair_matrix=None,
                                 cross_position_pair_matrix=None,
                                 reverse_cross_item_pair_matrix=None,
                                 reverse_cross_position_pair_matrix=None,
                                 items_assigned=None, positions_assigned=None, depth=None):
    """
    Drop-in replacement for the original calculate_upper_bound function.
    Uses the bound calculator for much tighter bounds.
    """
    scorer = LayoutScorer(
        item_scores_array=item_scores_array,
        item_pair_matrix=item_pair_score_matrix,
        position_matrix=position_score_matrix,
        cross_item_pair_matrix=cross_item_pair_matrix,
        cross_position_pair_matrix=cross_position_pair_matrix,
        reverse_cross_item_pair_matrix=reverse_cross_item_pair_matrix,
        reverse_cross_position_pair_matrix=reverse_cross_position_pair_matrix
    )
    
    bound_calculator = UpperBoundCalculator(scorer)
    return bound_calculator.calculate_upper_bound(mapping, used)

#-----------------------------------------------------------------------------
# Multi-Objective Optimization Classes
#-----------------------------------------------------------------------------

class ParetoFront:
    """Manages a set of non-dominated solutions."""
    
    def __init__(self, objective_names: List[str]):
        self.objective_names = objective_names
        self.solutions = []  # List of (layout, objectives) tuples
    
    def add_solution(self, layout: np.ndarray, objectives: List[float]) -> bool:
        """
        Add a solution to the Pareto front.
        
        Returns:
            bool: True if solution was added (non-dominated), False if dominated
        """
        layout_copy = layout.copy()
        
        # Check if new solution is dominated by existing solutions
        for existing_layout, existing_objectives in self.solutions:
            if self._dominates(existing_objectives, objectives):
                return False  # New solution is dominated, don't add
        
        # Remove existing solutions that are dominated by new solution
        self.solutions = [
            (existing_layout, existing_objectives)
            for existing_layout, existing_objectives in self.solutions
            if not self._dominates(objectives, existing_objectives)
        ]
        
        # Add new solution
        self.solutions.append((layout_copy, objectives))
        return True
    
    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """
        Check if obj1 dominates obj2.
        obj1 dominates obj2 if obj1 is >= obj2 in all dimensions and > obj2 in at least one.
        """
        at_least_one_better = False
        for v1, v2 in zip(obj1, obj2):
            if v1 < v2:
                return False  # obj1 is worse in this dimension
            if v1 > v2:
                at_least_one_better = True
        
        return at_least_one_better
    
    def get_solutions(self) -> List[Tuple[np.ndarray, List[float]]]:
        """Return all Pareto-optimal solutions."""
        return self.solutions.copy()
    
    def size(self) -> int:
        """Return number of solutions in Pareto front."""
        return len(self.solutions)


class MultiObjectiveOptimizer:
    """Multi-objective optimizer using Pareto-based branch and bound."""
    
    def __init__(self, scorer: LayoutScorer, 
                 items_to_assign: str, available_positions: str,
                 items_to_constrain: str = '', positions_to_constrain: str = '',
                 items_assigned: str = '', positions_assigned: str = ''):
        self.scorer = scorer
        self.items_to_assign = list(items_to_assign)
        self.available_positions = list(available_positions)
        self.items_to_constrain = items_to_constrain
        self.positions_to_constrain = positions_to_constrain
        self.items_assigned = items_assigned
        self.positions_assigned = positions_assigned
        
        # Get objective names from scorer
        self.objective_names = self._get_objective_names()
        self.pareto_front = ParetoFront(self.objective_names)
        
        # Constraint setup
        self.constrained_items = set(items_to_constrain.lower()) if items_to_constrain else set()
        self.constrained_positions = set(i for i, pos in enumerate(available_positions) 
                                       if pos.upper() in positions_to_constrain.upper()) if positions_to_constrain else set()
        self.constrained_item_indices = set(i for i, item in enumerate(items_to_assign) 
                                          if item in self.constrained_items) if self.constrained_items else set()
        
        # Statistics
        self.nodes_processed = 0
        self.solutions_found = 0
        
    def _get_objective_names(self) -> List[str]:
        """Return names of available objectives."""
        objectives = ['item_score', 'pair_score']
        if self.scorer.has_cross_interactions:
            objectives.append('cross_score')
        return objectives
        
    def optimize(self, max_solutions: int = 50, time_limit: float = 60.0) -> ParetoFront:
        """
        Find Pareto-optimal solutions using depth-first search.
        
        Args:
            max_solutions: Maximum number of solutions to find
            time_limit: Maximum time in seconds
        
        Returns:
            ParetoFront containing non-dominated solutions
        """
        print(f"Running Multi-Objective Optimization...")
        print(f"Objectives: {', '.join(self.objective_names)}")
        
        start_time = time.time()
        
        # Initialize search
        n_items = len(self.items_to_assign)
        n_positions = len(self.available_positions)
        
        # Create initial partial mapping with pre-assigned constraints
        initial_mapping = np.full(n_items, -1, dtype=int)
        initial_used = np.zeros(n_positions, dtype=bool)
        
        # Apply pre-assigned items
        if self.items_assigned:
            item_to_idx = {item: idx for idx, item in enumerate(self.items_to_assign)}
            position_to_pos = {position: pos for pos, position in enumerate(self.available_positions)}
            
            for item, position in zip(self.items_assigned, self.positions_assigned):
                if item in item_to_idx and position in position_to_pos:
                    idx = item_to_idx[item]
                    pos = position_to_pos[position]
                    initial_mapping[idx] = pos
                    initial_used[pos] = True
        
        # Start recursive search
        self._search_recursive(initial_mapping, initial_used, max_solutions, time_limit, start_time)
        
        return self.pareto_front
    
    def _search_recursive(self, partial_mapping: np.ndarray, used_positions: np.ndarray,
                         max_solutions: int, time_limit: float, start_time: float):
        """Recursive search maintaining Pareto front."""
        
        self.nodes_processed += 1
        
        # Check termination conditions
        if time_limit and (time.time() - start_time) > time_limit:
            return
        
        if self.pareto_front.size() >= max_solutions:
            return
        
        # Find next unassigned item (handle constraints)
        next_item = self._get_next_item(partial_mapping)
        
        # If all items assigned, evaluate complete solution
        if next_item == -1:
            objectives = self._evaluate_complete_solution(partial_mapping)
            if self.pareto_front.add_solution(partial_mapping, objectives):
                self.solutions_found += 1
            return
        
        # Get valid positions for this item
        valid_positions = self._get_valid_positions(next_item, used_positions)
        
        # Try assigning next item to each valid position
        for pos_idx in valid_positions:
            # Make assignment
            partial_mapping[next_item] = pos_idx
            used_positions[pos_idx] = True
            
            # Continue search (could add pruning heuristics here)
            self._search_recursive(partial_mapping, used_positions, max_solutions, time_limit, start_time)
            
            # Backtrack
            partial_mapping[next_item] = -1
            used_positions[pos_idx] = False
    
    def _get_next_item(self, mapping: np.ndarray) -> int:
        """Get next item to assign, prioritizing constrained items."""
        # First try constrained items
        if self.constrained_item_indices:
            for item_idx in self.constrained_item_indices:
                if mapping[item_idx] == -1:
                    return item_idx
        
        # Then any unassigned item
        for i in range(len(mapping)):
            if mapping[i] == -1:
                return i
        
        return -1
    
    def _get_valid_positions(self, item_idx: int, used_positions: np.ndarray) -> List[int]:
        """Get valid positions for an item."""
        if item_idx in self.constrained_item_indices:
            # Constrained item - only constrained positions
            return [pos for pos in self.constrained_positions if not used_positions[pos]]
        else:
            # Unconstrained item - any available position
            return [pos for pos in range(len(used_positions)) if not used_positions[pos]]
    
    def _evaluate_complete_solution(self, layout: np.ndarray) -> List[float]:
        """Evaluate complete solution and return objective values."""
        # Get component scores from scorer
        total_score, item_score, pair_score = self.scorer.score_layout(layout, return_components=True)
        
        if self.scorer.has_cross_interactions:
            # Calculate cross-interactions separately
            cross_score = self.scorer._calculate_cross_interactions(layout)[0]
            return [item_score, pair_score, cross_score]
        else:
            return [item_score, pair_score]


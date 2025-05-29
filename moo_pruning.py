# moo_pruning.py
"""
Multi-objective pruning optimization for keyboard layout search.
"""

import numpy as np
from typing import List, Tuple, Set
from numba import jit

class MOOPruner:
    """Handles multi-objective pruning calculations."""
    
    def __init__(self, norm_item_scores: dict, norm_item_pair_scores: dict, 
                 norm_position_scores: dict, norm_position_pair_scores: dict,
                 items_list: List[str], positions_list: List[str]):
        """
        Initialize pruner with normalized score dictionaries.
        
        Args:
            norm_item_scores: {item: score} - flat structure
            norm_item_pair_scores: {(item1, item2): score}
            norm_position_scores: {position: score}
            norm_position_pair_scores: {(pos1, pos2): score}
            items_list: List of items in order used by mapping arrays
            positions_list: List of positions in order used by mapping arrays
        """
        self.norm_item_scores = norm_item_scores
        self.norm_item_pair_scores = norm_item_pair_scores
        self.norm_position_scores = norm_position_scores
        self.norm_position_pair_scores = norm_position_pair_scores
        self.items_list = items_list
        self.positions_list = positions_list
        
        # Pre-compute max scores for faster lookups
        self._precompute_max_scores()
    
    def _precompute_max_scores(self):
        """Pre-compute maximum possible scores for efficiency."""
        # Max item score for each item (item scores are flat: {item: score})
        self.max_item_scores = {}
        for item in self.items_list:
            if item in self.norm_item_scores:
                self.max_item_scores[item] = self.norm_item_scores[item]
            else:
                self.max_item_scores[item] = 0.0
        
        # Max position score across all positions
        self.max_position_score = max(self.norm_position_scores.values()) if self.norm_position_scores else 0.0
        
        # Max item-pair score (used for conservative estimates)
        self.max_item_pair_score = max(self.norm_item_pair_scores.values()) if self.norm_item_pair_scores else 0.0
        
        # Max position-pair score
        self.max_position_pair_score = max(self.norm_position_pair_scores.values()) if self.norm_position_pair_scores else 0.0
    
    def calculate_upper_bounds(self, mapping: np.ndarray, used_positions: np.ndarray) -> List[float]:
        """
        Calculate upper bounds for each objective given current partial mapping.
        
        Args:
            mapping: Current item->position mapping (-1 for unassigned)
            used_positions: Boolean array of used positions
            
        Returns:
            List of [item_upper_bound, item_pair_upper_bound]
        """
        # Find unassigned items and available positions
        unassigned_items = [i for i, pos in enumerate(mapping) if pos == -1]
        available_positions = [i for i, used in enumerate(used_positions) if not used]
        
        if not unassigned_items:
            return [0.0, 0.0]  # No more items to assign
        
        # Calculate current partial scores for assigned items
        current_item_score, current_pair_score = self._calculate_partial_scores(mapping)
        
        # Upper bound for item component
        item_upper = current_item_score + self._calculate_item_upper_bound(unassigned_items, available_positions)
        
        # Upper bound for item-pair component  
        pair_upper = current_pair_score + self._calculate_pair_upper_bound(mapping, unassigned_items, available_positions)
        
        return [item_upper, pair_upper]
    
    def _calculate_partial_scores(self, mapping: np.ndarray) -> Tuple[float, float]:
        """Calculate current scores for assigned items only."""
        item_score = 0.0
        pair_score = 0.0
        
        # Item scores (item scores are flat: {item: score})
        for item_idx, pos_idx in enumerate(mapping):
            if pos_idx >= 0:  # Item is assigned
                item = self.items_list[item_idx]
                position = self.positions_list[pos_idx]
                
                # Add item score (flat structure)
                if item in self.norm_item_scores:
                    item_score += self.norm_item_scores[item]
                
                # Add position score
                if position in self.norm_position_scores:
                    item_score += self.norm_position_scores[position]
        
        # Item-pair and position-pair scores
        assigned_items = [(i, pos) for i, pos in enumerate(mapping) if pos >= 0]
        
        for i, (item1_idx, pos1_idx) in enumerate(assigned_items):
            for item2_idx, pos2_idx in assigned_items[i+1:]:
                item1 = self.items_list[item1_idx]
                item2 = self.items_list[item2_idx]
                pos1 = self.positions_list[pos1_idx]
                pos2 = self.positions_list[pos2_idx]
                
                # Item-pair score
                pair_key = tuple(sorted([item1, item2]))
                if pair_key in self.norm_item_pair_scores:
                    pair_score += self.norm_item_pair_scores[pair_key]
                
                # Position-pair score
                pos_pair_key = tuple(sorted([pos1, pos2]))
                if pos_pair_key in self.norm_position_pair_scores:
                    pair_score += self.norm_position_pair_scores[pos_pair_key]
        
        return item_score, pair_score
    
    def _calculate_item_upper_bound(self, unassigned_items: List[int], 
                                  available_positions: List[int]) -> float:
        """Calculate upper bound for item component."""
        if not unassigned_items or not available_positions:
            return 0.0
        
        upper_bound = 0.0
        
        # For each unassigned item, add its score plus best available position score
        for item_idx in unassigned_items:
            item = self.items_list[item_idx]
            
            # Item score (flat structure: {item: score})
            item_contrib = 0.0
            if item in self.norm_item_scores:
                item_contrib += self.norm_item_scores[item]
            
            # Add best available position score
            if available_positions and self.norm_position_scores:
                best_position_score = max(
                    self.norm_position_scores.get(self.positions_list[pos_idx], 0.0)
                    for pos_idx in available_positions
                )
                item_contrib += best_position_score
            
            upper_bound += item_contrib
        
        return upper_bound
    
    def _calculate_pair_upper_bound(self, mapping: np.ndarray, unassigned_items: List[int], 
                                  available_positions: List[int]) -> float:
        """Calculate upper bound for pair component (conservative estimate)."""
        if len(unassigned_items) < 2:
            return 0.0
        
        # Conservative approach: assume all unassigned pairs get maximum pair score
        n_unassigned = len(unassigned_items)
        n_new_pairs = (n_unassigned * (n_unassigned - 1)) // 2
        
        # Also need to account for pairs between assigned and unassigned items
        n_assigned = len([pos for pos in mapping if pos >= 0])
        n_mixed_pairs = n_assigned * n_unassigned
        
        # Use maximum pair scores as conservative upper bound
        upper_bound = (n_new_pairs + n_mixed_pairs) * (self.max_item_pair_score + self.max_position_pair_score)
        
        return upper_bound
    
    def can_prune_branch(self, mapping: np.ndarray, used_positions: np.ndarray, 
                        pareto_objectives: List[List[float]]) -> bool:
        """
        Check if current branch can be pruned based on upper bounds.

        Only prune if ALL existing solutions dominate the upper bounds. 
        This ensures you only eliminate branches that definitely cannot 
        improve the Pareto front.
        
        Args:
            mapping: Current partial mapping
            used_positions: Used position flags
            pareto_objectives: Current Pareto front objectives
            
        Returns:
            True if branch can be pruned, False otherwise
        """
        if not pareto_objectives:
            return False  # No solutions to compare against yet
        
        # Calculate upper bounds for this branch
        upper_bounds = self.calculate_upper_bounds(mapping, used_positions)
        
        # ✅ FIXED: Only prune if ALL existing solutions dominate upper bounds
        for existing_objectives in pareto_objectives:
            if not self._dominates(existing_objectives, upper_bounds):
                return False  # ✅ Conservative - don't prune if any solution doesn't dominate
        
        return True  # All solutions dominate upper bounds - safe to prune
    
    @staticmethod
    def _dominates(obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2 (assumes higher is better)."""
        if len(obj1) != len(obj2):
            return False
        
        # obj1 dominates obj2 if obj1 is >= obj2 in all objectives 
        # and > obj2 in at least one objective
        all_geq = all(obj1[i] >= obj2[i] for i in range(len(obj1)))
        any_greater = any(obj1[i] > obj2[i] for i in range(len(obj1)))
        
        return all_geq and any_greater


def create_moo_pruner(normalized_scores: Tuple, items_list: List[str], 
                     positions_list: List[str]) -> MOOPruner:
    """
    Create MOO pruner from normalized scores tuple.
    
    Args:
        normalized_scores: (item_scores, item_pair_scores, position_scores, position_pair_scores)
        items_list: List of items in mapping order
        positions_list: List of positions in mapping order
        
    Returns:
        Configured MOOPruner instance
    """
    item_scores, item_pair_scores, position_scores, position_pair_scores = normalized_scores
    
    return MOOPruner(
        norm_item_scores=item_scores,
        norm_item_pair_scores=item_pair_scores, 
        norm_position_scores=position_scores,
        norm_position_pair_scores=position_pair_scores,
        items_list=items_list,
        positions_list=positions_list
    )

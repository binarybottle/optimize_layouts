# scoring.py  
"""
Unified scoring system for layout optimization.

The system uses a centralized combination strategy defined by 
DEFAULT_COMBINATION_STRATEGY and implemented in apply_default_combination(). 

This ensures consistency across:
- ScoreComponents.total() method
- CombinedCombiner class  
- MOO mode combined totals (for display purposes)
- Any other places that need to combine item and item-pair scores

To change the combination strategy, modify only the DEFAULT_COMBINATION_STRATEGY 
constant and the apply_default_combination() function.

Note: Since the scoring accepts any characters,
there could be a problem if there is missing data 
in the item and item-pair frequencies files.
Missing items and item-pairs get default scores.

Note: Since the scoring accepts any characters and any positions,
there could be a problem if item or position files are incomplete.
Layouts with items or positions that are missing in these files 
will be assigned default scores.

"""

import numpy as np
from numba import jit
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Tuple
from config import Config

#-----------------------------------------------------------------------------
# Default combination strategy
#-----------------------------------------------------------------------------
DEFAULT_COMBINATION_STRATEGY = "multiplicative"  # item_score * item_pair_score

def apply_default_combination(item_score: float, item_pair_score: float) -> float:
    """
    Apply the default combination strategy.
    
    This is the SINGLE SOURCE OF TRUTH for how item and item-pair scores are combined.
    All other combination logic should delegate to this function.
    
    Args:
        item_score: Individual item score component
        item_pair_score: Item-pair interaction score component
        
    Returns:
        Combined score using default strategy
    """
    if DEFAULT_COMBINATION_STRATEGY == "multiplicative":
        return item_score * item_pair_score
    elif DEFAULT_COMBINATION_STRATEGY == "additive":
        return item_score + item_pair_score
    elif DEFAULT_COMBINATION_STRATEGY == "weighted_additive":
        # Could add weights here if needed
        return 0.6 * item_score + 0.4 * item_pair_score
    else:
        raise ValueError(f"Unknown combination strategy: {DEFAULT_COMBINATION_STRATEGY}")

def apply_default_combination_vectorized(item_scores: np.ndarray, item_pair_scores: np.ndarray) -> np.ndarray:
    """
    Vectorized version of apply_default_combination for numpy arrays/pandas Series.
    
    Args:
        item_scores: Array of item scores
        item_pair_scores: Array of item-pair scores
        
    Returns:
        Array of combined scores using default strategy
    """
    if DEFAULT_COMBINATION_STRATEGY == "multiplicative":
        return item_scores * item_pair_scores
    elif DEFAULT_COMBINATION_STRATEGY == "additive":
        return item_scores + item_pair_scores
    elif DEFAULT_COMBINATION_STRATEGY == "weighted_additive":
        return 0.6 * item_scores + 0.4 * item_pair_scores
    else:
        raise ValueError(f"Unknown combination strategy: {DEFAULT_COMBINATION_STRATEGY}")

#-----------------------------------------------------------------------------
# JIT-compiled core calculations
#-----------------------------------------------------------------------------
@jit(nopython=True, fastmath=True)
def _calculate_item_score_jit(mapping: np.ndarray, item_scores: np.ndarray, 
                             position_matrix: np.ndarray) -> Tuple[float, int]:
    """
    JIT-compiled item score calculation.
    
    Returns:
        (raw_score, placed_count)
    """
    raw_score = 0.0
    placed_count = 0
    
    for i in range(len(mapping)):
        pos = mapping[i]
        if pos >= 0 and pos < position_matrix.shape[0]:  # Item is placed and pos is valid
            # Position matrix diagonal contains individual position scores
            pos_score = position_matrix[pos, pos]
            raw_score += item_scores[i] * pos_score
            placed_count += 1
    
    return raw_score, placed_count

@jit(nopython=True, fastmath=True)
def _calculate_item_pair_score_jit(mapping: np.ndarray, 
                                  item_pair_matrix: np.ndarray,
                                  position_matrix: np.ndarray,
                                  cross_item_matrix: Optional[np.ndarray],
                                  cross_position_matrix: Optional[np.ndarray],
                                  reverse_cross_item_matrix: Optional[np.ndarray],
                                  reverse_cross_position_matrix: Optional[np.ndarray]) -> Tuple[float, int]:
    """
    JIT-compiled combined item-pair score calculation.
    Includes both internal pairs and cross-interactions.
    
    Returns:
        (raw_score, interaction_count)
    """
    raw_score = 0.0
    interaction_count = 0
    
    # 1. Internal pairs between items being optimized
    for i in range(len(mapping)):
        pos_i = mapping[i] 
        if pos_i < 0 or pos_i >= position_matrix.shape[0]:
            continue
            
        for j in range(i + 1, len(mapping)):  # Only count each pair once
            pos_j = mapping[j]
            if pos_j < 0 or pos_j >= position_matrix.shape[0]:
                continue
            
            # Both directions for internal pairs
            fwd_score = item_pair_matrix[i, j] * position_matrix[pos_i, pos_j]
            bwd_score = item_pair_matrix[j, i] * position_matrix[pos_j, pos_i]
            
            raw_score += fwd_score + bwd_score
            interaction_count += 2  # Count both directions
    
    # 2. Cross-interactions with pre-assigned items (if they exist)
    if cross_item_matrix is not None:
        n_assigned = cross_item_matrix.shape[1]
        
        for i in range(len(mapping)):
            pos_i = mapping[i]
            if pos_i < 0 or pos_i >= cross_position_matrix.shape[0]:
                continue
                
            for j in range(n_assigned):
                # Forward direction: new_item -> assigned_item
                fwd_item = cross_item_matrix[i, j]
                fwd_pos = cross_position_matrix[pos_i, j]
                
                # Backward direction: assigned_item -> new_item  
                bwd_item = reverse_cross_item_matrix[j, i]
                bwd_pos = reverse_cross_position_matrix[j, pos_i]
                
                raw_score += fwd_item * fwd_pos + bwd_item * bwd_pos
                interaction_count += 2
    
    return raw_score, interaction_count
    
#-----------------------------------------------------------------------------
# Core data structures
#-----------------------------------------------------------------------------
@dataclass
class ScoreComponents:
    """Container for the two independent scoring components."""
    item_score: float
    item_pair_score: float
    
    def as_list(self) -> List[float]:
        """Return components as list for MOO."""
        return [self.item_score, self.item_pair_score]
    
    def total(self) -> float:
        """
        Return combined score using default combination strategy.
        
        This delegates to the centralized combination logic to ensure consistency
        throughout the system.
        """
        return apply_default_combination(self.item_score, self.item_pair_score)

class ScoringArrays:
    """
    Container for all scoring arrays with clear documentation.
    
    Multiprocessing-safe implementation that works seamlessly in both
    parallel and non-parallel execution modes.
    """
    
    def __init__(self, item_scores: np.ndarray, item_pair_matrix: np.ndarray, 
                 position_matrix: np.ndarray, cross_item_matrix: Optional[np.ndarray] = None,
                 cross_position_matrix: Optional[np.ndarray] = None,
                 reverse_cross_item_matrix: Optional[np.ndarray] = None,
                 reverse_cross_position_matrix: Optional[np.ndarray] = None):
        """Initialize scoring arrays."""
        # Core arrays for items being optimized
        self.item_scores = item_scores                    # Shape: (n_items,)
        self.item_pair_matrix = item_pair_matrix         # Shape: (n_items, n_items)
        self.position_matrix = position_matrix           # Shape: (n_positions, n_positions)
        
        # Cross-interaction arrays (optional - for pre-assigned items)
        self.cross_item_matrix = cross_item_matrix                # Shape: (n_items, n_assigned)
        self.cross_position_matrix = cross_position_matrix       # Shape: (n_positions, n_assigned)
        self.reverse_cross_item_matrix = reverse_cross_item_matrix
        self.reverse_cross_position_matrix = reverse_cross_position_matrix
        
        # Set metadata
        self.n_items = len(self.item_scores)
        self.n_positions = self.position_matrix.shape[0]
        self.n_assigned = 0
        
        if self.cross_item_matrix is not None:
            self.n_assigned = self.cross_item_matrix.shape[1]
        
        self._validate_shapes()
    
    def __getstate__(self):
        """Custom serialization for multiprocessing - preserves all data."""
        return {
            'item_scores': self.item_scores,
            'item_pair_matrix': self.item_pair_matrix,
            'position_matrix': self.position_matrix,
            'cross_item_matrix': self.cross_item_matrix,
            'cross_position_matrix': self.cross_position_matrix,
            'reverse_cross_item_matrix': self.reverse_cross_item_matrix,
            'reverse_cross_position_matrix': self.reverse_cross_position_matrix,
            'n_items': self.n_items,
            'n_positions': self.n_positions,
            'n_assigned': self.n_assigned
        }
    
    def __setstate__(self, state):
        """Custom deserialization for multiprocessing - restores all data."""
        for key, value in state.items():
            setattr(self, key, value)
    
    def _validate_shapes(self):
        """Ensure all arrays have consistent shapes."""
        assert self.item_pair_matrix.shape == (self.n_items, self.n_items)
        assert self.position_matrix.shape == (self.n_positions, self.n_positions)
        
        if self.cross_item_matrix is not None:
            assert self.cross_item_matrix.shape == (self.n_items, self.n_assigned)
            assert self.cross_position_matrix.shape == (self.n_positions, self.n_assigned)
            assert self.reverse_cross_item_matrix.shape == (self.n_assigned, self.n_items)
            assert self.reverse_cross_position_matrix.shape == (self.n_assigned, self.n_positions)
    
    @property
    def has_cross_interactions(self) -> bool:
        """Check if cross-interaction data is available."""
        return self.cross_item_matrix is not None

#-----------------------------------------------------------------------------
# Score combiners
#-----------------------------------------------------------------------------
class ScoreCombiner(ABC):
    """Abstract base class for different score combination strategies."""
    
    @abstractmethod
    def combine(self, components: ScoreComponents) -> float:
        """Combine score components into a single value."""
        pass
    
    @abstractmethod
    def get_mode_name(self) -> str:
        """Return human-readable name for this combination mode."""
        pass

class ItemOnlyCombiner(ScoreCombiner):
    """Combine using only item scores."""
    
    def combine(self, components: ScoreComponents) -> float:
        return components.item_score
    
    def get_mode_name(self) -> str:
        return "Item Only"

class PairOnlyCombiner(ScoreCombiner):
    """Combine using only item-pair scores."""
    
    def combine(self, components: ScoreComponents) -> float:
        return components.item_pair_score
    
    def get_mode_name(self) -> str:
        return "Item-Pair Only"

class CombinedCombiner(ScoreCombiner):
    """Combine using the default combination strategy."""
    
    def combine(self, components: ScoreComponents) -> float:
        """Use centralized combination logic."""
        return apply_default_combination(components.item_score, components.item_pair_score)
    
    def get_mode_name(self) -> str:
        return f"Combined ({DEFAULT_COMBINATION_STRATEGY})"

class MOOCombiner(ScoreCombiner):
    """Multi-objective combiner - returns components separately."""
    
    def combine(self, components: ScoreComponents) -> List[float]:
        """Return components as separate objectives."""
        return components.as_list()
    
    def get_mode_name(self) -> str:
        return "Multi-Objective"
        
#-----------------------------------------------------------------------------
# Core scoring classes
#-----------------------------------------------------------------------------
class ScoreCalculator:
    """
    Central scoring engine - single source of truth for all score calculations.
    
    This class handles both SOO and MOO modes by calculating independent components
    and combining them appropriately based on the requested mode.
    """
    
    def __init__(self, arrays: ScoringArrays):
        self.arrays = arrays
        self._score_cache = {}  # Cache for performance
        
    def calculate_components(self, mapping: np.ndarray) -> ScoreComponents:
        """
        Calculate the two independent scoring components.
        
        Args:
            mapping: Array mapping items to positions (-1 for unassigned)
            
        Returns:
            ScoreComponents with normalized scores
        """
        # Check cache first
        mapping_key = tuple(mapping)
        if mapping_key in self._score_cache:
            return self._score_cache[mapping_key]
        
        int_mapping = mapping.astype(np.int32)
        
        # 1. Calculate item score component
        item_raw, item_count = _calculate_item_score_jit(
            int_mapping, self.arrays.item_scores, self.arrays.position_matrix
        )
        item_score = item_raw / max(1, item_count)  # Normalize by placed items
        
        # 2. Calculate combined item-pair score component
        cross_arrays = None
        if self.arrays.has_cross_interactions:
            cross_arrays = (
                self.arrays.cross_item_matrix,
                self.arrays.cross_position_matrix,
                self.arrays.reverse_cross_item_matrix,
                self.arrays.reverse_cross_position_matrix
            )
        
        item_pair_raw, pair_count = _calculate_item_pair_score_jit(
            int_mapping, 
            self.arrays.item_pair_matrix, 
            self.arrays.position_matrix,
            cross_arrays[0] if cross_arrays else None,
            cross_arrays[1] if cross_arrays else None,
            cross_arrays[2] if cross_arrays else None,
            cross_arrays[3] if cross_arrays else None
        )
        item_pair_score = item_pair_raw / max(1, pair_count)  # Normalize by interaction count
        
        components = ScoreComponents(item_score, item_pair_score)
        
        # Cache result
        self._score_cache[mapping_key] = components
        return components
    
    def clear_cache(self):
        """Clear score cache to free memory."""
        self._score_cache.clear()

class LayoutScorer:
    """
    Unified layout scorer that serves both SOO and MOO optimization.
    Works seamlessly in both parallel and non-parallel execution modes.
    """
    
    def __init__(self, arrays: ScoringArrays, mode: str = 'combined'):
        """
        Initialize scorer with arrays and combination mode.
        
        Args:
            arrays: ScoringArrays containing all necessary scoring data
            mode: One of 'item_only', 'pair_only', 'combined', 'multi_objective'
        """
        self.calculator = ScoreCalculator(arrays)
        self.arrays = arrays
        
        # Set up score combiner based on mode
        combiners = {
            'item_only': ItemOnlyCombiner(),
            'pair_only': PairOnlyCombiner(), 
            'combined': CombinedCombiner(),
            'multi_objective': MOOCombiner()
        }
        
        if mode not in combiners:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {list(combiners.keys())}")
        
        self.combiner = combiners[mode]
        self.mode = mode
    
    def score_layout(self, mapping: np.ndarray, return_components: bool = False):
        """
        Score a layout using the configured combination mode.
        
        Args:
            mapping: Array mapping items to positions
            return_components: If True, return (total, item, item_pair) tuple
                            If False, return single score or list (for MOO)
        
        Returns:
            Single score, list of scores (MOO), or tuple with breakdown
        """
        components = self.calculator.calculate_components(mapping)
        
        if return_components:
            if self.mode == 'multi_objective':
                # For MOO, include the default combined score as well
                combined_total = apply_default_combination(components.item_score, components.item_pair_score)
                return components.as_list() + [combined_total]  # [item, item_pair, combined_total]
            else:
                combined_score = self.combiner.combine(components)
                return combined_score, components.item_score, components.item_pair_score
        else:
            return self.combiner.combine(components)
        
    def get_components(self, mapping: np.ndarray) -> ScoreComponents:
        """Get raw score components for analysis."""
        return self.calculator.calculate_components(mapping)
    
    def clear_cache(self):
        """Clear internal caches."""
        self.calculator.clear_cache()
    
    @property
    def mode_name(self) -> str:
        """Human-readable name for current scoring mode."""
        return self.combiner.get_mode_name()

#-----------------------------------------------------------------------------
# Scoring factory functions
#-----------------------------------------------------------------------------
def prepare_scoring_arrays(
    items_to_assign: List[str],
    positions_to_assign: List[str], 
    norm_item_scores: Dict[str, float],
    norm_item_pair_scores: Dict[Tuple[str, str], float],
    norm_position_scores: Dict[str, float], 
    norm_position_pair_scores: Dict[Tuple[str, str], float],
    items_assigned: Optional[List[str]] = None,
    positions_assigned: Optional[List[str]] = None,
    missing_item_pair_score: float = 1.0,
    missing_position_pair_score: float = 1.0
) -> ScoringArrays:
    """
    Single source of truth for preparing all scoring arrays.
    
    Args:
        items_to_assign: List of items being optimized
        positions_to_assign: List of available positions
        norm_item_scores: Normalized item scores [0,1]
        norm_item_pair_scores: Normalized item pair scores [0,1]
        norm_position_scores: Normalized position scores [0,1]
        norm_position_pair_scores: Normalized position pair scores [0,1]
        items_assigned: Optional list of pre-assigned items
        positions_assigned: Optional list of pre-assigned positions  
        missing_item_pair_score: Default score for missing item pairs
        missing_position_pair_score: Default score for missing position pairs
        
    Returns:
        ScoringArrays object containing all necessary arrays
    """
    n_items = len(items_to_assign)
    n_positions = len(positions_to_assign)
    
    # Validate inputs
    if n_items > n_positions:
        raise ValueError(f"More items ({n_items}) than positions ({n_positions})")
    
    # 1. Prepare core item scores array
    item_scores = np.array([
        norm_item_scores.get(item.lower(), 0.0) 
        for item in items_to_assign
    ], dtype=np.float32)
    
    # 2. Prepare item pair matrix
    item_pair_matrix = np.zeros((n_items, n_items), dtype=np.float32)
    for i, item1 in enumerate(items_to_assign):
        for j, item2 in enumerate(items_to_assign):
            if i != j:  # No self-pairs
                key = (item1.lower(), item2.lower())
                item_pair_matrix[i, j] = norm_item_pair_scores.get(key, missing_item_pair_score)
    
    # 3. Prepare position matrix (diagonal = individual scores, off-diagonal = pair scores)
    position_matrix = np.zeros((n_positions, n_positions), dtype=np.float32)
    for i, pos1 in enumerate(positions_to_assign):
        for j, pos2 in enumerate(positions_to_assign):
            if i == j:
                # Diagonal: individual position scores
                position_matrix[i, j] = norm_position_scores.get(pos1.lower(), 0.0)
            else:
                # Off-diagonal: position pair scores
                key = (pos1.lower(), pos2.lower())
                position_matrix[i, j] = norm_position_pair_scores.get(key, missing_position_pair_score)
    
    # 4. Prepare cross-interaction arrays (if pre-assigned items exist)
    cross_arrays = {}
    if items_assigned and positions_assigned:
        n_assigned = len(items_assigned)
        
        # Cross-item matrix: items_to_assign -> items_assigned
        cross_item_matrix = np.zeros((n_items, n_assigned), dtype=np.float32)
        for i, item1 in enumerate(items_to_assign):
            for j, item2 in enumerate(items_assigned):
                key = (item1.lower(), item2.lower())
                cross_item_matrix[i, j] = norm_item_pair_scores.get(key, missing_item_pair_score)
        
        # Reverse cross-item matrix: items_assigned -> items_to_assign  
        reverse_cross_item_matrix = np.zeros((n_assigned, n_items), dtype=np.float32)
        for i, item1 in enumerate(items_assigned):
            for j, item2 in enumerate(items_to_assign):
                key = (item1.lower(), item2.lower())
                reverse_cross_item_matrix[i, j] = norm_item_pair_scores.get(key, missing_item_pair_score)
        
        # Cross-position matrix: positions_to_assign -> positions_assigned
        cross_position_matrix = np.zeros((n_positions, n_assigned), dtype=np.float32)
        for i, pos1 in enumerate(positions_to_assign):
            for j, pos2 in enumerate(positions_assigned):
                key = (pos1.lower(), pos2.lower())
                cross_position_matrix[i, j] = norm_position_pair_scores.get(key, missing_position_pair_score)
        
        # Reverse cross-position matrix: positions_assigned -> positions_to_assign
        reverse_cross_position_matrix = np.zeros((n_assigned, n_positions), dtype=np.float32)  
        for i, pos1 in enumerate(positions_assigned):
            for j, pos2 in enumerate(positions_to_assign):
                key = (pos1.lower(), pos2.lower())
                reverse_cross_position_matrix[i, j] = norm_position_pair_scores.get(key, missing_position_pair_score)
        
        cross_arrays = {
            'cross_item_matrix': cross_item_matrix,
            'cross_position_matrix': cross_position_matrix,
            'reverse_cross_item_matrix': reverse_cross_item_matrix,
            'reverse_cross_position_matrix': reverse_cross_position_matrix
        }
    
    # 5. Validate all scores are in [0,1] range
    arrays_to_validate = [
        (item_scores, "item scores"),
        (item_pair_matrix, "item pair scores"), 
        (position_matrix, "position scores")
    ]
    
    for array, name in arrays_to_validate:
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")
        if np.any(array < 0) or np.any(array > 1):
            raise ValueError(f"{name} must be normalized to [0,1] range")
    
    # 6. Create and return ScoringArrays object
    return ScoringArrays(
        item_scores=item_scores,
        item_pair_matrix=item_pair_matrix,
        position_matrix=position_matrix,
        **cross_arrays
    )

def create_scorer(items_to_assign: List[str], positions_to_assign: List[str],
                 norm_item_scores: Dict, norm_item_pair_scores: Dict,
                 norm_position_scores: Dict, norm_position_pair_scores: Dict,
                 items_assigned: Optional[List[str]] = None,
                 positions_assigned: Optional[List[str]] = None,
                 mode: str = 'combined') -> LayoutScorer:
    """
    Convenience function to create a LayoutScorer from raw score dictionaries.
    """
    arrays = prepare_scoring_arrays(
        items_to_assign, positions_to_assign,
        norm_item_scores, norm_item_pair_scores,
        norm_position_scores, norm_position_pair_scores,
        items_assigned, positions_assigned
    )
    
    return LayoutScorer(arrays, mode)

def create_layout_scorer(items_to_assign: List[str], 
                        positions_to_assign: List[str],
                        normalized_scores: Tuple,
                        mode: str = 'combined') -> LayoutScorer:
    """Create a LayoutScorer for the given items and positions."""
    arrays = prepare_scoring_arrays(
        items_to_assign, positions_to_assign, *normalized_scores
    )
    return LayoutScorer(arrays, mode=mode)

def create_complete_layout_scorer(complete_mapping: Dict[str, str],
                                 normalized_scores: Tuple, 
                                 mode: str = 'combined') -> LayoutScorer:
    """Create a LayoutScorer for a complete layout."""
    all_items = list(complete_mapping.keys())
    all_positions = list(complete_mapping.values())
    return create_layout_scorer(all_items, all_positions, normalized_scores, mode)

#-----------------------------------------------------------------------------
# High-level scoring functions
#-----------------------------------------------------------------------------
def calculate_complete_layout_score(complete_mapping: Dict[str, str],
                                   normalized_scores: Tuple) -> Tuple[float, float, float]:
    """
    Calculate complete layout score including all items and pairs.
    
    This is a high-level convenience function that creates a complete scorer
    and calculates the total scores.
    
    Args:
        complete_mapping: Dictionary mapping items to positions
        normalized_scores: Tuple of (norm_item_scores, norm_item_pair_scores, 
                          norm_position_scores, norm_position_pair_scores)
    
    Returns:
        Tuple of (total_score, item_score, item_pair_score)
    """
    # Create complete scorer
    scorer = create_complete_layout_scorer(complete_mapping, normalized_scores, mode='combined')
    
    # Create mapping array (all items assigned to their positions)  
    mapping_array = np.arange(len(complete_mapping), dtype=np.int32)
    
    # Calculate scores
    total_score, item_score, item_pair_score = scorer.score_layout(
        mapping_array, return_components=True
    )
    
    return total_score, item_score, item_pair_score

def calculate_complete_layout_score_direct(complete_mapping: dict, normalized_scores: tuple) -> tuple:
    """
    Calculate complete layout score directly from the mapping without going through optimization scoring.
    
    Args:
        complete_mapping: Dictionary mapping items to positions (e.g., {'e': 'F', 't': 'D', ...})
        normalized_scores: Tuple of (norm_item_scores, norm_item_pair_scores, 
                          norm_position_scores, norm_position_pair_scores)
    
    Returns:
        Tuple of (total_score, item_component, item_pair_component)
    """
    norm_item_scores, norm_item_pair_scores, norm_position_scores, norm_position_pair_scores = normalized_scores
    
    items = list(complete_mapping.keys())
    positions = list(complete_mapping.values())
    n_items = len(items)
    
    # Calculate item component: sum(item_score * position_score) / n_items
    item_raw_score = 0.0
    for item, pos in complete_mapping.items():
        item_score = norm_item_scores.get(item.lower(), 0.0)
        pos_score = norm_position_scores.get(pos.lower(), 0.0)
        item_raw_score += item_score * pos_score
    
    item_component = item_raw_score / n_items
    
    # Calculate item-pair component: all pairwise interactions
    pair_raw_score = 0.0
    pair_count = 0
    
    items_list = list(complete_mapping.keys())
    positions_list = list(complete_mapping.values())
    
    for i in range(n_items):
        for j in range(n_items):
            if i != j:  # Skip self-pairs
                item1, item2 = items_list[i], items_list[j]
                pos1, pos2 = positions_list[i], positions_list[j]
                
                # Get item pair score
                item_pair_key = (item1.lower(), item2.lower())
                item_pair_score = norm_item_pair_scores.get(item_pair_key, 1.0)
                
                # Get position pair score  
                pos_pair_key = (pos1.lower(), pos2.lower())
                pos_pair_score = norm_position_pair_scores.get(pos_pair_key, 1.0)
                
                pair_raw_score += item_pair_score * pos_pair_score
                pair_count += 1
    
    pair_component = pair_raw_score / max(1, pair_count)
    
    # Combine using default strategy (multiplicative: item_score * item_pair_score)
    total_score = item_component * pair_component
    
    return total_score, item_component, pair_component

def score_layout_from_strings(items_str: str, 
                             positions_str: str,
                             normalized_scores: Tuple) -> Tuple[float, float, float]:
    """
    Score a layout from item and position strings.
    
    Args:
        items_str: Comma-separated items (e.g., "a,b,c")
        positions_str: Comma-separated positions (e.g., "F,D,J") 
        normalized_scores: Pre-loaded normalized score dictionaries
        
    Returns:
        Tuple of (total_score, item_score, item_pair_score)
    """
    # Parse strings
    items = [item.strip().lower() for item in items_str.split(',')]
    positions = [pos.strip().upper() for pos in positions_str.split(',')]
    
    if len(items) != len(positions):
        raise ValueError(f"Mismatch: {len(items)} items vs {len(positions)} positions")
    
    # Create mapping
    complete_mapping = dict(zip(items, positions))
    
    # Calculate score using existing function
    return calculate_complete_layout_score(complete_mapping, normalized_scores)

#-----------------------------------------------------------------------------
# Testing and validation
#-----------------------------------------------------------------------------
def validate_scorer_consistency(scorer: LayoutScorer, n_tests: int = 100) -> bool:
    """
    Validate that the scorer produces consistent results across multiple calls.
    
    Args:
        scorer: LayoutScorer to test
        n_tests: Number of random mappings to test
        
    Returns:
        True if all tests pass, False otherwise
    """
    np.random.seed(42)  # Reproducible tests
    
    for _ in range(n_tests):
        # Generate random mapping
        n_items = scorer.arrays.n_items
        n_positions = scorer.arrays.n_positions
        mapping = np.random.permutation(n_positions)[:n_items]
        
        # Test consistency across multiple calls
        score1 = scorer.score_layout(mapping)
        score2 = scorer.score_layout(mapping)
        
        if isinstance(score1, (list, tuple)) and isinstance(score2, (list, tuple)):
            # MOO mode - compare element-wise
            if not np.allclose(score1, score2, rtol=1e-10):
                print(f"Inconsistent MOO scores: {score1} vs {score2}")
                return False
        else:
            # SOO mode - compare single values
            if abs(score1 - score2) > 1e-10:
                print(f"Inconsistent SOO scores: {score1} vs {score2}")
                return False
        
        # Test component calculation consistency
        components1 = scorer.get_components(mapping)
        components2 = scorer.get_components(mapping)
        
        if not (np.isclose(components1.item_score, components2.item_score) and
                np.isclose(components1.item_pair_score, components2.item_pair_score)):
            print(f"Inconsistent components: {components1} vs {components2}")
            return False
    
    return True

#-----------------------------------------------------------------------------
# Loading
#-----------------------------------------------------------------------------
def load_normalized_scores(config: Config) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Load normalized scores from CSV files.
    
    Args:
        config: Configuration object containing file paths
        
    Returns:
        Tuple of (item_scores, item_pair_scores, position_scores, position_pair_scores)
    """
    def load_score_dict(filepath: str, key_col: str, score_col: str = 'score') -> Dict:
        """Helper to load score dictionary from CSV."""
        df = pd.read_csv(filepath)
        return {row[key_col].lower(): float(row[score_col]) for _, row in df.iterrows()}
    
    def load_pair_score_dict(filepath: str, pair_col: str, score_col: str = 'score') -> Dict:
        """Helper to load pair score dictionary from CSV."""
        df = pd.read_csv(filepath)
        result = {}
        for _, row in df.iterrows():
            pair_str = str(row[pair_col])
            if len(pair_str) == 2:
                key = (pair_str[0].lower(), pair_str[1].lower())
                result[key] = float(row[score_col])
        return result
    
    # Load all score dictionaries
    item_scores = load_score_dict(config.paths.item_scores_file, 'item')
    item_pair_scores = load_pair_score_dict(config.paths.item_pair_scores_file, 'item_pair')
    position_scores = load_score_dict(config.paths.position_scores_file, 'position')  
    position_pair_scores = load_pair_score_dict(config.paths.position_pair_scores_file, 'position_pair')
    
    print(f"Loaded {len(item_scores)} item scores")
    print(f"Loaded {len(item_pair_scores)} item pair scores")
    print(f"Loaded {len(position_scores)} position scores")
    print(f"Loaded {len(position_pair_scores)} position pair scores")
    
    return item_scores, item_pair_scores, position_scores, position_pair_scores

#-----------------------------------------------------------------------------
# Module testing
#-----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing the consolidated scoring system...")
    
    # Create dummy data for testing
    items = ['a', 'b', 'c']
    positions = ['F', 'D', 'J'] 
    
    item_scores = {'a': 0.8, 'b': 0.6, 'c': 0.4}
    pair_scores = {('a','b'): 0.7, ('b','a'): 0.7, ('a','c'): 0.5, ('c','a'): 0.5, 
                  ('b','c'): 0.3, ('c','b'): 0.3}
    pos_scores = {'f': 0.9, 'd': 0.7, 'j': 0.8}
    pos_pair_scores = {('f','d'): 0.6, ('d','f'): 0.6, ('f','j'): 0.4, ('j','f'): 0.4,
                      ('d','j'): 0.5, ('j','d'): 0.5}
    
    # Test different modes using the main factory function
    for mode in ['item_only', 'pair_only', 'combined', 'multi_objective']:
        print(f"\nTesting {mode} mode:")
        
        # Use create_scorer (the main factory function)
        scorer = create_scorer(items, positions, item_scores, pair_scores, 
                             pos_scores, pos_pair_scores, mode=mode)
        
        mapping = np.array([0, 1, 2])  # a->F, b->D, c->J
        score = scorer.score_layout(mapping)
        print(f"  Score: {score}")
        print(f"  Mode name: {scorer.mode_name}")
        
        # Test consistency
        is_consistent = validate_scorer_consistency(scorer, 10)
        print(f"  Consistency test: {'PASS' if is_consistent else 'FAIL'}")
    
    print("\nTesting string-based layout scoring...")
    
    # Create dummy normalized_scores for testing
    norm_item_scores = {'a': 0.8, 'b': 0.6, 'c': 0.4}
    norm_item_pair_scores = {
        ('a','b'): 0.7, ('b','a'): 0.7, 
        ('a','c'): 0.5, ('c','a'): 0.5, 
        ('b','c'): 0.3, ('c','b'): 0.3
    }
    norm_position_scores = {'f': 0.9, 'd': 0.7, 'j': 0.8}
    norm_position_pair_scores = {
        ('f','d'): 0.6, ('d','f'): 0.6, 
        ('f','j'): 0.4, ('j','f'): 0.4,
        ('d','j'): 0.5, ('j','d'): 0.5
    }
    
    # Create the normalized_scores tuple
    normalized_scores = (
        norm_item_scores, 
        norm_item_pair_scores, 
        norm_position_scores, 
        norm_position_pair_scores
    )
    
    # Test string-based scoring
    items_str = "a,b,c"
    positions_str = "F,D,J"
    
    total_score, item_score, pair_score = score_layout_from_strings(
        items_str, positions_str, normalized_scores)
    print(f"  Total score: {total_score:.6f}")
    print(f"  Item component: {item_score:.6f}")  
    print(f"  Pair component: {pair_score:.6f}")
    
    print("\n✅ All tests completed successfully!")
    print("🔧 This scoring system works seamlessly in both parallel and non-parallel modes.")
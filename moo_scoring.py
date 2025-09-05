#!/usr/bin/env python3
"""
Weighted Multi-Objective Scoring for Layout Optimization

This module provides a unified scoring system that combines direct 
position-pair scoring table lookup with item-pair score weighting, 
specifically designed for multi-objective optimization of layouts.

Core Features:
- Supports arbitrary number of objectives in a position-pair scoring table
- Direct score lookup with weighting
- Optimized for partial layout scoring during search

Usage:
    scorer = WeightedMOOScorer(
        objectives=['engram7_load', 'engram7_strength'],
        position_pair_score_table='input/keypair_engram7_scores.csv',
        items=['e', 't', 'a', 'o'],
        positions=['F', 'D', 'S', 'J']
    )
    
    scores = scorer.score_layout(mapping_array)  # Returns [obj1_score, obj2_score, ...]
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ScoringArrays:
    """Minimal compatibility wrapper for existing search infrastructure."""
    item_scores: np.ndarray
    item_pair_matrix: np.ndarray  
    position_matrix: np.ndarray
    
    def __post_init__(self):
        self.n_items = len(self.item_scores)
        self.n_positions = self.position_matrix.shape[0]


class WeightedMOOScorer:
    """
    Multi-objective scorer using weighted direct position-pair lookup.
    
    This scorer implements the approach from score_layouts.py:
    1. Direct table lookup from position-pair scores with item-pair score weighting
    3. Normalization by item-pair score totals
    """
    
    def __init__(self, objectives: List[str], position_pair_score_table: str,
                 items: List[str], positions: List[str], 
                 weights: Optional[List[float]] = None, 
                 maximize: Optional[List[bool]] = None,
                 item_pair_score_table: str = "input/normalized-english-letter-pair-counts-google-ngrams.csv",
                 verbose: bool = False):
        """
        Initialize weighted MOO scorer.
        
        Args:
            objectives: List of objective names (must match position_pair table columns)
            position_pair_score_table: Path to CSV with position_pair scores for each objective
            items: List of items being optimized (e.g., ['e', 't', 'a'])
            positions: List of available positions (e.g., ['F', 'D', 'S'])
            weights: Optional weights for each objective (default: all 1.0)
            maximize: Optional direction for each objective (default: all True)
            item_pair_score_table: Path to English item-pair frequencies CSV
        """
        self.objectives = objectives
        self.items = [item.upper() for item in items]
        self.positions = [pos.upper() for pos in positions]
        self.objective_weights = weights or [1.0] * len(objectives)
        self.objective_maximize = maximize or [True] * len(objectives)
        
        # Validate inputs
        if len(self.objective_weights) != len(objectives):
            raise ValueError(f"Weights length ({len(self.objective_weights)}) != objectives length ({len(objectives)})")
        if len(self.objective_maximize) != len(objectives):
            raise ValueError(f"Maximize flags length ({len(self.objective_maximize)}) != objectives length ({len(objectives)})")
        
        if verbose:
            print(f"Initializing WeightedMOOScorer:")
            print(f"  Objectives: {objectives}")
            print(f"  Items: {self.items}")
            print(f"  Positions: {self.positions}")
        else:
            print(f"Loading {len(objectives)} objectives, {len(items)} items...")

        # Load position_pair scores for all objectives
        self.position_pair_scores = self._load_position_pair_scores(position_pair_score_table)
            
        # Load English bigram frequencies
        self.item_pair_scores = self._load_item_pair_scores(item_pair_score_table)
        self.use_itempair_weighting = len(self.item_pair_scores) > 0
          
        if self.use_itempair_weighting:
            self.item_pair_total_score = sum(self.item_pair_scores.values())
            print(f"Item-pair weighting: {len(self.item_pair_scores)} item-pairs, total score: {self.item_pair_total_score:,.0f}")
        else:
            print(f"Using unweighted scoring (no item-pair score file)")

        # Create compatibility arrays for existing search infrastructure
        n_items, n_positions = len(self.items), len(self.positions)
        self.arrays = ScoringArrays(
            item_scores=np.ones(n_items, dtype=np.float32),
            item_pair_matrix=np.ones((n_items, n_items), dtype=np.float32),
            position_matrix=np.ones((n_positions, n_positions), dtype=np.float32)
        )

    def _load_position_pair_scores(self, position_pair_score_table: str) -> Dict[str, Dict[str, float]]:
        """Load position-pair scores for all objectives from CSV table."""
        if not Path(position_pair_score_table).exists():
            raise FileNotFoundError(f"Position-pair table not found: {position_pair_score_table}")

        try:
            df = pd.read_csv(position_pair_score_table, dtype={'position_pair': str})
        except Exception as e:
            raise ValueError(f"Error reading position-pair table: {e}")

        if 'position_pair' not in df.columns:
            raise ValueError("Position-pair table must have 'position_pair' column")
        
        # Validate objectives exist in table
        missing = [obj for obj in self.objectives if obj not in df.columns]
        if missing:
            raise ValueError(f"Missing objectives in position-pair table: {missing}")
        
        # Load scores for each objective
        position_pair_scores = {}
        for obj in self.objectives:
            scores = {}
            valid_pairs = 0
            
            for _, row in df.iterrows():
                key_pair = str(row['position_pair']).strip("'\"")
                if len(key_pair) == 2 and not pd.isna(row[obj]):
                    scores[key_pair.upper()] = float(row[obj])
                    valid_pairs += 1
            
            position_pair_scores[obj] = scores
            print(f"    {obj}: {valid_pairs} position-pair scores loaded")
        
        return position_pair_scores
    
    def _load_item_pair_scores(self, item_pair_score_table: str) -> Dict[str, float]:
        """Load item-pair frequencies for weighting."""
        if not Path(item_pair_score_table).exists():
            print(f"    Warning: Item-pair score file not found: {item_pair_score_table}")
            return {}
        
        try:
            df = pd.read_csv(item_pair_score_table)
        except Exception as e:
            print(f"    Warning: Error reading item-pair score file: {e}")
            return {}
        
        # Find appropriate columns
        item_pair_col = self._find_column(df, ['item_pair', 'pair', 'bigram', 'letter_pair'])
        freq_col = self._find_column(df, ['score', 'normalized_frequency', 'frequency'])
        
        if not item_pair_col or not freq_col:
            print(f"    Warning: Required columns not found in item-pair score file")
            return {}
        
        frequencies = {}
        for _, row in df.iterrows():
            item_pair = str(row[item_pair_col]).strip().upper()
            if len(item_pair) == 2:
                frequencies[item_pair] = float(row[freq_col])
        
        return frequencies
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column name from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def score_layout(self, mapping: np.ndarray, return_components: bool = False) -> List[float]:
        """
        Score layout for all objectives using item-pair score weighting.

        Args:
            mapping: Array where mapping[i] = position_index for items[i] (-1 for unassigned)
            return_components: If True, return scores + combined average
            
        Returns:
            List of objective scores, optionally with combined average appended
        """
        scores = []
        
        for i, obj in enumerate(self.objectives):
            score = self._score_single_objective(mapping, obj)
            
            # Apply weights and direction transformations
            weighted_score = score * self.objective_weights[i]
            if not self.objective_maximize[i]:
                weighted_score = 1.0 - weighted_score
            
            scores.append(weighted_score)
        
        if return_components:
            combined_average = sum(scores) / len(scores) if scores else 0.0
            return scores + [combined_average]
        else:
            return scores
    
    def _score_single_objective(self, mapping: np.ndarray, objective: str) -> float:
        """
        Score layout for single objective using item-pair-score-weighted position-pair lookup.
        
        This implements the core scoring logic from score_layouts.py:
        1. Map item-pairs to position-pairs based on current layout
        2. Look up scores directly from position-pair scoring table
        3. Weight by item-pair scores (frequencies)
        4. Return item-pair-score-weighted average
        """
        position_pair_scores = self.position_pair_scores[objective]
        
        # Get currently placed items and their positions
        placed_items = []
        placed_positions = []
        
        for i, pos_idx in enumerate(mapping):
            if pos_idx >= 0:  # Item is assigned
                placed_items.append(self.items[i])
                placed_positions.append(self.positions[pos_idx])
        
        if len(placed_items) < 2:
            return 0.0  # Need at least 2 items for pairwise scoring
        
        # Calculate item-pair-score-weighted score
        if self.use_itempair_weighting:
            return self._calculate_item_pair_weighted_score(
                placed_items, placed_positions, position_pair_scores)
        else:
            return self._calculate_unweighted_score(
                placed_items, placed_positions, position_pair_scores)
    
    def _calculate_item_pair_weighted_score(self, items: List[str], positions: List[str], 
                                          position_pair_scores: Dict[str, float]) -> float:
        """Calculate score using item-pair score weighting."""
        weighted_total = 0.0
        item_pair_score_total = 0.0
        
        # Score all ordered pairs
        for i in range(len(items)):
            for j in range(len(items)):
                if i != j:  # Skip self-pairs
                    letter_pair = items[i] + items[j]
                    key_pair = positions[i] + positions[j]
                    
                    item_pair_score = self.item_pair_scores.get(letter_pair, 0.0)
                    if item_pair_score > 0 and key_pair in position_pair_scores:
                        score = position_pair_scores[key_pair]
                        weighted_total += score * item_pair_score
                        item_pair_score_total += item_pair_score
        
        return weighted_total / item_pair_score_total if item_pair_score_total > 0 else 0.0
    
    def _calculate_unweighted_score(self, items: List[str], positions: List[str],
                                  position_pair_scores: Dict[str, float]) -> float:
        """Calculate score without item_pair_score weighting (all pairs equal)."""
        total_score = 0.0
        pair_count = 0
        
        # Score all ordered pairs
        for i in range(len(items)):
            for j in range(len(items)):
                if i != j:  # Skip self-pairs
                    key_pair = positions[i] + positions[j]
                    
                    if key_pair in position_pair_scores:
                        total_score += position_pair_scores[key_pair]
                        pair_count += 1
        
        return total_score / pair_count if pair_count > 0 else 0.0
    
    def get_objective_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about objective score ranges for analysis."""
        stats = {}
        
        for obj, scores in self.position_pair_scores.items():
            if scores:
                values = list(scores.values())
                stats[obj] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                    'count': len(values)
                }
        
        return stats
    
    def clear_cache(self):
        """Clear any caches (no caching in this implementation)."""
        pass


def validate_item_pair_scoring_consistency(items: str, positions: str, objectives: List[str],
                                         position_pair_score_table: str, item_pair_score_table: str,
                                         verbose: bool = False) -> Dict[str, float]:
    """
    Validate that WeightedMOOScorer produces consistent results.
    
    This function can be used to compare results with score_layouts.py
    or to test scorer behavior on known layouts.
    """
    # Create mapping from strings
    items_list = list(items.upper())
    positions_list = list(positions.upper())
    
    if len(items_list) != len(positions_list):
        raise ValueError(f"Items length ({len(items_list)}) != positions length ({len(positions_list)})")
    
    # Create scorer
    scorer = WeightedMOOScorer(
        objectives=objectives,
        position_pair_score_table=position_pair_score_table,
        items=items_list,
        positions=positions_list,
        item_pair_score_table=item_pair_score_table
    )
    
    # Create mapping array (complete layout)
    mapping = np.arange(len(items_list), dtype=np.int32)
    
    # Score layout
    scores = scorer.score_layout(mapping)
    
    if verbose:
        print(f"\nValidation Results:")
        print(f"  Layout: {items} -> {positions}")
        for i, (obj, score) in enumerate(zip(objectives, scores)):
            print(f"  {obj}: {score:.9f}")
    
    return dict(zip(objectives, scores))


if __name__ == "__main__":
    # Example usage and basic testing
    print("Testing WeightedMOOScorer...")
    
    # Test configuration
    test_objectives = ['engram7_load', 'engram7_strength']
    test_items = ['e', 't', 'a', 'o']
    test_positions = ['F', 'D', 'S', 'J']
    
    try:
        scorer = WeightedMOOScorer(
            objectives=test_objectives,
            position_pair_score_table='input/keypair_engram7_scores.csv',
            items=test_items,
            positions=test_positions
        )
        
        # Test complete layout
        mapping = np.array([0, 1, 2, 3], dtype=np.int32)  # e->F, t->D, a->S, o->J
        scores = scorer.score_layout(mapping)
        
        print(f"\nTest Results:")
        print(f"  Layout: {test_items} -> {test_positions}")
        for obj, score in zip(test_objectives, scores):
            print(f"  {obj}: {score:.9f}")
        
        # Test objective statistics
        stats = scorer.get_objective_stats()
        print(f"\nObjective Statistics:")
        for obj, stat in stats.items():
            print(f"  {obj}: range [{stat['min']:.3f}, {stat['max']:.3f}], mean {stat['mean']:.3f}")
        
        print("\nWeightedMOOScorer test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Make sure 'input/keypair_engram7_scores.csv' exists with required objectives.")
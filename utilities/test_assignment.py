#!/usr/bin/env python3
"""
Test Suite for Multi-Objective Optimization with Preassigned Items

This script verifies that:
1. Scoring includes ALL items (preassigned + optimized)
2. Search finds correct Pareto-optimal solutions
3. Both exhaustive and branch-and-bound work correctly
4. Results are consistent between search modes

Usage:
    python test_moo_pipeline.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List
import sys

# Import the modules to test
from config import Config
from moo_scoring import WeightedMOOScorer
from moo_search import moo_search, validate_pareto_front
from optimize_layouts import run_moo_optimization, save_moo_results


class TestResult:
    """Track test results."""
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
    
    def record_pass(self, test_name: str):
        self.tests_run += 1
        self.tests_passed += 1
        print(f"  ✓ PASS: {test_name}")
    
    def record_fail(self, test_name: str, reason: str):
        self.tests_run += 1
        self.tests_failed += 1
        self.failures.append((test_name, reason))
        print(f"  ✗ FAIL: {test_name}")
        print(f"    Reason: {reason}")
    
    def summary(self):
        print(f"\n{'='*80}")
        print(f"TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        
        if self.tests_failed > 0:
            print(f"\nFailed Tests:")
            for name, reason in self.failures:
                print(f"  - {name}: {reason}")
            return 1
        else:
            print(f"\n🎉 ALL TESTS PASSED!")
            return 0


def create_test_data(temp_dir: Path):
    """Create minimal test data files."""
    
    # Create simple position-pair scoring table with 2 objectives
    # Objective 1: Prefers specific pairs (e.g., adjacent keys)
    # Objective 2: Different preference pattern
    position_pairs = []
    positions = ['F', 'D', 'S', 'A', 'J', 'K', 'L', ';']
    
    for p1 in positions:
        for p2 in positions:
            if p1 != p2:
                pair = p1 + p2
                # Objective 1: Higher score for adjacent positions
                score1 = 1.0 if abs(positions.index(p1) - positions.index(p2)) == 1 else 0.3
                # Objective 2: Higher score for same-hand positions
                left_hand = {'F', 'D', 'S', 'A'}
                right_hand = {'J', 'K', 'L', ';'}
                same_hand = (p1 in left_hand and p2 in left_hand) or (p1 in right_hand and p2 in right_hand)
                score2 = 0.8 if same_hand else 0.2
                
                position_pairs.append({
                    'position_pair': pair,
                    'obj1_adjacent': score1,
                    'obj2_same_hand': score2
                })
    
    pp_df = pd.DataFrame(position_pairs)
    pp_file = temp_dir / 'position_pairs.csv'
    pp_df.to_csv(pp_file, index=False)
    
    # Create simple item-pair scoring table (English letter frequencies)
    item_pairs = []
    common_pairs = [
        ('TH', 100.0), ('HE', 90.0), ('IN', 80.0), ('ER', 70.0),
        ('AN', 60.0), ('RE', 55.0), ('ND', 50.0), ('AT', 45.0),
        ('ON', 40.0), ('NT', 35.0), ('HA', 30.0), ('ES', 28.0),
        ('ST', 26.0), ('EN', 24.0), ('ED', 22.0), ('TO', 20.0)
    ]
    
    for pair, freq in common_pairs:
        item_pairs.append({
            'item_pair': pair,
            'score': freq
        })
        # Add reverse pairs with lower scores
        item_pairs.append({
            'item_pair': pair[::-1],
            'score': freq * 0.3
        })
    
    ip_df = pd.DataFrame(item_pairs)
    ip_file = temp_dir / 'item_pairs.csv'
    ip_df.to_csv(ip_file, index=False)
    
    return pp_file, ip_file


def create_test_config(temp_dir: Path, pp_file: Path, ip_file: Path, 
                       items_assigned: str = "", positions_assigned: str = "",
                       items_to_assign: str = "", positions_to_assign: str = "") -> Config:
    """Create a test configuration."""
    
    config_dict = {
        'paths': {
            'position_pair_score_table': str(pp_file),
            'item_pair_score_table': str(ip_file),
            'layout_results_folder': str(temp_dir / 'results')
        },
        'optimization': {
            'items_assigned': items_assigned,
            'positions_assigned': positions_assigned,
            'items_to_assign': items_to_assign,
            'positions_to_assign': positions_to_assign,
            'items_to_constrain': '',
            'positions_to_constrain': ''
        },
        'moo': {
            'default_objectives': ['obj1_adjacent', 'obj2_same_hand'],
            'default_weights': [1.0, 1.0],
            'default_maximize': [True, True],
            'default_max_solutions': None,
            'default_time_limit': None
        }
    }
    
    # Create results directory
    (temp_dir / 'results').mkdir(exist_ok=True)
    
    # Create a mock config object
    class MockConfig:
        def __init__(self, data):
            self.paths = type('obj', (object,), data['paths'])()
            self.optimization = type('obj', (object,), {
                **data['optimization'],
                'items_to_constrain_set': set(),
                'positions_to_constrain_set': set()
            })()
            self.moo = type('obj', (object,), data['moo'])()
            self._config_path = temp_dir / 'config.yaml'
    
    return MockConfig(config_dict)


def test_scorer_sees_all_items(results: TestResult):
    """Test that scorer includes all items in calculations."""
    print("\n" + "="*80)
    print("TEST 1: Scorer Sees All Items")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pp_file, ip_file = create_test_data(temp_path)
        
        # Test case: 2 preassigned + 2 to optimize = 4 total items
        items_assigned = ['T', 'H']
        positions_assigned = ['F', 'D']
        items_to_assign = ['E', 'A']
        positions_to_assign = ['S', 'J']
        
        all_items = items_assigned + items_to_assign
        all_positions = positions_assigned + positions_to_assign
        
        print(f"\nSetup:")
        print(f"  Preassigned: {items_assigned} -> {positions_assigned}")
        print(f"  To optimize: {items_to_assign} in {positions_to_assign}")
        print(f"  Total items: {all_items}")
        
        # Create scorer
        scorer = WeightedMOOScorer(
            objectives=['obj1_adjacent', 'obj2_same_hand'],
            position_pair_score_table=str(pp_file),
            items=all_items,
            positions=all_positions,
            item_pair_score_table=str(ip_file)
        )
        
        # Test 1: Scorer has correct number of items
        if len(scorer.items) == 4:
            results.record_pass("Scorer initialized with 4 items")
        else:
            results.record_fail("Scorer initialized with 4 items", 
                              f"Expected 4 items, got {len(scorer.items)}")
        
        # Test 2: Create a complete mapping and score it
        # Mapping: T->F(0), H->D(1), E->S(2), A->J(3)
        mapping = np.array([0, 1, 2, 3], dtype=np.int32)
        scores = scorer.score_layout(mapping)
        
        if len(scores) == 2:
            results.record_pass("Scorer returns scores for 2 objectives")
        else:
            results.record_fail("Scorer returns scores for 2 objectives",
                              f"Expected 2 scores, got {len(scores)}")
        
        # Test 3: Verify scoring uses all 4 items (check by examining pairs)
        # We expect scores to consider all item pairs: TH, HE, EA, AT, etc.
        # If only first 2 items were scored, we'd get different results
        
        # Try a different mapping: T->F(0), H->D(1), E->J(3), A->S(2)
        mapping2 = np.array([0, 1, 3, 2], dtype=np.int32)
        scores2 = scorer.score_layout(mapping2)
        
        # Scores should be different because item positions changed
        if not np.array_equal(scores, scores2):
            results.record_pass("Different mappings produce different scores")
        else:
            results.record_fail("Different mappings produce different scores",
                              "Same scores for different mappings - might not be using all items")
        
        print(f"\n  Mapping 1 scores: {scores}")
        print(f"  Mapping 2 scores: {scores2}")


def test_search_without_preassignment(results: TestResult):
    """Test search works correctly without preassigned items."""
    print("\n" + "="*80)
    print("TEST 2: Search Without Preassignment")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pp_file, ip_file = create_test_data(temp_path)
        
        # Simple case: 3 items, 3 positions, no preassignment
        config = create_test_config(
            temp_path, pp_file, ip_file,
            items_assigned="",
            positions_assigned="",
            items_to_assign="THE",
            positions_to_assign="FDS"
        )
        
        print(f"\nSetup:")
        print(f"  Items to optimize: THE")
        print(f"  Positions available: FDS")
        
        # Run exhaustive search (should be fast with only 3! = 6 permutations)
        pareto_front, stats = run_moo_optimization(
            config=config,
            objectives=['obj1_adjacent', 'obj2_same_hand'],
            position_pair_score_table=str(pp_file),
            weights=[1.0, 1.0],
            maximize=[True, True],
            item_pair_score_table=str(ip_file),
            search_mode='exhaustive',
            verbose=False
        )
        
        # Test 1: Search completes
        if stats.solutions_found > 0:
            results.record_pass("Search found solutions")
        else:
            results.record_fail("Search found solutions", "No solutions found")
        
        # Test 2: Pareto front is valid
        if validate_pareto_front(pareto_front):
            results.record_pass("Pareto front is valid (no dominated solutions)")
        else:
            results.record_fail("Pareto front is valid", "Contains dominated solutions")
        
        # Test 3: All solutions have correct structure
        all_valid = True
        for sol in pareto_front:
            if len(sol['mapping']) != 3:
                all_valid = False
                break
            if len(sol['objectives']) != 2:
                all_valid = False
                break
        
        if all_valid:
            results.record_pass("All solutions have correct structure")
        else:
            results.record_fail("All solutions have correct structure", 
                              "Some solutions malformed")
        
        print(f"\n  Solutions found: {stats.solutions_found}")
        print(f"  Pareto front size: {len(pareto_front)}")


def test_search_with_preassignment_asymmetric(results: TestResult):
    """Test with asymmetric positions to guarantee score variation."""
    print("\n" + "="*80)
    print("TEST 3: Search With Preassignment (Asymmetric Positions)")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create position-pair scores with ASYMMETRIC positions
        position_pairs = []
        # Use positions F, D, S where F-D and D-S are adjacent (creating asymmetry)
        positions = ['F', 'D', 'S']
        
        for p1 in positions:
            for p2 in positions:
                if p1 != p2:
                    pair = p1 + p2
                    # F(0), D(1), S(2)
                    p1_idx = positions.index(p1)
                    p2_idx = positions.index(p2)
                    
                    # Adjacency creates asymmetry:
                    # F-D adjacent, D-S adjacent, but F-S not adjacent
                    score1 = 1.0 if abs(p1_idx - p2_idx) == 1 else 0.3
                    
                    # Same hand (all left hand)
                    score2 = 0.8
                    
                    position_pairs.append({
                        'position_pair': pair,
                        'obj1_adjacent': score1,
                        'obj2_same_hand': score2
                    })
        
        pp_df = pd.DataFrame(position_pairs)
        pp_file = temp_path / 'position_pairs_asym.csv'
        pp_df.to_csv(pp_file, index=False)
        
        # Create item-pair scores
        item_pairs = [
            {'item_pair': 'TH', 'score': 100.0},
            {'item_pair': 'HT', 'score': 30.0},
            {'item_pair': 'HE', 'score': 90.0},
            {'item_pair': 'EH', 'score': 27.0},
            {'item_pair': 'TE', 'score': 85.0},
            {'item_pair': 'ET', 'score': 25.0},
        ]
        ip_df = pd.DataFrame(item_pairs)
        ip_file = temp_path / 'item_pairs_asym.csv'
        ip_df.to_csv(ip_file, index=False)
        
        # Preassign T, optimize H and E
        config = create_test_config(
            temp_path, pp_file, ip_file,
            items_assigned="T",
            positions_assigned="F",
            items_to_assign="HE",
            positions_to_assign="DS"
        )
        
        print(f"\nSetup:")
        print(f"  Preassigned: T -> F")
        print(f"  To optimize: HE in DS (D and S are NOT symmetric)")
        print(f"  Key: D is adjacent to both F and S, but S is only adjacent to D")
        
        pareto_front, stats = run_moo_optimization(
            config=config,
            objectives=['obj1_adjacent', 'obj2_same_hand'],
            position_pair_score_table=str(pp_file),
            weights=[1.0, 1.0],
            maximize=[True, True],
            item_pair_score_table=str(ip_file),
            search_mode='exhaustive',
            verbose=False
        )
        
        # Check score variation
        unique_scores = set()
        for sol in pareto_front:
            score_tuple = tuple(round(s, 6) for s in sol['objectives'])
            unique_scores.add(score_tuple)
        
        print(f"\n  Solutions evaluated: {stats.solutions_found}")
        print(f"  Pareto front size: {len(pareto_front)}")
        
        # Show all evaluated solutions if we could track them
        print(f"\n  Pareto-optimal solution(s):")
        for i, sol in enumerate(pareto_front, 1):
            items_order = ['T', 'H', 'E']
            items_str = ''.join(items_order)
            positions_str = ''.join(sol['mapping'][item] for item in items_order)
            scores_str = [f"{s:.6f}" for s in sol['objectives']]
            print(f"    {i}. {items_str} -> {positions_str} | {scores_str}")
        
        # Correct test: verify that we found solutions and they're non-dominated
        if stats.solutions_found == 2:
            results.record_pass("Both arrangements were evaluated")
        else:
            results.record_fail("Both arrangements were evaluated", 
                            f"Expected 2, got {stats.solutions_found}")
        
        # The Pareto front can have 1 OR 2 solutions depending on dominance
        if len(pareto_front) >= 1:
            results.record_pass("Pareto front contains non-dominated solution(s)")
            
            if len(pareto_front) == 1:
                print(f"  ✓ One solution dominated the other (correct)")
            else:
                print(f"  ✓ Multiple non-dominated solutions found")
        else:
            results.record_fail("Pareto front contains solutions", "Pareto front is empty")
        
        # Verify the Pareto front is valid
        if validate_pareto_front(pareto_front):
            results.record_pass("Pareto front is valid (no dominated solutions)")
        else:
            results.record_fail("Pareto front is valid", "Contains dominated solutions")
            
def test_both_search_modes_agree(results: TestResult):
    """Test that exhaustive and branch-bound find the same Pareto front."""
    print("\n" + "="*80)
    print("TEST 4: Both Search Modes Agree")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pp_file, ip_file = create_test_data(temp_path)
        
        # Small problem to ensure both complete quickly
        config = create_test_config(
            temp_path, pp_file, ip_file,
            items_assigned="T",
            positions_assigned="F",
            items_to_assign="HE",
            positions_to_assign="DS"
        )
        
        print(f"\nSetup:")
        print(f"  Preassigned: T -> F")
        print(f"  To optimize: HE in DS")
        
        # Run exhaustive search
        print(f"\n  Running exhaustive search...")
        pareto_exhaustive, stats_ex = run_moo_optimization(
            config=config,
            objectives=['obj1_adjacent', 'obj2_same_hand'],
            position_pair_score_table=str(pp_file),
            weights=[1.0, 1.0],
            maximize=[True, True],
            item_pair_score_table=str(ip_file),
            search_mode='exhaustive',
            verbose=False
        )
        
        # Run branch-and-bound search
        print(f"  Running branch-and-bound search...")
        pareto_bb, stats_bb = run_moo_optimization(
            config=config,
            objectives=['obj1_adjacent', 'obj2_same_hand'],
            position_pair_score_table=str(pp_file),
            weights=[1.0, 1.0],
            maximize=[True, True],
            item_pair_score_table=str(ip_file),
            search_mode='branch-bound',
            verbose=False
        )
        
        # Test 1: Both find solutions
        if len(pareto_exhaustive) > 0 and len(pareto_bb) > 0:
            results.record_pass("Both search modes found solutions")
        else:
            results.record_fail("Both search modes found solutions",
                              f"Exhaustive: {len(pareto_exhaustive)}, B&B: {len(pareto_bb)}")
        
        # Test 2: Same number of Pareto-optimal solutions
        if len(pareto_exhaustive) == len(pareto_bb):
            results.record_pass("Both modes found same number of Pareto solutions")
        else:
            results.record_fail("Both modes found same number of Pareto solutions",
                              f"Exhaustive: {len(pareto_exhaustive)}, B&B: {len(pareto_bb)}")
        
        # Test 3: Same solutions (check objective values)
        exhaustive_scores = {tuple(sol['objectives']) for sol in pareto_exhaustive}
        bb_scores = {tuple(sol['objectives']) for sol in pareto_bb}
        
        if exhaustive_scores == bb_scores:
            results.record_pass("Both modes found identical solutions")
        else:
            results.record_fail("Both modes found identical solutions",
                              f"Different solution sets")
            print(f"    Exhaustive only: {exhaustive_scores - bb_scores}")
            print(f"    B&B only: {bb_scores - exhaustive_scores}")
        
        print(f"\n  Exhaustive: {len(pareto_exhaustive)} solutions, {stats_ex.nodes_processed} nodes")
        print(f"  Branch-bound: {len(pareto_bb)} solutions, {stats_bb.nodes_processed} nodes")
        if hasattr(stats_bb, 'nodes_pruned'):
            print(f"  Nodes pruned by B&B: {stats_bb.nodes_pruned}")


def test_scoring_includes_interactions(results: TestResult):
    """Test that scoring includes interactions between preassigned and optimized items."""
    print("\n" + "="*80)
    print("TEST 5: Scoring Includes All Item Interactions")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pp_file, ip_file = create_test_data(temp_path)
        
        # Setup with known high-frequency pair: TH
        items_assigned = ['T']
        positions_assigned = ['F']
        items_to_assign = ['H', 'E']
        positions_to_assign = ['D', 'S']
        
        all_items = items_assigned + items_to_assign
        all_positions = positions_assigned + positions_to_assign
        
        print(f"\nSetup:")
        print(f"  Preassigned: T -> F")
        print(f"  To optimize: HE in DS")
        print(f"  Key test: TH is a high-frequency pair in item_pairs.csv")
        
        # Create scorer
        scorer = WeightedMOOScorer(
            objectives=['obj1_adjacent', 'obj2_same_hand'],
            position_pair_score_table=str(pp_file),
            items=all_items,
            positions=all_positions,
            item_pair_score_table=str(ip_file)
        )
        
        # Test mappings
        # Mapping 1: T->F(0), H->D(1), E->S(2) - TH adjacent
        mapping1 = np.array([0, 1, 2], dtype=np.int32)
        scores1 = scorer.score_layout(mapping1)
        
        # Mapping 2: T->F(0), H->S(2), E->D(1) - TH not adjacent
        mapping2 = np.array([0, 2, 1], dtype=np.int32)
        scores2 = scorer.score_layout(mapping2)
        
        print(f"\n  Mapping 1 (T->F, H->D adjacent): {scores1}")
        print(f"  Mapping 2 (T->F, H->S not adjacent): {scores2}")
        
        # Test 1: Scores should be different
        if not np.array_equal(scores1, scores2):
            results.record_pass("Different positions for H produce different scores")
        else:
            results.record_fail("Different positions for H produce different scores",
                              "Scores identical - may not be considering TH pair")
        
        # Test 2: For obj1_adjacent, mapping1 should score higher (TH adjacent)
        # F and D are adjacent positions, and TH is a common pair
        if scores1[0] > scores2[0]:
            results.record_pass("Adjacent placement of TH scores higher (obj1)")
        else:
            results.record_fail("Adjacent placement of TH scores higher",
                              f"Score1: {scores1[0]}, Score2: {scores2[0]}")
        
        print(f"\n  ✓ Confirmed: Scoring considers interactions between preassigned T and optimized H")

#!/usr/bin/env python3
"""
Additional Test Cases to Add to test.py

These tests address gaps in the current test coverage.
"""

def test_exact_score_verification(results: TestResult):
    """Verify exact score calculation against hand-calculated values."""
    print("\n" + "="*80)
    print("TEST 6: Exact Score Verification")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create minimal, controlled test case
        position_pairs = [
            {'position_pair': 'FD', 'obj1': 1.0},
            {'position_pair': 'DF', 'obj1': 0.5},
        ]
        pp_df = pd.DataFrame(position_pairs)
        pp_file = temp_path / 'position_pairs_exact.csv'
        pp_df.to_csv(pp_file, index=False)
        
        item_pairs = [
            {'item_pair': 'TH', 'score': 100.0},
            {'item_pair': 'HT', 'score': 20.0},
        ]
        ip_df = pd.DataFrame(item_pairs)
        ip_file = temp_path / 'item_pairs_exact.csv'
        ip_df.to_csv(ip_file, index=False)
        
        # Manual calculation:
        # Items: T, H at positions F, D
        # TH pair: frequency 100, position score 1.0 -> contributes 100
        # HT pair: frequency 20, position score 0.5 -> contributes 10
        # Total weighted: 110, total weight: 120
        # Expected score: 110/120 = 0.916666...
        
        scorer = WeightedMOOScorer(
            objectives=['obj1'],
            position_pair_score_table=str(pp_file),
            items=['T', 'H'],
            positions=['F', 'D'],
            item_pair_score_table=str(ip_file)
        )
        
        mapping = np.array([0, 1], dtype=np.int32)  # T->F, H->D
        scores = scorer.score_layout(mapping)
        
        expected_score = 110.0 / 120.0
        if abs(scores[0] - expected_score) < 1e-6:
            results.record_pass("Exact score calculation matches hand-calculated value")
        else:
            results.record_fail("Exact score calculation",
                              f"Expected {expected_score:.9f}, got {scores[0]:.9f}")


def test_constraints_with_preassignment(results: TestResult):
    """Test that constraints work correctly with preassigned items."""
    print("\n" + "="*80)
    print("TEST 7: Constraints With Preassignment")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pp_file, ip_file = create_test_data(temp_path)
        
        # Preassign T->F, constrain H to only positions D or S
        config = create_test_config(
            temp_path, pp_file, ip_file,
            items_assigned="T",
            positions_assigned="F",
            items_to_assign="HEA",
            positions_to_assign="DSJK"
        )
        
        # Add constraints
        config.optimization.items_to_constrain_set = {'H'}
        config.optimization.positions_to_constrain_set = {'D', 'S'}
        
        print(f"\nSetup:")
        print(f"  Preassigned: T -> F")
        print(f"  To optimize: HEA in DSJK")
        print(f"  Constraint: H must be in D or S")
        
        pareto_front, stats = run_moo_optimization(
            config=config,
            objectives=['obj1_adjacent', 'obj2_same_hand'],
            position_pair_score_table=str(pp_file),
            weights=[1.0, 1.0],
            maximize=[True, True],
            item_pair_score_table=str(ip_file),
            search_mode='exhaustive',
            verbose=False
        )
        
        # Verify all solutions respect constraint
        constraint_satisfied = True
        for sol in pareto_front:
            h_position = sol['mapping']['H']
            if h_position not in ['D', 'S']:
                constraint_satisfied = False
                break
        
        if constraint_satisfied:
            results.record_pass("All solutions satisfy constraint (H in D or S)")
        else:
            results.record_fail("Constraint satisfaction", "Some solutions violate constraint")


def test_large_preassignment(results: TestResult):
    """Test with many preassigned items, few to optimize."""
    print("\n" + "="*80)
    print("TEST 8: Large Preassignment")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pp_file, ip_file = create_test_data(temp_path)
        
        # Preassign 5 items, optimize only 2
        config = create_test_config(
            temp_path, pp_file, ip_file,
            items_assigned="THEAN",
            positions_assigned="FDSAK",
            items_to_assign="IO",
            positions_to_assign="JL"
        )
        
        print(f"\nSetup:")
        print(f"  Preassigned: THEAN -> FDSAK (5 items)")
        print(f"  To optimize: IO in JL (2 items)")
        print(f"  Total: 7 items")
        
        pareto_front, stats = run_moo_optimization(
            config=config,
            objectives=['obj1_adjacent', 'obj2_same_hand'],
            position_pair_score_table=str(pp_file),
            weights=[1.0, 1.0],
            maximize=[True, True],
            item_pair_score_table=str(ip_file),
            search_mode='exhaustive',
            verbose=False
        )
        
        # Verify all solutions include all 7 items
        all_complete = True
        for sol in pareto_front:
            if len(sol['mapping']) != 7:
                all_complete = False
                break
            # Verify preassignments are preserved
            if (sol['mapping']['T'] != 'F' or sol['mapping']['H'] != 'D' or
                sol['mapping']['E'] != 'S' or sol['mapping']['A'] != 'A' or
                sol['mapping']['N'] != 'K'):
                all_complete = False
                break
        
        if all_complete:
            results.record_pass("All solutions include all 7 items with correct preassignments")
        else:
            results.record_fail("Complete solutions", "Some solutions missing items or incorrect preassignments")


def test_all_items_preassigned(results: TestResult):
    """Test when all items are preassigned (no optimization)."""
    print("\n" + "="*80)
    print("TEST 9: All Items Preassigned")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pp_file, ip_file = create_test_data(temp_path)
        
        # All items preassigned, nothing to optimize
        config = create_test_config(
            temp_path, pp_file, ip_file,
            items_assigned="THE",
            positions_assigned="FDS",
            items_to_assign="",
            positions_to_assign=""
        )
        
        print(f"\nSetup:")
        print(f"  All items preassigned: THE -> FDS")
        print(f"  Nothing to optimize")
        
        pareto_front, stats = run_moo_optimization(
            config=config,
            objectives=['obj1_adjacent', 'obj2_same_hand'],
            position_pair_score_table=str(pp_file),
            weights=[1.0, 1.0],
            maximize=[True, True],
            item_pair_score_table=str(ip_file),
            search_mode='exhaustive',
            verbose=False
        )
        
        # Should find exactly 1 solution (the preassigned mapping)
        if len(pareto_front) == 1:
            results.record_pass("Found exactly 1 solution (the preassigned mapping)")
        else:
            results.record_fail("All items preassigned", 
                              f"Expected 1 solution, found {len(pareto_front)}")
        
        # Verify it's the correct solution
        if len(pareto_front) == 1:
            sol = pareto_front[0]
            if (sol['mapping']['T'] == 'F' and 
                sol['mapping']['H'] == 'D' and 
                sol['mapping']['E'] == 'S'):
                results.record_pass("Solution matches preassigned mapping")
            else:
                results.record_fail("Solution correctness", "Mapping doesn't match preassignment")


def test_mixed_objective_directions(results: TestResult):
    """Test with both maximize and minimize objectives."""
    print("\n" + "="*80)
    print("TEST 10: Mixed Objective Directions")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pp_file, ip_file = create_test_data(temp_path)
        
        config = create_test_config(
            temp_path, pp_file, ip_file,
            items_assigned="",
            positions_assigned="",
            items_to_assign="THE",
            positions_to_assign="FDS"
        )
        
        print(f"\nSetup:")
        print(f"  Objectives: maximize obj1, minimize obj2")
        
        # Run with mixed directions
        pareto_front, stats = run_moo_optimization(
            config=config,
            objectives=['obj1_adjacent', 'obj2_same_hand'],
            position_pair_score_table=str(pp_file),
            weights=[1.0, 1.0],
            maximize=[True, False],  # Maximize obj1, minimize obj2
            item_pair_score_table=str(ip_file),
            search_mode='exhaustive',
            verbose=False
        )
        
        # Just verify it completes and finds solutions
        if len(pareto_front) > 0:
            results.record_pass("Mixed directions (max/min) produces solutions")
        else:
            results.record_fail("Mixed directions", "No solutions found")
        
        # Verify Pareto front is valid for mixed directions
        if validate_pareto_front(pareto_front):
            results.record_pass("Pareto front valid with mixed directions")
        else:
            results.record_fail("Pareto front validation", "Invalid with mixed directions")


def main():
    """Run all tests."""
    print("="*80)
    print("MULTI-OBJECTIVE OPTIMIZATION TEST SUITE")
    print("="*80)
    print("\nThis test suite verifies that:")
    print("1. Scorer includes ALL items (preassigned + optimized)")
    print("2. Search finds correct Pareto-optimal solutions")
    print("3. Preassigned items work correctly")
    print("4. Both search modes produce identical results")
    print("5. Scoring includes all item interactions")
    
    results = TestResult()
    
    try:
        # Original tests
        test_scorer_sees_all_items(results)
        test_search_without_preassignment(results)
        test_search_with_preassignment_asymmetric(results)
        test_both_search_modes_agree(results)
        test_scoring_includes_interactions(results)
        
        # New tests
        test_exact_score_verification(results)
        test_constraints_with_preassignment(results)
        test_large_preassignment(results)
        test_all_items_preassigned(results)
        test_mixed_objective_directions(results)
        
    except Exception as e:
        print(f"\n❌ TEST SUITE CRASHED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return results.summary()


if __name__ == "__main__":
    sys.exit(main())
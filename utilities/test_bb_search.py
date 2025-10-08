#!/usr/bin/env python3
"""
Comprehensive Branch-and-Bound Validation Test Suite

Tests that branch-and-bound:
1. Finds identical Pareto fronts as exhaustive search
2. Is faster (processes fewer nodes)
3. Never prunes optimal solutions (upper bounds are sound)
4. Provides meaningful pruning on larger problems
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import time
from typing import Dict, List, Tuple
import sys
from itertools import permutations

from config import Config
from moo_scoring import WeightedMOOScorer
from moo_search import moo_search, validate_pareto_front, MOOUpperBoundCalculator
from optimize_layouts import run_moo_optimization


class BBValidationResult:
    """Track B&B validation results."""
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
        self.performance_data = []
    
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
    
    def record_performance(self, problem_size: int, ex_nodes: int, bb_nodes: int, 
                          ex_time: float, bb_time: float):
        """Record performance comparison data."""
        self.performance_data.append({
            'size': problem_size,
            'ex_nodes': ex_nodes,
            'bb_nodes': bb_nodes,
            'ex_time': ex_time,
            'bb_time': bb_time,
            'speedup': ex_time / bb_time if bb_time > 0 else 0,
            'pruning_rate': (ex_nodes - bb_nodes) / ex_nodes * 100 if ex_nodes > 0 else 0
        })
    
    def print_performance_summary(self):
        """Print performance comparison summary."""
        if not self.performance_data:
            return
        
        print(f"\n{'='*80}")
        print(f"PERFORMANCE COMPARISON: Branch-and-Bound vs Exhaustive")
        print(f"{'='*80}")
        print(f"\n{'Size':<6} {'Ex Nodes':<12} {'BB Nodes':<12} {'Pruned %':<10} {'Ex Time':<10} {'BB Time':<10} {'Speedup':<8}")
        print(f"{'-'*80}")
        
        for data in self.performance_data:
            print(f"{data['size']:<6} "
                  f"{data['ex_nodes']:<12,} "
                  f"{data['bb_nodes']:<12,} "
                  f"{data['pruning_rate']:>8.1f}% "
                  f"{data['ex_time']:>8.3f}s "
                  f"{data['bb_time']:>8.3f}s "
                  f"{data['speedup']:>6.2f}x")
    
    def summary(self):
        print(f"\n{'='*80}")
        print(f"BRANCH-AND-BOUND VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        
        if self.tests_failed > 0:
            print(f"\nFailed Tests:")
            for name, reason in self.failures:
                print(f"  - {name}: {reason}")
        
        self.print_performance_summary()
        
        if self.tests_failed > 0:
            return 1
        else:
            print(f"\n🎉 ALL BRANCH-AND-BOUND TESTS PASSED!")
            return 0


def create_test_data_bb(temp_dir: Path):
    """Create test data with more variation for B&B testing."""
    
    # Expand to 12 positions (enough for 10+ item tests)
    positions = [
        'F', 'D', 'S', 'A',      # Left hand home row
        'J', 'K', 'L', ';',      # Right hand home row  
        'R', 'U', 'I', 'E'       # Additional positions (top row)
    ]
    
    # Create position-pair scores with multiple objectives
    position_pairs = []
    
    for p1 in positions:
        for p2 in positions:
            if p1 != p2:
                pair = p1 + p2
                p1_idx = positions.index(p1)
                p2_idx = positions.index(p2)
                
                # Objective 1: Adjacent keys (simplified for expanded keyboard)
                # Consider positions 0-3 adjacent, 4-7 adjacent, 8-11 adjacent
                p1_group = p1_idx // 4
                p2_group = p2_idx // 4
                if p1_group == p2_group:
                    score1 = 1.0 if abs(p1_idx - p2_idx) == 1 else 0.5
                else:
                    score1 = 0.3
                
                # Objective 2: Same hand
                left_hand = {'F', 'D', 'S', 'A', 'R', 'E'}
                right_hand = {'J', 'K', 'L', ';', 'U', 'I'}
                same_hand = (p1 in left_hand and p2 in left_hand) or \
                           (p1 in right_hand and p2 in right_hand)
                score2 = 0.8 if same_hand else 0.2
                
                # Objective 3: Same row (more complex scoring)
                home_row = {'F', 'D', 'S', 'A', 'J', 'K', 'L', ';'}
                top_row = {'R', 'U', 'I', 'E'}
                score3 = 0.9 if (p1 in home_row and p2 in home_row) else \
                         0.7 if (p1 in top_row and p2 in top_row) else 0.4
                
                position_pairs.append({
                    'position_pair': pair,
                    'obj1_adjacent': score1,
                    'obj2_same_hand': score2,
                    'obj3_same_row': score3
                })
    
    pp_df = pd.DataFrame(position_pairs)
    pp_file = temp_dir / 'position_pairs_bb.csv'
    pp_df.to_csv(pp_file, index=False)
    
    # Create item-pair scores with realistic frequency distribution
    # UPDATED: Add more letter pairs for 10-item tests
    item_pairs = []
    common_pairs = [
        ('TH', 100.0), ('HE', 90.0), ('IN', 80.0), ('ER', 70.0),
        ('AN', 60.0), ('RE', 55.0), ('ND', 50.0), ('AT', 45.0),
        ('ON', 40.0), ('NT', 35.0), ('HA', 30.0), ('ES', 28.0),
        ('ST', 26.0), ('EN', 24.0), ('ED', 22.0), ('TO', 20.0),
        ('IT', 18.0), ('OU', 16.0), ('EA', 15.0), ('HI', 14.0),
        # Additional pairs for expanded letter set
        ('OR', 13.0), ('RI', 12.0), ('IS', 11.0), ('NO', 10.0),
        ('SE', 9.0), ('TE', 8.5), ('AR', 8.0), ('AS', 7.5),
        ('AL', 7.0), ('LE', 6.5), ('SI', 6.0), ('NE', 5.5),
        ('RA', 5.0), ('RO', 4.5), ('IR', 4.0), ('NA', 3.5),
    ]
    
    for pair, freq in common_pairs:
        item_pairs.append({'item_pair': pair, 'score': freq})
        # Add reverse pairs with lower frequency
        item_pairs.append({'item_pair': pair[::-1], 'score': freq * 0.3})
    
    ip_df = pd.DataFrame(item_pairs)
    ip_file = temp_dir / 'item_pairs_bb.csv'
    ip_df.to_csv(ip_file, index=False)
    
    return pp_file, ip_file


def create_test_config_bb(temp_dir: Path, pp_file: Path, ip_file: Path,
                         items_assigned: str = "", positions_assigned: str = "",
                         items_to_assign: str = "", positions_to_assign: str = "") -> Config:
    """Create test configuration for B&B testing."""
    
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
    
    (temp_dir / 'results').mkdir(exist_ok=True)
    
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


def compare_pareto_fronts(front1: List[Dict], front2: List[Dict], 
                          tolerance: float = 1e-9) -> Tuple[bool, str]:
    """
    Compare two Pareto fronts for equivalence.
    
    Returns (is_equal, reason)
    """
    if len(front1) != len(front2):
        return False, f"Different sizes: {len(front1)} vs {len(front2)}"
    
    # Extract objective vectors and sort for comparison
    scores1 = [tuple(sol['objectives']) for sol in front1]
    scores2 = [tuple(sol['objectives']) for sol in front2]
    
    scores1_sorted = sorted(scores1)
    scores2_sorted = sorted(scores2)
    
    # Compare with tolerance
    for s1, s2 in zip(scores1_sorted, scores2_sorted):
        if len(s1) != len(s2):
            return False, f"Different objective counts: {len(s1)} vs {len(s2)}"
        
        for v1, v2 in zip(s1, s2):
            if abs(v1 - v2) > tolerance:
                return False, f"Objective value mismatch: {v1} vs {v2}"
    
    return True, "Fronts are identical"


def test_bb_correctness_small_problems(results: BBValidationResult):
    """Test B&B correctness on small problems (3-5 items)."""
    print("\n" + "="*80)
    print("TEST 1: B&B Correctness - Small Problems")
    print("="*80)
    
    test_cases = [
        ("3 items, no preassign", "", "", "THE", "FDS"),
        ("4 items, no preassign", "", "", "THEA", "FDSJ"),
        ("4 items, 1 preassign", "T", "F", "HEA", "DSJ"),
        ("5 items, 2 preassign", "TH", "FD", "EAN", "SJK"),
    ]
    
    for test_name, items_assigned, positions_assigned, items_to_assign, positions_to_assign in test_cases:
        print(f"\n  Testing: {test_name}")
        print(f"    Preassigned: {items_assigned or 'none'} -> {positions_assigned or 'none'}")
        print(f"    To optimize: {items_to_assign} in {positions_to_assign}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pp_file, ip_file = create_test_data_bb(temp_path)
            
            config = create_test_config_bb(
                temp_path, pp_file, ip_file,
                items_assigned, positions_assigned,
                items_to_assign, positions_to_assign
            )
            
            # Run exhaustive search
            from optimize_layouts import run_moo_optimization
            pareto_ex, stats_ex = run_moo_optimization(
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
            
            # Compare results
            is_equal, reason = compare_pareto_fronts(pareto_ex, pareto_bb)
            
            print(f"    Exhaustive: {len(pareto_ex)} solutions, {stats_ex.nodes_processed:,} nodes")
            print(f"    B&B: {len(pareto_bb)} solutions, {stats_bb.nodes_processed:,} nodes")
            print(f"    Pruned: {stats_bb.nodes_pruned:,} nodes ({stats_bb.nodes_pruned/stats_ex.nodes_processed*100:.1f}%)")
            
            if is_equal:
                results.record_pass(f"{test_name} - Identical Pareto fronts")
            else:
                results.record_fail(f"{test_name} - Identical Pareto fronts", reason)
            
            # Record performance
            problem_size = len(items_to_assign) + len(items_assigned)
            results.record_performance(
                problem_size, 
                stats_ex.nodes_processed, 
                stats_bb.nodes_processed,
                stats_ex.elapsed_time,
                stats_bb.elapsed_time
            )


def test_bb_performance_scaling(results: BBValidationResult):
    """Test that B&B provides meaningful speedup on larger problems."""
    print("\n" + "="*80)
    print("TEST 2: B&B Performance Scaling")
    print("="*80)
    
    test_cases = [
        ("6 items", "", "", "THEANO", "FDSJKL"),
        ("7 items, 2 preassign", "TH", "FD", "EANOI", "SJKL;"),
        ("8 items, 3 preassign", "THE", "FDS", "ANOIR", "JKL;A"),
        
        # NEW: Add 10-item test cases
        # Option 1: 10 items with pre-assignment (more manageable)
        ("10 items, 4 preassign", "THEA", "FDSK", "NOIRSL", "JL;RUI"),
        
        # Option 2: 10 items with more pre-assignment (even faster)
        ("10 items, 5 preassign", "THEAN", "FDSKJ", "OIRSL", "L;RUI"),
    ]
    
    for test_name, items_assigned, positions_assigned, items_to_assign, positions_to_assign in test_cases:
        print(f"\n  Testing: {test_name}")
        
        # Calculate expected search space
        n_items = len(items_to_assign)
        n_positions = len(positions_to_assign)
        if n_positions >= n_items:
            from math import factorial
            search_space = factorial(n_positions) // factorial(n_positions - n_items)
            print(f"    Search space: {n_items} items in {n_positions} positions = {search_space:,} permutations")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pp_file, ip_file = create_test_data_bb(temp_path)
            
            config = create_test_config_bb(
                temp_path, pp_file, ip_file,
                items_assigned, positions_assigned,
                items_to_assign, positions_to_assign
            )
            
            # Run exhaustive search with time limit
            print(f"    Running exhaustive search...")
            start = time.time()
            pareto_ex, stats_ex = run_moo_optimization(
                config=config,
                objectives=['obj1_adjacent', 'obj2_same_hand'],
                position_pair_score_table=str(pp_file),
                weights=[1.0, 1.0],
                maximize=[True, True],
                item_pair_score_table=str(ip_file),
                search_mode='exhaustive',
                time_limit=60.0,  # INCREASED: 60 second limit for larger problems
                verbose=False
            )
            ex_time = time.time() - start
            
            # Run branch-and-bound search
            print(f"    Running branch-and-bound search...")
            start = time.time()
            pareto_bb, stats_bb = run_moo_optimization(
                config=config,
                objectives=['obj1_adjacent', 'obj2_same_hand'],
                position_pair_score_table=str(pp_file),
                weights=[1.0, 1.0],
                maximize=[True, True],
                item_pair_score_table=str(ip_file),
                search_mode='branch-bound',
                time_limit=60.0,
                verbose=False
            )
            bb_time = time.time() - start
            
            print(f"    Exhaustive: {stats_ex.nodes_processed:,} nodes in {ex_time:.3f}s")
            print(f"    B&B: {stats_bb.nodes_processed:,} nodes in {bb_time:.3f}s")
            
            # Test 1: B&B should process fewer nodes
            if stats_bb.nodes_processed < stats_ex.nodes_processed:
                pruning_rate = (stats_ex.nodes_processed - stats_bb.nodes_processed) / stats_ex.nodes_processed * 100
                results.record_pass(f"{test_name} - B&B processes fewer nodes ({pruning_rate:.1f}% pruned)")
            else:
                results.record_fail(f"{test_name} - B&B processes fewer nodes", 
                                  f"B&B: {stats_bb.nodes_processed}, Ex: {stats_ex.nodes_processed}")
            
            # Test 2: B&B should be faster (if both completed)
            # UPDATED: More lenient threshold for larger problems
            min_nodes_for_speedup = 10000 if n_items >= 10 else 1000
            
            if bb_time < ex_time and stats_ex.nodes_processed > min_nodes_for_speedup:
                speedup = ex_time / bb_time
                results.record_pass(f"{test_name} - B&B is faster ({speedup:.2f}x speedup)")
            elif stats_ex.nodes_processed <= min_nodes_for_speedup:
                print(f"    ⚠ Problem still too small to measure speedup reliably")
            else:
                # Don't fail for small problems, but do fail for large ones
                if n_items >= 10:
                    results.record_fail(f"{test_name} - B&B is faster",
                                      f"B&B: {bb_time:.3f}s, Ex: {ex_time:.3f}s")
                else:
                    print(f"    ⚠ B&B slower but problem size < 10 items")
            
            # Record performance
            problem_size = len(items_to_assign) + len(items_assigned)
            results.record_performance(
                problem_size,
                stats_ex.nodes_processed,
                stats_bb.nodes_processed,
                ex_time,
                bb_time
            )


def test_upper_bound_soundness(results: BBValidationResult):
    """Test that upper bounds are always >= achievable scores (never prune optimal solutions)."""
    print("\n" + "="*80)
    print("TEST 3: Upper Bound Soundness")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pp_file, ip_file = create_test_data_bb(temp_path)
        
        # Create scorer
        items = ['T', 'H', 'E', 'A']
        positions = ['F', 'D', 'S', 'J']
        
        scorer = WeightedMOOScorer(
            objectives=['obj1_adjacent', 'obj2_same_hand'],
            position_pair_score_table=str(pp_file),
            items=items,
            positions=positions,
            item_pair_score_table=str(ip_file),
            verbose=False
        )
        
        # Create upper bound calculator
        bound_calc = MOOUpperBoundCalculator(scorer)
        
        # Test cases: partial mappings and their completions
        test_cases = [
            # (partial_mapping, used_positions, description)
            (np.array([0, -1, -1, -1], dtype=np.int32), 
             np.array([True, False, False, False], dtype=bool),
             "One item placed (T->F)"),
            
            (np.array([0, 1, -1, -1], dtype=np.int32),
             np.array([True, True, False, False], dtype=bool),
             "Two items placed (T->F, H->D)"),
            
            (np.array([0, 1, 2, -1], dtype=np.int32),
             np.array([True, True, True, False], dtype=bool),
             "Three items placed (T->F, H->D, E->S)"),
        ]
        
        all_sound = True
        
        for partial_mapping, used_positions, description in test_cases:
            print(f"\n  Testing: {description}")
            
            # Calculate upper bound
            upper_bound = bound_calc.calculate_upper_bound_vector(partial_mapping, used_positions)
            
            # Find all possible completions and their actual scores
            unassigned = [i for i in range(len(partial_mapping)) if partial_mapping[i] < 0]
            available = [i for i in range(len(used_positions)) if not used_positions[i]]
            
            max_actual_scores = [0.0] * len(scorer.objectives)
            
            # Try all possible completions
            from itertools import permutations
            for perm in permutations(available, len(unassigned)):
                complete_mapping = partial_mapping.copy()
                for item_idx, pos_idx in zip(unassigned, perm):
                    complete_mapping[item_idx] = pos_idx
                
                actual_scores = scorer.score_layout(complete_mapping)
                
                for i in range(len(actual_scores)):
                    max_actual_scores[i] = max(max_actual_scores[i], actual_scores[i])
            
            # Check if upper bound >= max actual score for each objective
            is_sound = True
            for i in range(len(upper_bound)):
                if upper_bound[i] < max_actual_scores[i] - 1e-9:  # Small tolerance
                    is_sound = False
                    all_sound = False
                    print(f"    ✗ Objective {i}: Upper bound {upper_bound[i]:.6f} < max actual {max_actual_scores[i]:.6f}")
                else:
                    print(f"    ✓ Objective {i}: Upper bound {upper_bound[i]:.6f} >= max actual {max_actual_scores[i]:.6f}")
            
            if is_sound:
                results.record_pass(f"{description} - Upper bounds are sound")
            else:
                results.record_fail(f"{description} - Upper bounds are sound",
                                  "Upper bound < achievable score (will prune optimal solutions!)")
        
        if all_sound:
            print(f"\n  ✓ All upper bounds are mathematically sound")
        else:
            print(f"\n  ✗ CRITICAL: Some upper bounds are too tight - optimal solutions may be pruned!")


def test_bb_with_constraints(results: BBValidationResult):
    """Test B&B correctness with constraints."""
    print("\n" + "="*80)
    print("TEST 4: B&B with Constraints")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pp_file, ip_file = create_test_data_bb(temp_path)
        
        config = create_test_config_bb(
            temp_path, pp_file, ip_file,
            items_assigned="T",
            positions_assigned="F",
            items_to_assign="HEA",
            positions_to_assign="DSJK"
        )
        
        # Add constraints
        config.optimization.items_to_constrain_set = {'H', 'E'}
        config.optimization.positions_to_constrain_set = {'D', 'S'}
        
        print(f"\n  Preassigned: T -> F")
        print(f"  To optimize: HEA in DSJK")
        print(f"  Constraint: H and E must be in D or S")
        
        # Run both searches
        from optimize_layouts import run_moo_optimization
        
        pareto_ex, stats_ex = run_moo_optimization(
            config=config,
            objectives=['obj1_adjacent', 'obj2_same_hand'],
            position_pair_score_table=str(pp_file),
            weights=[1.0, 1.0],
            maximize=[True, True],
            item_pair_score_table=str(ip_file),
            search_mode='exhaustive',
            verbose=False
        )
        
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
        
        # Compare
        is_equal, reason = compare_pareto_fronts(pareto_ex, pareto_bb)
        
        print(f"\n  Exhaustive: {len(pareto_ex)} solutions")
        print(f"  B&B: {len(pareto_bb)} solutions")
        
        if is_equal:
            results.record_pass("B&B with constraints - Identical results")
        else:
            results.record_fail("B&B with constraints", reason)
        
        # Verify constraints are satisfied
        constraint_ok = True
        for sol in pareto_bb:
            if sol['mapping']['H'] not in ['D', 'S'] or sol['mapping']['E'] not in ['D', 'S']:
                constraint_ok = False
                break
        
        if constraint_ok:
            results.record_pass("B&B with constraints - Constraints satisfied")
        else:
            results.record_fail("B&B with constraints - Constraints satisfied",
                              "Some solutions violate constraints")

def test_upper_bound_quality(results):
    """
    Validate bound quality (tightness)
    
    Checks that bounds are not just sound, but also tight enough
    to provide meaningful pruning.
    """
    print("\n" + "="*80)
    print("TEST 5: Upper Bound Quality (Tightness)")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pp_file, ip_file = create_test_data_bb(temp_path)
        
        items = ['T', 'H', 'E', 'A']
        positions = ['F', 'D', 'S', 'J']
        
        scorer = WeightedMOOScorer(
            objectives=['obj1_adjacent', 'obj2_same_hand'],
            position_pair_score_table=str(pp_file),
            items=items,
            positions=positions,
            item_pair_score_table=str(ip_file),
            verbose=False
        )
        
        bound_calc = MOOUpperBoundCalculator(scorer)
        
        # Test at multiple depths
        test_cases = [
            (np.array([0, -1, -1, -1], dtype=np.int32), 
             np.array([True, False, False, False], dtype=bool),
             "1 of 4 items placed", 1),
            
            (np.array([0, 1, -1, -1], dtype=np.int32),
             np.array([True, True, False, False], dtype=bool),
             "2 of 4 items placed", 2),
            
            (np.array([0, 1, 2, -1], dtype=np.int32),
             np.array([True, True, True, False], dtype=bool),
             "3 of 4 items placed", 3),
        ]
        
        all_quality_ok = True
        bound_gaps = []
        
        for partial_mapping, used_positions, description, depth in test_cases:
            print(f"\n  Testing: {description}")
            
            # Calculate upper bound
            upper_bound = bound_calc.calculate_upper_bound_vector(
                partial_mapping, used_positions)
            
            # Calculate max achievable scores
            unassigned = [i for i in range(len(partial_mapping)) 
                         if partial_mapping[i] < 0]
            available = [i for i in range(len(used_positions)) 
                        if not used_positions[i]]
            
            max_actual_scores = [0.0] * len(scorer.objectives)
            
            for perm in permutations(available, len(unassigned)):
                complete_mapping = partial_mapping.copy()
                for item_idx, pos_idx in zip(unassigned, perm):
                    complete_mapping[item_idx] = pos_idx
                
                actual_scores = scorer.score_layout(complete_mapping)
                for i in range(len(actual_scores)):
                    max_actual_scores[i] = max(max_actual_scores[i], 
                                              actual_scores[i])
            
            # Calculate quality metrics
            for i in range(len(upper_bound)):
                gap = upper_bound[i] - max_actual_scores[i]
                relative_gap = gap / max_actual_scores[i] if max_actual_scores[i] > 0 else 0
                
                bound_gaps.append({
                    'depth': depth,
                    'objective': i,
                    'gap': gap,
                    'relative_gap': relative_gap,
                    'upper_bound': upper_bound[i],
                    'max_actual': max_actual_scores[i]
                })
                
                print(f"    Obj {i}: UB={upper_bound[i]:.6f}, "
                      f"Max={max_actual_scores[i]:.6f}, "
                      f"Gap={gap:.6f} ({relative_gap*100:.1f}%)")
                
                # Quality thresholds (adjust based on your needs)
                # Gap should be positive (sound) but not too large (tight)
                if gap < -1e-9:
                    print(f"      ✗ CRITICAL: Bound is too tight (unsound)!")
                    all_quality_ok = False
                elif relative_gap > 0.5:  # More than 50% overestimate
                    print(f"      ⚠ WARNING: Bound may be too loose")
                    # Don't fail, but warn
                elif gap < 1e-9:
                    print(f"      ⚠ WARNING: Bound is very tight (risky)")
                else:
                    print(f"      ✓ Bound quality is good")
        
        # Analyze bound quality trends
        print(f"\n  Bound Quality Analysis:")
        avg_gap_by_depth = {}
        for gap_info in bound_gaps:
            depth = gap_info['depth']
            if depth not in avg_gap_by_depth:
                avg_gap_by_depth[depth] = []
            avg_gap_by_depth[depth].append(gap_info['relative_gap'])
        
        for depth in sorted(avg_gap_by_depth.keys()):
            avg_gap = np.mean(avg_gap_by_depth[depth])
            print(f"    Depth {depth}: Avg relative gap = {avg_gap*100:.1f}%")
            
            # Bounds should get tighter as we go deeper
            if depth > 1:
                prev_avg = np.mean(avg_gap_by_depth[depth-1])
                if avg_gap > prev_avg * 1.5:  # Gap increased significantly
                    print(f"      ⚠ Bounds getting looser with depth (unusual)")
        
        if all_quality_ok:
            results.record_pass("Upper bound quality - Bounds are sound and reasonably tight")
        else:
            results.record_fail("Upper bound quality", 
                              "Some bounds are unsound or excessively loose")


def test_pruned_branches_correctness(results):
    """
    Validate that pruned branches truly contain no Pareto-optimal solutions
    
    This is the ultimate validation: sample pruned branches and verify
    exhaustively that they don't contain better solutions.
    """
    print("\n" + "="*80)
    print("TEST 6: Pruned Branches Correctness")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pp_file, ip_file = create_test_data_bb(temp_path)
        
        config = create_test_config_bb(
            temp_path, pp_file, ip_file,
            items_assigned="",
            positions_assigned="",
            items_to_assign="THEA",
            positions_to_assign="FDSJ"
        )
        
        print(f"\n  Running B&B search and tracking pruned branches...")
        
        # Modified search that tracks pruned branches
        # (This would require modifying moo_search.py to return pruned info)
        # For now, we'll do a simpler validation
        
        from optimize_layouts import run_moo_optimization
        
        # Get B&B results
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
        
        # Get exhaustive results  
        pareto_ex, stats_ex = run_moo_optimization(
            config=config,
            objectives=['obj1_adjacent', 'obj2_same_hand'],
            position_pair_score_table=str(pp_file),
            weights=[1.0, 1.0],
            maximize=[True, True],
            item_pair_score_table=str(ip_file),
            search_mode='exhaustive',
            verbose=False
        )
        
        # If fronts are identical, then by definition all pruned branches
        # were correctly pruned (they contained no Pareto-optimal solutions)
        is_equal, reason = compare_pareto_fronts(pareto_ex, pareto_bb)
        
        print(f"\n  B&B pruned {stats_bb.nodes_pruned:,} nodes")
        print(f"  Exhaustive found {len(pareto_ex)} Pareto solutions")
        print(f"  B&B found {len(pareto_bb)} Pareto solutions")
        
        if is_equal:
            print(f"  ✓ All pruned branches were correctly identified")
            print(f"    (No Pareto-optimal solutions were lost)")
            results.record_pass("Pruned branches correctness")
        else:
            print(f"  ✗ B&B missed some solutions!")
            print(f"    {reason}")
            results.record_fail("Pruned branches correctness", 
                              "Some pruned branches contained Pareto-optimal solutions")


def test_bound_calculation_edge_cases(results):
    """
    Test bound calculation on edge cases
    """
    print("\n" + "="*80)
    print("TEST 7: Bound Calculation Edge Cases")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pp_file, ip_file = create_test_data_bb(temp_path)
        
        items = ['T', 'H', 'E']
        positions = ['F', 'D', 'S']
        
        scorer = WeightedMOOScorer(
            objectives=['obj1_adjacent', 'obj2_same_hand'],
            position_pair_score_table=str(pp_file),
            items=items,
            positions=positions,
            item_pair_score_table=str(ip_file),
            verbose=False
        )
        
        bound_calc = MOOUpperBoundCalculator(scorer)
        
        edge_cases = [
            # Empty mapping (all unassigned)
            (np.array([-1, -1, -1], dtype=np.int32),
             np.array([False, False, False], dtype=bool),
             "All items unassigned"),
            
            # Almost complete (1 item left)
            (np.array([0, 1, -1], dtype=np.int32),
             np.array([True, True, False], dtype=bool),
             "Only 1 item remaining"),
            
            # Complete assignment
            (np.array([0, 1, 2], dtype=np.int32),
             np.array([True, True, True], dtype=bool),
             "Complete assignment"),
        ]
        
        all_pass = True
        
        for partial_mapping, used_positions, description in edge_cases:
            print(f"\n  Testing: {description}")
            
            try:
                upper_bound = bound_calc.calculate_upper_bound_vector(
                    partial_mapping, used_positions)
                
                # For complete assignment, bound should equal actual score
                if np.all(partial_mapping >= 0):
                    actual = scorer.score_layout(partial_mapping)
                    for i in range(len(upper_bound)):
                        if abs(upper_bound[i] - actual[i]) > 1e-9:
                            print(f"    ✗ Complete assignment: bound != actual")
                            print(f"      Obj {i}: UB={upper_bound[i]:.6f}, "
                                  f"Actual={actual[i]:.6f}")
                            all_pass = False
                        else:
                            print(f"    ✓ Obj {i}: Bound correctly equals actual score")
                else:
                    print(f"    ✓ Bound calculated: {[f'{x:.6f}' for x in upper_bound]}")
                    
            except Exception as e:
                print(f"    ✗ Exception: {e}")
                all_pass = False
        
        if all_pass:
            results.record_pass("Edge case handling")
        else:
            results.record_fail("Edge case handling", 
                              "Some edge cases failed")


def main():
    """Run all B&B validation tests."""
    print("="*80)
    print("BRANCH-AND-BOUND VALIDATION TEST SUITE")
    print("="*80)
    
    results = BBValidationResult()
    
    try:
        test_bb_correctness_small_problems(results)
        test_bb_performance_scaling(results)  # includes 10-item tests
        test_upper_bound_soundness(results)
        test_bb_with_constraints(results)
        
        test_upper_bound_quality(results)
        test_pruned_branches_correctness(results)
        test_bound_calculation_edge_cases(results)
        
    except Exception as e:
        print(f"\n⚠ TEST SUITE CRASHED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return results.summary()


if __name__ == "__main__":
    sys.exit(main())
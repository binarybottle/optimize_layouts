# slurm/slurm_setup_test_scoring_consistency.py
"""
Validation script to ensure all scoring methods produce identical results.
"""

import numpy as np
from typing import Dict, Tuple
from config import load_config
from scoring import (LayoutScorer, prepare_scoring_arrays, 
                    calculate_complete_layout_score, apply_default_combination)
from optimize_layout import load_normalized_scores

def test_scoring_consistency(config_path: str = "config.yaml") -> bool:
    """
    Test that all scoring methods produce identical results.
    
    Returns:
        True if all methods are consistent, False otherwise
    """
    print("🧪 Testing scoring consistency across all methods...")
    
    # Load config and data
    config = load_config(config_path)
    normalized_scores = load_normalized_scores(config)
    
    # Test parameters
    items_to_test = list(config.optimization.items_to_assign)[:3]  # Test with first 3 items
    positions_to_test = list(config.optimization.positions_to_assign)[:3]
    
    print(f"Testing with items: {items_to_test}")
    print(f"Testing with positions: {positions_to_test}")
    
    # Create test mapping
    test_mapping = dict(zip(items_to_test, positions_to_test))
    test_mapping_array = np.array([0, 1, 2], dtype=np.int32)  # items[0]->pos[0], etc.
    
    print(f"Test mapping: {test_mapping}")
    
    # Method 1: LayoutScorer (Combined mode)
    arrays = prepare_scoring_arrays(
        items_to_assign=items_to_test,
        positions_to_assign=positions_to_test,
        norm_item_scores=normalized_scores[0],
        norm_item_pair_scores=normalized_scores[1],
        norm_position_scores=normalized_scores[2],
        norm_position_pair_scores=normalized_scores[3]
    )
    
    scorer_combined = LayoutScorer(arrays, mode='combined')
    scorer_combined_result = scorer_combined.score_layout(test_mapping_array)
    scorer_components = scorer_combined.get_components(test_mapping_array)
    
    print(f"\nMethod 1 - LayoutScorer (Combined):")
    print(f"  Total score: {scorer_combined_result:.12f}")
    print(f"  Item component: {scorer_components.item_score:.12f}")
    print(f"  Pair component: {scorer_components.item_pair_score:.12f}")
    
    # Method 2: LayoutScorer (Multi-objective mode)
    scorer_moo = LayoutScorer(arrays, mode='multi_objective')
    scorer_moo_result = scorer_moo.score_layout(test_mapping_array)
    
    print(f"\nMethod 2 - LayoutScorer (Multi-objective):")
    print(f"  Objectives: {scorer_moo_result}")
    print(f"  Combined (manual): {apply_default_combination(scorer_moo_result[0], scorer_moo_result[1]):.12f}")
    
    # Method 3: Complete layout scoring
    complete_total, complete_item, complete_pair = calculate_complete_layout_score(
        test_mapping, normalized_scores
    )
    
    print(f"\nMethod 3 - Complete Layout Scoring:")
    print(f"  Total score: {complete_total:.12f}")
    print(f"  Item component: {complete_item:.12f}")
    print(f"  Pair component: {complete_pair:.12f}")
    
    # Method 4: Manual calculation using apply_default_combination
    manual_combined = apply_default_combination(scorer_components.item_score, scorer_components.item_pair_score)
    
    print(f"\nMethod 4 - Manual Combination:")
    print(f"  Total score: {manual_combined:.12f}")
    
    # Validation checks
    print(f"\n🔍 Validation Results:")
    
    # Check 1: Combined scorer vs complete layout
    check1 = abs(scorer_combined_result - complete_total) < 1e-10
    print(f"  Combined scorer vs Complete layout: {'✅ PASS' if check1 else '❌ FAIL'} (diff: {abs(scorer_combined_result - complete_total):.2e})")
    
    # Check 2: MOO objectives vs individual components
    check2a = abs(scorer_moo_result[0] - scorer_components.item_score) < 1e-10
    check2b = abs(scorer_moo_result[1] - scorer_components.item_pair_score) < 1e-10
    print(f"  MOO objectives vs components: {'✅ PASS' if check2a and check2b else '❌ FAIL'}")
    
    # Check 3: Manual combination vs total
    check3 = abs(manual_combined - scorer_combined_result) < 1e-10
    print(f"  Manual combination vs Combined scorer: {'✅ PASS' if check3 else '❌ FAIL'} (diff: {abs(manual_combined - scorer_combined_result):.2e})")
    
    # Check 4: Components consistency
    check4a = abs(scorer_components.item_score - complete_item) < 1e-10
    check4b = abs(scorer_components.item_pair_score - complete_pair) < 1e-10
    print(f"  Component consistency: {'✅ PASS' if check4a and check4b else '❌ FAIL'}")
    
    # Check 5: Total consistency via .total() method
    check5 = abs(scorer_components.total() - scorer_combined_result) < 1e-10
    print(f"  ScoreComponents.total() consistency: {'✅ PASS' if check5 else '❌ FAIL'}")
    
    all_checks_pass = all([check1, check2a, check2b, check3, check4a, check4b, check5])
    
    if all_checks_pass:
        print(f"\n✅ ALL CHECKS PASSED - Scoring is consistent across all methods!")
    else:
        print(f"\n❌ SOME CHECKS FAILED - Scoring inconsistencies detected!")
        print(f"\nDebugging info:")
        print(f"  Combination strategy: {apply_default_combination.__name__}")
        print(f"  Arrays shapes: item_scores={arrays.item_scores.shape}, position_matrix={arrays.position_matrix.shape}")
    
    return all_checks_pass

def test_scale_consistency():
    """Test that scores are in reasonable ranges."""
    print(f"\n📏 Testing score scale consistency...")
    
    config = load_config()
    normalized_scores = load_normalized_scores(config)
    
    # Check that all normalized scores are in [0,1] range
    def check_range(scores_dict, name):
        values = list(scores_dict.values())
        min_val, max_val = min(values), max(values)
        in_range = 0 <= min_val <= max_val <= 1
        print(f"  {name}: [{min_val:.6f}, {max_val:.6f}] {'✅' if in_range else '❌'}")
        return in_range
    
    checks = []
    checks.append(check_range(normalized_scores[0], "Item scores"))
    checks.append(check_range(normalized_scores[1], "Item pair scores"))
    checks.append(check_range(normalized_scores[2], "Position scores"))
    checks.append(check_range(normalized_scores[3], "Position pair scores"))
    
    if all(checks):
        print(f"  ✅ All scores are properly normalized to [0,1] range")
        return True
    else:
        print(f"  ❌ Some scores are outside [0,1] range - check normalization!")
        return False

if __name__ == "__main__":
    print("🔬 Scoring Consistency Validation")
    print("=" * 50)
    
    # Test basic consistency
    consistency_ok = test_scoring_consistency()
    
    # Test scale consistency
    scale_ok = test_scale_consistency()
    
    print(f"\n" + "=" * 50)
    if consistency_ok and scale_ok:
        print(f"🎉 ALL TESTS PASSED - Your scoring system is consistent!")
        print(f"   You can now safely use all optimization methods.")
    else:
        print(f"⚠️  SOME TESTS FAILED - Fix the issues before running optimizations.")
        print(f"   Check the output above for specific problems.")
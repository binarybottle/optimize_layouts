# moo_analysis.py
"""
Multi-Objective Optimization Problem Analysis and Diagnostics

Analyzes whether your optimization problem has good multi-objective characteristics:
- Objective space diversity and ranges
- Pareto front quality and size  
- Trade-off meaningfulness
- Problem difficulty assessment
- Worker function validation

Usage:
    python moo_analysis.py --config config.yaml
    python moo_analysis.py --config config.yaml --quick
    python moo_analysis.py --config config.yaml --test-size 6 --sample-size 500
"""

import numpy as np
import random
import time
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from itertools import permutations
import math

from config import Config, load_config
from scoring import LayoutScorer, prepare_scoring_arrays, load_normalized_scores

@dataclass
class MOOAnalysisResult:
    """Results from MOO problem analysis."""
    problem_size: Tuple[int, int]  # (n_items, n_positions)
    sample_size: int
    total_solutions: int
    unique_objectives: int
    duplicate_rate: float
    
    # Objective ranges and statistics
    obj1_range: Tuple[float, float]
    obj2_range: Tuple[float, float]
    obj1_std: float
    obj2_std: float
    
    # Pareto front analysis
    pareto_size: int
    pareto_percentage: float
    pareto_solutions: List[Dict]
    
    # Problem quality assessment
    objective_diversity_score: float  # 0-1, higher is better
    trade_off_quality_score: float    # 0-1, higher is better
    moo_suitability_score: float      # 0-1, higher is better
    
    # Issues and recommendations
    issues: List[str]
    recommendations: List[str]
    
    # Worker function test results
    worker_test_passed: bool
    worker_pareto_size: int
    worker_error: Optional[str]

def analyze_moo_problem_quality(config: Config, 
                               test_size: Optional[int] = None,
                               sample_size: int = 200,
                               verbose: bool = True) -> MOOAnalysisResult:
    """
    Comprehensive analysis of MOO problem characteristics.
    
    Args:
        config: Configuration object
        test_size: Number of items to test with (None = auto-detect optimal size)
        sample_size: Number of random solutions to generate for analysis
        verbose: Whether to print detailed analysis
        
    Returns:
        MOOAnalysisResult with comprehensive analysis
    """
    if verbose:
        print("🔬 MOO Problem Quality Analysis")
        print("=" * 50)
    
    # Auto-detect optimal test size
    total_items = len(config.optimization.items_to_assign)
    if test_size is None:
        if total_items <= 6:
            test_size = total_items
        elif total_items <= 10:
            test_size = 6
        else:
            test_size = 7  # Balance between realism and performance
    
    test_size = min(test_size, total_items)
    
    # Setup test problem
    items = list(config.optimization.items_to_assign)[:test_size]
    positions = list(config.optimization.positions_to_assign)[:test_size]
    
    if verbose:
        print(f"📋 Test Configuration:")
        print(f"   Items ({len(items)}): {items}")
        print(f"   Positions ({len(positions)}): {positions}")
        print(f"   Sample size: {sample_size}")
        print(f"   Total permutations: {math.factorial(len(items)):,}")
    
    # Load data and create scorer
    normalized_scores = load_normalized_scores(config)
    arrays = prepare_scoring_arrays(
        items_to_assign=items,
        positions_to_assign=positions,
        norm_item_scores=normalized_scores[0],
        norm_item_pair_scores=normalized_scores[1],
        norm_position_scores=normalized_scores[2],
        norm_position_pair_scores=normalized_scores[3]
    )
    
    scorer = LayoutScorer(arrays, mode='multi_objective')
    
    # Generate test solutions
    solutions = _generate_test_solutions(scorer, items, positions, sample_size, verbose)
    
    # Analyze the solutions
    objective_analysis = _analyze_objectives(solutions, verbose)
    pareto_analysis = _compute_pareto_analysis(solutions, verbose)
    quality_scores = _assess_problem_quality(objective_analysis, pareto_analysis, verbose)
    issues_and_recs = _generate_recommendations(objective_analysis, pareto_analysis, quality_scores)
    worker_test = _test_worker_function(scorer, items, positions, sample_size, verbose)
    
    # Create comprehensive result
    result = MOOAnalysisResult(
        problem_size=(len(items), len(positions)),
        sample_size=len(solutions),
        total_solutions=len(solutions),
        unique_objectives=objective_analysis['unique_count'],
        duplicate_rate=objective_analysis['duplicate_rate'],
        
        obj1_range=objective_analysis['obj1_range'],
        obj2_range=objective_analysis['obj2_range'],
        obj1_std=objective_analysis['obj1_std'],
        obj2_std=objective_analysis['obj2_std'],
        
        pareto_size=pareto_analysis['pareto_size'],
        pareto_percentage=pareto_analysis['pareto_percentage'],
        pareto_solutions=pareto_analysis['pareto_solutions'][:5],  # First 5 for display
        
        objective_diversity_score=quality_scores['diversity'],
        trade_off_quality_score=quality_scores['trade_off'],
        moo_suitability_score=quality_scores['overall'],
        
        issues=issues_and_recs['issues'],
        recommendations=issues_and_recs['recommendations'],
        
        worker_test_passed=worker_test['passed'],
        worker_pareto_size=worker_test['pareto_size'],
        worker_error=worker_test['error']
    )
    
    if verbose:
        _print_analysis_summary(result)
    
    return result

def _generate_test_solutions(scorer: LayoutScorer, items: List[str], positions: List[str], 
                           sample_size: int, verbose: bool) -> List[Dict]:
    """Generate random test solutions for analysis."""
    if verbose:
        print(f"\n📊 Generating test solutions...")
    
    solutions = []
    
    # Get all permutations and sample randomly
    all_perms = list(permutations(range(len(positions)), len(items)))
    actual_sample_size = min(sample_size, len(all_perms))
    
    # Set seed for reproducible results
    random.seed(42)
    sample_perms = random.sample(all_perms, actual_sample_size)
    
    for i, perm in enumerate(sample_perms):
        mapping_array = np.array(perm, dtype=np.int32)
        objectives = scorer.score_layout(mapping_array)
        
        # Create item mapping for display
        item_mapping = {items[j]: positions[perm[j]] for j in range(len(items))}
        
        solution = {
            'mapping': item_mapping,
            'objectives': objectives,
            'index': i
        }
        solutions.append(solution)
    
    if verbose:
        print(f"   Generated {len(solutions)} solutions")
    
    return solutions

def _analyze_objectives(solutions: List[Dict], verbose: bool) -> Dict:
    """Analyze objective space characteristics."""
    if verbose:
        print(f"\n🔬 Objective Space Analysis:")
    
    # Extract objective values
    objective_tuples = [tuple(sol['objectives']) for sol in solutions]
    unique_objectives = set(objective_tuples)
    
    obj1_values = [sol['objectives'][0] for sol in solutions]
    obj2_values = [sol['objectives'][1] for sol in solutions]
    
    duplicate_rate = (len(solutions) - len(unique_objectives)) / len(solutions)
    
    analysis = {
        'unique_count': len(unique_objectives),
        'duplicate_rate': duplicate_rate,
        'obj1_range': (min(obj1_values), max(obj1_values)),
        'obj2_range': (min(obj2_values), max(obj2_values)),
        'obj1_std': np.std(obj1_values),
        'obj2_std': np.std(obj2_values),
        'obj1_values': obj1_values,
        'obj2_values': obj2_values
    }
    
    if verbose:
        print(f"   Total solutions: {len(solutions)}")
        print(f"   Unique objective vectors: {len(unique_objectives)}")
        print(f"   Duplicate rate: {duplicate_rate*100:.1f}%")
        print(f"   Objective 1 range: [{analysis['obj1_range'][0]:.6f}, {analysis['obj1_range'][1]:.6f}]")
        print(f"   Objective 2 range: [{analysis['obj2_range'][0]:.6f}, {analysis['obj2_range'][1]:.6f}]")
        print(f"   Objective 1 std dev: {analysis['obj1_std']:.6f}")
        print(f"   Objective 2 std dev: {analysis['obj2_std']:.6f}")
        
        # Check for extremely narrow ranges
        obj1_range_size = analysis['obj1_range'][1] - analysis['obj1_range'][0]
        obj2_range_size = analysis['obj2_range'][1] - analysis['obj2_range'][0]
        
        if obj1_range_size < 1e-6:
            print(f"   ⚠️  Objective 1 values are extremely close (range: {obj1_range_size:.2e})")
        if obj2_range_size < 1e-6:
            print(f"   ⚠️  Objective 2 values are extremely close (range: {obj2_range_size:.2e})")
    
    return analysis

def _compute_pareto_analysis(solutions: List[Dict], verbose: bool) -> Dict:
    """Compute and analyze Pareto front."""
    if verbose:
        print(f"\n🎯 Pareto Front Analysis:")
    
    def dominates(obj1, obj2):
        """Check if obj1 dominates obj2."""
        better_in_at_least_one = False
        for i in range(len(obj1)):
            if obj1[i] < obj2[i]:
                return False
            elif obj1[i] > obj2[i]:
                better_in_at_least_one = True
        return better_in_at_least_one
    
    # Compute Pareto front
    pareto_front = []
    for candidate in solutions:
        is_dominated = False
        candidate_obj = candidate['objectives']
        
        for other in solutions:
            if other != candidate:
                other_obj = other['objectives']
                if dominates(other_obj, candidate_obj):
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto_front.append(candidate)
    
    pareto_percentage = len(pareto_front) / len(solutions) if solutions else 0
    
    analysis = {
        'pareto_size': len(pareto_front),
        'pareto_percentage': pareto_percentage,
        'pareto_solutions': pareto_front
    }
    
    if verbose:
        print(f"   Pareto front size: {len(pareto_front)}")
        print(f"   Pareto percentage: {pareto_percentage*100:.1f}%")
        
        if len(pareto_front) == 1:
            print(f"   ❌ SINGLE PARETO SOLUTION DETECTED!")
            winner = pareto_front[0]
            print(f"      Winner: [{winner['objectives'][0]:.8f}, {winner['objectives'][1]:.8f}]")
        elif len(pareto_front) <= 5:
            print(f"   Pareto solutions:")
            for i, sol in enumerate(pareto_front):
                obj = sol['objectives']
                print(f"      {i+1}: [{obj[0]:.6f}, {obj[1]:.6f}]")
        elif len(pareto_front) > 0:
            print(f"   First 3 Pareto solutions:")
            for i, sol in enumerate(pareto_front[:3]):
                obj = sol['objectives']
                print(f"      {i+1}: [{obj[0]:.6f}, {obj[1]:.6f}]")
    
    return analysis

def _assess_problem_quality(obj_analysis: Dict, pareto_analysis: Dict, verbose: bool) -> Dict:
    """Assess overall problem quality for MOO."""
    if verbose:
        print(f"\n📈 Problem Quality Assessment:")
    
    # Objective diversity score (based on standard deviations and ranges)
    obj1_range_size = obj_analysis['obj1_range'][1] - obj_analysis['obj1_range'][0]
    obj2_range_size = obj_analysis['obj2_range'][1] - obj_analysis['obj2_range'][0]
    
    # Normalize by typical score ranges (assuming 0-1 normalized scores)
    diversity_score = min(1.0, (obj1_range_size + obj2_range_size) / 0.2)  # 0.1 range each = good
    
    # Trade-off quality (Pareto percentage should be 5-30% for good trade-offs)
    pareto_pct = pareto_analysis['pareto_percentage']
    if 0.05 <= pareto_pct <= 0.30:
        trade_off_score = 1.0
    elif pareto_pct < 0.05:
        trade_off_score = max(0.0, pareto_pct / 0.05)  # Too few solutions are optimal
    else:
        trade_off_score = max(0.0, (1.0 - pareto_pct) / 0.7)  # Too many are optimal
    
    # Overall suitability (combination of factors)
    duplicate_penalty = max(0.0, 1.0 - obj_analysis['duplicate_rate'] * 2)  # Penalize duplicates
    overall_score = (diversity_score * 0.4 + trade_off_score * 0.4 + duplicate_penalty * 0.2)
    
    scores = {
        'diversity': diversity_score,
        'trade_off': trade_off_score,
        'overall': overall_score
    }
    
    if verbose:
        print(f"   Objective diversity score: {diversity_score:.2f}/1.0")
        print(f"   Trade-off quality score: {trade_off_score:.2f}/1.0")
        print(f"   Overall MOO suitability: {overall_score:.2f}/1.0")
    
    return scores

def _generate_recommendations(obj_analysis: Dict, pareto_analysis: Dict, quality_scores: Dict) -> Dict:
    """Generate issues and recommendations based on analysis."""
    issues = []
    recommendations = []
    
    # Check for narrow objective ranges
    obj1_range_size = obj_analysis['obj1_range'][1] - obj_analysis['obj1_range'][0]
    obj2_range_size = obj_analysis['obj2_range'][1] - obj_analysis['obj2_range'][0]
    
    if obj1_range_size < 0.01:
        issues.append(f"Objective 1 has very narrow range ({obj1_range_size:.6f})")
        recommendations.append("Consider adjusting item scoring to create more diversity")
    
    if obj2_range_size < 0.01:
        issues.append(f"Objective 2 has very narrow range ({obj2_range_size:.6f})")
        recommendations.append("Consider adjusting pair scoring to create more diversity")
    
    # Check Pareto front size
    pareto_pct = pareto_analysis['pareto_percentage']
    if pareto_pct < 0.02:
        issues.append("Very few solutions are Pareto-optimal (single winner problem)")
        recommendations.append("Objectives may be too aligned - consider conflicting scoring strategies")
    elif pareto_pct > 0.5:
        issues.append("Too many solutions are Pareto-optimal (weak trade-offs)")
        recommendations.append("Objectives may not conflict enough - strengthen trade-off mechanisms")
    
    # Check duplicate rate
    if obj_analysis['duplicate_rate'] > 0.1:
        issues.append(f"High duplicate rate ({obj_analysis['duplicate_rate']*100:.1f}%)")
        recommendations.append("Scoring function may need higher precision or more granularity")
    
    # Overall quality assessment
    if quality_scores['overall'] < 0.3:
        issues.append("Overall MOO suitability is low")
        recommendations.append("Consider redesigning scoring functions for better multi-objective characteristics")
    
    return {'issues': issues, 'recommendations': recommendations}

def _test_worker_function(scorer: LayoutScorer, items: List[str], positions: List[str], 
                        sample_size: int, verbose: bool) -> Dict:
    """Test the parallel worker function."""
    if verbose:
        print(f"\n🧪 Worker Function Test:")
    
    try:
        # Create a simple test worker function inline
        def test_worker_function(args):
            chunk_mappings, scorer_arrays, items_list, positions_list = args
            
            # Recreate scorer
            test_scorer = LayoutScorer(scorer_arrays, mode='multi_objective')
            
            chunk_solutions = []
            for mapping_array in chunk_mappings:
                objectives = test_scorer.score_layout(mapping_array)
                item_mapping = {items_list[i]: positions_list[mapping_array[i]] 
                               for i in range(len(mapping_array))}
                
                solution = {
                    'mapping': item_mapping,
                    'objectives': objectives
                }
                chunk_solutions.append(solution)
            
            # Simple Pareto front computation
            def compute_pareto_front(solutions):
                def dominates(obj1, obj2):
                    better_in_at_least_one = False
                    for i in range(len(obj1)):
                        if obj1[i] < obj2[i]:
                            return False
                        elif obj1[i] > obj2[i]:
                            better_in_at_least_one = True
                    return better_in_at_least_one
                
                pareto_front = []
                for candidate in solutions:
                    is_dominated = False
                    candidate_obj = candidate['objectives']
                    
                    for pareto_solution in pareto_front:
                        pareto_obj = pareto_solution['objectives']
                        if dominates(pareto_obj, candidate_obj):
                            is_dominated = True
                            break
                    
                    if not is_dominated:
                        pareto_front = [sol for sol in pareto_front 
                                       if not dominates(candidate_obj, sol['objectives'])]
                        pareto_front.append(candidate)
                
                return pareto_front
            
            local_pareto = compute_pareto_front(chunk_solutions)
            return local_pareto, len(chunk_mappings), 0
        
        # Test with subset of solutions
        test_size = min(20, sample_size)
        test_mappings = []
        for i in range(test_size):
            perm = np.random.permutation(len(positions))[:len(items)]
            test_mappings.append(np.array(perm, dtype=np.int32))
        
        # Call test worker function
        worker_result = test_worker_function((test_mappings, scorer.arrays, items, positions))
        worker_pareto, worker_nodes, worker_pruned = worker_result
        
        result = {
            'passed': True,
            'pareto_size': len(worker_pareto),
            'nodes_processed': worker_nodes,
            'error': None
        }
        
        if verbose:
            print(f"   ✅ Worker processed {worker_nodes} nodes")
            print(f"   ✅ Worker Pareto front: {len(worker_pareto)} solutions")
            
            if len(worker_pareto) == 1:
                print(f"   ⚠️  Worker produces single solution - may indicate problem design issue")
            elif len(worker_pareto) > 0:
                print(f"   ✅ Worker produces multiple solutions")
        
        return result
        
    except Exception as e:
        result = {
            'passed': False,
            'pareto_size': 0,
            'nodes_processed': 0,
            'error': str(e)
        }
        
        if verbose:
            print(f"   ❌ Worker test failed: {e}")
        
        return result

def _print_analysis_summary(result: MOOAnalysisResult):
    """Print comprehensive analysis summary."""
    print(f"\n" + "=" * 50)
    print(f"🎯 MOO PROBLEM ANALYSIS SUMMARY")
    print(f"=" * 50)
    
    # Overall assessment
    if result.moo_suitability_score >= 0.7:
        status = "✅ EXCELLENT"
    elif result.moo_suitability_score >= 0.5:
        status = "⚠️  FAIR"
    else:
        status = "❌ POOR"
    
    print(f"Overall MOO Suitability: {status} ({result.moo_suitability_score:.2f}/1.0)")
    
    # Key metrics
    print(f"\nKey Metrics:")
    print(f"  Problem size: {result.problem_size[0]} items × {result.problem_size[1]} positions")
    print(f"  Pareto front: {result.pareto_size} solutions ({result.pareto_percentage*100:.1f}%)")
    print(f"  Objective diversity: {result.objective_diversity_score:.2f}/1.0")
    print(f"  Trade-off quality: {result.trade_off_quality_score:.2f}/1.0")
    
    # Issues and recommendations
    if result.issues:
        print(f"\n⚠️  Issues Found:")
        for issue in result.issues:
            print(f"    • {issue}")
    
    if result.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in result.recommendations:
            print(f"    • {rec}")
    
    # Worker test
    if result.worker_test_passed:
        print(f"\n✅ Worker function test passed")
        if result.worker_pareto_size == 1:
            print(f"   ⚠️  Note: Worker produces single Pareto solution")
    else:
        print(f"\n❌ Worker function test failed: {result.worker_error}")
    
    if not result.issues:
        print(f"\n🎉 No issues found - your problem is well-suited for MOO!")

def quick_moo_check(config: Config) -> str:
    """Quick MOO problem check returning simple status."""
    result = analyze_moo_problem_quality(config, test_size=4, sample_size=50, verbose=False)
    
    if result.moo_suitability_score >= 0.7:
        return "EXCELLENT"
    elif result.moo_suitability_score >= 0.5:
        return "FAIR"
    else:
        return "POOR"

def save_analysis_report(result: MOOAnalysisResult, config_path: str, output_file: str = None):
    """Save detailed analysis report to file."""
    if output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"moo_analysis_report_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        f.write("MOO Problem Analysis Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: {config_path}\n\n")
        
        f.write(f"Overall Assessment: {result.moo_suitability_score:.2f}/1.0\n")
        f.write(f"Problem Size: {result.problem_size[0]} items × {result.problem_size[1]} positions\n")
        f.write(f"Pareto Front: {result.pareto_size} solutions ({result.pareto_percentage*100:.1f}%)\n\n")
        
        f.write("Detailed Metrics:\n")
        f.write(f"  Objective Diversity: {result.objective_diversity_score:.2f}/1.0\n")
        f.write(f"  Trade-off Quality: {result.trade_off_quality_score:.2f}/1.0\n")
        f.write(f"  Objective 1 Range: [{result.obj1_range[0]:.6f}, {result.obj1_range[1]:.6f}]\n")
        f.write(f"  Objective 2 Range: [{result.obj2_range[0]:.6f}, {result.obj2_range[1]:.6f}]\n")
        f.write(f"  Duplicate Rate: {result.duplicate_rate*100:.1f}%\n\n")
        
        if result.issues:
            f.write("Issues Found:\n")
            for issue in result.issues:
                f.write(f"  • {issue}\n")
            f.write("\n")
        
        if result.recommendations:
            f.write("Recommendations:\n")
            for rec in result.recommendations:
                f.write(f"  • {rec}\n")
            f.write("\n")
        
        f.write(f"Worker Test: {'PASSED' if result.worker_test_passed else 'FAILED'}\n")
        if result.worker_error:
            f.write(f"Worker Error: {result.worker_error}\n")
    
    print(f"📄 Analysis report saved to: {output_file}")

def main():
    """Command line interface for MOO problem analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze MOO problem quality and characteristics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python moo_analysis.py --config config.yaml
  
  # Quick check only
  python moo_analysis.py --config config.yaml --quick
  
  # Detailed analysis with custom parameters
  python moo_analysis.py --config config.yaml --test-size 8 --sample-size 500
  
  # Save report to file
  python moo_analysis.py --config config.yaml --save-report analysis_results.txt
        """
    )
    
    parser.add_argument('--config', default='config.yaml', 
                       help='Configuration file path (default: config.yaml)')
    parser.add_argument('--test-size', type=int, default=None,
                       help='Number of items to test (default: auto-detect)')
    parser.add_argument('--sample-size', type=int, default=200,
                       help='Number of random solutions to analyze (default: 200)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick assessment only (faster, less detailed)')
    parser.add_argument('--save-report', type=str, default=None,
                       help='Save detailed report to file')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output (just final assessment)')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        if args.quick:
            # Quick assessment
            print("Running quick MOO problem assessment...")
            status = quick_moo_check(config)
            print(f"\n🎯 MOO Problem Quality: {status}")
            
            if status == "POOR":
                print("⚠️  Consider running detailed analysis for specific recommendations")
            elif status == "FAIR":
                print("💡 Some optimization potential - use detailed analysis for insights")
            else:
                print("✅ Problem looks well-suited for multi-objective optimization!")
        
        else:
            # Detailed analysis
            verbose = not args.quiet
            result = analyze_moo_problem_quality(
                config, 
                test_size=args.test_size, 
                sample_size=args.sample_size,
                verbose=verbose
            )
            
            # Save report if requested
            if args.save_report:
                save_analysis_report(result, args.config, args.save_report)
            
            # Return appropriate exit code
            if result.moo_suitability_score < 0.3:
                exit(1)  # Poor suitability
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please check that the configuration file and input files exist.")
        exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()